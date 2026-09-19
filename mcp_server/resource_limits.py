"""Process-local resource guards; deliberately not a filesystem/network sandbox."""

from __future__ import annotations

import os
import subprocess
import threading

MEMORY_LIMIT_BYTES = 1024 * 1024 * 1024
_JOB_HANDLE = None


def enforce_memory_limit(limit_bytes=MEMORY_LIMIT_BYTES):
    """Install the hard ceiling before importing native parsers. Failure is fatal."""
    global _JOB_HANDLE
    if type(limit_bytes) is not int or not 64 * 1024 * 1024 <= limit_bytes <= 2 * 1024**3:
        raise ValueError("INVALID_MEMORY_LIMIT")
    if os.name == "nt":
        import ctypes as c
        from ctypes import wintypes as w

        class Basic(c.Structure):
            _fields_ = [
                ("process_time", c.c_int64),
                ("job_time", c.c_int64),
                ("flags", w.DWORD),
                ("min_ws", c.c_size_t),
                ("max_ws", c.c_size_t),
                ("active", w.DWORD),
                ("affinity", c.c_size_t),
                ("priority", w.DWORD),
                ("scheduling", w.DWORD),
            ]

        class IO(c.Structure):
            _fields_ = [
                (name, c.c_uint64)
                for name in [
                    "read_ops",
                    "write_ops",
                    "other_ops",
                    "read_bytes",
                    "write_bytes",
                    "other_bytes",
                ]
            ]

        class Extended(c.Structure):
            _fields_ = [
                ("basic", Basic),
                ("io", IO),
                ("process_memory", c.c_size_t),
                ("job_memory", c.c_size_t),
                ("peak_process", c.c_size_t),
                ("peak_job", c.c_size_t),
            ]

        kernel = c.WinDLL("kernel32", use_last_error=True)
        kernel.CreateJobObjectW.argtypes = [c.c_void_p, w.LPCWSTR]
        kernel.CreateJobObjectW.restype = w.HANDLE
        kernel.SetInformationJobObject.argtypes = [w.HANDLE, c.c_int, c.c_void_p, w.DWORD]
        kernel.SetInformationJobObject.restype = w.BOOL
        kernel.QueryInformationJobObject.argtypes = [
            w.HANDLE,
            c.c_int,
            c.c_void_p,
            w.DWORD,
            c.c_void_p,
        ]
        kernel.QueryInformationJobObject.restype = w.BOOL
        kernel.AssignProcessToJobObject.argtypes = [w.HANDLE, w.HANDLE]
        kernel.AssignProcessToJobObject.restype = w.BOOL
        kernel.GetCurrentProcess.restype = w.HANDLE
        kernel.CloseHandle.argtypes = [w.HANDLE]
        kernel.CloseHandle.restype = w.BOOL
        handle = kernel.CreateJobObjectW(None, None)
        if not handle:
            raise OSError(c.get_last_error(), "MEMORY_LIMIT_SETUP_FAILED")
        limits = Extended()
        limits.basic.flags = 0x100 | 0x200 | 0x2000
        limits.process_memory = limit_bytes
        limits.job_memory = limit_bytes
        try:
            if not kernel.SetInformationJobObject(handle, 9, c.byref(limits), c.sizeof(limits)):
                raise OSError(c.get_last_error(), "MEMORY_LIMIT_SETUP_FAILED")
            if not kernel.AssignProcessToJobObject(handle, kernel.GetCurrentProcess()):
                raise OSError(c.get_last_error(), "MEMORY_LIMIT_SETUP_FAILED")
            observed = Extended()
            if not kernel.QueryInformationJobObject(
                handle, 9, c.byref(observed), c.sizeof(observed), None
            ):
                raise OSError(c.get_last_error(), "MEMORY_LIMIT_SETUP_FAILED")
            if observed.process_memory != limit_bytes or observed.job_memory != limit_bytes:
                raise OSError("MEMORY_LIMIT_NOT_CONFIRMED")
        except BaseException:
            kernel.CloseHandle(handle)
            raise
        # Keep the anonymous job alive through process exit. Never close it while
        # this process runs: KILL_ON_JOB_CLOSE is an intentional child-lifetime guard.
        _JOB_HANDLE = handle
        mechanism = "windows_job_process_and_job_memory"
    elif os.name == "posix":
        import resource

        soft, hard = resource.getrlimit(resource.RLIMIT_AS)
        existing = [x for x in (soft, hard) if x != resource.RLIM_INFINITY]
        effective = min([limit_bytes, *existing])
        resource.setrlimit(resource.RLIMIT_AS, (effective, effective))
        if resource.getrlimit(resource.RLIMIT_AS) != (effective, effective):
            raise OSError("MEMORY_LIMIT_NOT_CONFIRMED")
        limit_bytes = effective
        mechanism = "posix_rlimit_address_space"
    else:
        raise OSError("MEMORY_LIMIT_PLATFORM_UNSUPPORTED")
    return {
        "memory_limit_enforced": True,
        "memory_limit_bytes": limit_bytes,
        "mechanism": mechanism,
        "os_sandboxed": False,
    }


class WorkerOutputLimit(ValueError):
    pass


def run_bounded(
    command, *, input, env, timeout, max_stdout=34 * 1024 * 1024, max_stderr=1024 * 1024
):
    """Drain both pipes with hard byte ceilings and kill only this child on excess."""
    process = subprocess.Popen(
        command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0,
    )
    outputs = [bytearray(), bytearray()]
    overflow = threading.Event()

    def drain(stream, index, limit):
        try:
            while True:
                chunk = stream.read(65536)
                if not chunk:
                    break
                if len(outputs[index]) + len(chunk) > limit:
                    overflow.set()
                    try:
                        process.kill()
                    except OSError:
                        pass
                    break
                outputs[index].extend(chunk)
        finally:
            stream.close()

    def send():
        try:
            process.stdin.write(input)
            process.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        finally:
            process.stdin.close()

    threads = [
        threading.Thread(target=drain, args=(process.stdout, 0, max_stdout), daemon=True),
        threading.Thread(target=drain, args=(process.stderr, 1, max_stderr), daemon=True),
        threading.Thread(target=send, daemon=True),
    ]
    started = []
    try:
        for thread in threads:
            thread.start()
            started.append(thread)
        process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        raise
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=5)
        if not process.stdin.closed:
            process.stdin.close()
        for thread in started:
            thread.join(timeout=3)
        if len(started) < 1:
            process.stdout.close()
        if len(started) < 2:
            process.stderr.close()
    if any(thread.is_alive() for thread in threads):
        raise WorkerOutputLimit("WORKER_PIPE_DID_NOT_CLOSE")
    if overflow.is_set():
        raise WorkerOutputLimit("WORKER_OUTPUT_LIMIT")
    return subprocess.CompletedProcess(
        command, process.returncode, bytes(outputs[0]), bytes(outputs[1])
    )
