"""Bounded parent-process immutable results; continuations never rerun a parser."""

from collections import OrderedDict
from concurrent.futures import Future
import hashlib
import threading
import time


class ResultCacheError(ValueError):
    def __init__(self, code):
        super().__init__(code)
        self.code = code


class PDFResultCache:
    def __init__(self, *, ttl=600, max_entries=2, max_bytes=64 * 1024 * 1024, clock=time.monotonic):
        self.ttl, self.max_entries, self.max_bytes, self.clock = ttl, max_entries, max_bytes, clock
        self._entries = OrderedDict()
        self._lock = threading.RLock()
        self._pending = {}

    def clear(self):
        with self._lock:
            self._entries.clear()

    def get(self, key, factory, *, continuation=False, expected_sha=None):
        # Coalesce identical misses; unrelated documents must not share a parser lock.
        with self._lock:
            now = self.clock()
            for k, item in list(self._entries.items()):
                if now - item[0] >= self.ttl:
                    del self._entries[k]
            item = self._entries.get(key)
            if item is None:
                if continuation or expected_sha is not None:
                    raise ResultCacheError("PDF_RESULT_EXPIRED")
                future = self._pending.get(key)
                producer = future is None
                if producer:
                    future = self._pending[key] = Future()
            else:
                if expected_sha is not None and expected_sha != item[2]:
                    raise ResultCacheError("PDF_RESULT_VERSION_CHANGED")
                self._entries.move_to_end(key)
                return item[1]
        if not producer:
            return future.result(timeout=120)
        try:
            payload = factory()
            with self._lock:
                if not isinstance(payload, str) or not payload.isascii():
                    raise ResultCacheError("PDF_RESULT_INVALID")
                if len(payload) > min(self.max_bytes, 32 * 1024 * 1024):
                    raise ResultCacheError("PDF_RESULT_TOO_LARGE")
                digest = hashlib.sha256(payload.encode("ascii")).hexdigest()
                while self._entries and (
                    len(self._entries) >= self.max_entries
                    or sum(len(v[1]) for v in self._entries.values()) + len(payload)
                    > self.max_bytes
                ):
                    self._entries.popitem(last=False)
                item = (self.clock(), payload, digest)
                self._entries[key] = item
                future.set_result(payload)
            return payload
        except BaseException as exc:
            future.set_exception(exc)
            raise
        finally:
            with self._lock:
                self._pending.pop(key, None)
