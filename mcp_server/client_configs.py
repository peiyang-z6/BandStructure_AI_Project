"""Build portable, non-installed stdio client configurations."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from mcp_server.version import __version__


SERVER_ENV = {
    "PYTHONDONTWRITEBYTECODE": "1",
    "CUDA_VISIBLE_DEVICES": "-1",
    "OMP_NUM_THREADS": "1",
    "OPENBLAS_NUM_THREADS": "1",
    "BAND_MCP_UPLOAD_ISOLATION": "1",
}


def _installed_paths(python_executable: str | Path, project_root: str | Path) -> tuple[Path, Path]:
    python = Path(python_executable)
    root = Path(project_root)
    if not python.is_absolute() or not python.is_file():
        raise ValueError("python executable must be an existing absolute file")
    if not root.is_absolute() or not root.is_dir():
        raise ValueError("project root must be an existing absolute directory")
    server = root / "mcp_server" / "server.py"
    if not server.is_file() or not server.resolve().is_relative_to(root.resolve()):
        raise ValueError("mcp_server/server.py is missing from project root")
    return python.resolve(), root.resolve()


def _toml_string(value: str) -> str:
    """JSON strings are valid TOML basic strings and handle Windows paths safely."""
    return json.dumps(value, ensure_ascii=True)


def build_client_configs(
    python_executable: str | Path,
    project_root: str | Path,
) -> dict[str, Any]:
    """Return client templates without installing or claiming client verification."""
    python, root = _installed_paths(python_executable, project_root)
    from mcp_server.artifacts import store_directory

    server_env = {**SERVER_ENV, "BAND_MCP_STORE_DIR": store_directory(create=False).as_posix()}
    command = python.as_posix()
    server = (root / "mcp_server" / "server.py").as_posix()
    common = {
        "type": "stdio",
        "command": command,
        "args": ["-B", server],
        "env": server_env,
    }
    vscode = {**common, "cwd": root.as_posix()}
    env_lines = "\n".join(f"{key} = {_toml_string(value)}" for key, value in server_env.items())
    codex_toml = (
        "[mcp_servers.bandstructure]\n"
        f"command = {_toml_string(command)}\n"
        f"args = [{_toml_string('-B')}, {_toml_string(server)}]\n"
        "startup_timeout_sec = 120\n"
        "tool_timeout_sec = 120\n\n"
        "[mcp_servers.bandstructure.env]\n"
        f"{env_lines}\n"
    )
    return {
        "claude_desktop.json": {"mcpServers": {"bandstructure": dict(common)}},
        "cursor.json": {"mcpServers": {"bandstructure": dict(common)}},
        "vscode.mcp.json": {"servers": {"bandstructure": vscode}},
        "codex.config.toml": codex_toml,
        "manifest.json": {
            "schema_version": 1,
            "delivery_version": __version__,
            "transport": "stdio",
            "client_templates": ["claude_desktop", "cursor", "vscode", "codex"],
            "verification_status": "generated_not_client_verified",
            "generated_configs_are_installed": False,
            "openai_responses_api": "use_authenticated_HTTP_or_authorized_tunnel_not_this_stdio_template",
            "windows_stdio_sandbox": "not_provided_by_vscode",
            "server_read_only": False,
            "side_effects": "Five analysis/export tools may persist bounded local cache artifacts; no arbitrary file edits or remote writes.",
            "references": {
                "vscode": "https://code.visualstudio.com/docs/agents/reference/mcp-configuration",
                "claude": "https://docs.anthropic.com/en/docs/claude-code/mcp",
                "openai_remote_mcp": "https://developers.openai.com/api/reference/cli/resources/responses/methods/create",
            },
        },
    }
