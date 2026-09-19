"""Initialize owner-only token/config files inside the Docker data volume."""

import json
import os
from pathlib import Path
import secrets


def initialize(root=Path("/data"), port=8765):
    if not 1 <= port <= 65535:
        raise ValueError("Invalid client port")
    root = Path(root)
    private, configs = root / "private", root / "client-configs"
    for directory in (root, private, configs):
        if directory.is_symlink():
            raise ValueError("Symlink setup directory refused")
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    token_path = private / "token"
    if token_path.is_symlink():
        raise ValueError("Symlink token refused")
    if not token_path.exists():
        fd = os.open(token_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(secrets.token_urlsafe(48))
    from mcp_server.http_server import Config

    old = os.environ.get("BAND_MCP_TOKEN_FILE")
    os.environ["BAND_MCP_TOKEN_FILE"] = str(token_path)
    try:
        token = Config.from_env().token
    finally:
        if old is None:
            os.environ.pop("BAND_MCP_TOKEN_FILE", None)
        else:
            os.environ["BAND_MCP_TOKEN_FILE"] = old
    url = f"http://127.0.0.1:{port}/mcp"
    headers = {"Authorization": "Bearer " + token}
    stdio = {"command": "docker", "args": ["exec", "-i", "bandstructure-v2", "bandstructure-mcp"]}
    bundle = {
        "vscode.mcp.json": {
            "servers": {"bandstructure": {"type": "http", "url": url, "headers": headers}}
        },
        "claude-code.mcp.json": {
            "mcpServers": {"bandstructure": {"type": "http", "url": url, "headers": headers}}
        },
        "claude-desktop.json": {"mcpServers": {"bandstructure": stdio}},
        "hermes.config.json": {
            "mcp_servers": {"bandstructure": {"url": url, "headers": headers, "timeout": 120}}
        },
    }
    for name, data in bundle.items():
        path = configs / name
        if path.is_symlink():
            raise ValueError("Symlink config refused")
        if not path.exists():
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(data, handle, ensure_ascii=False, indent=2)
    return {
        "status": "ready",
        "configs": str(configs),
        "token_printed": False,
        "note": "Configs contain credentials. Keep private; do not commit. Existing files are not overwritten.",
    }


if __name__ == "__main__":
    print(json.dumps(initialize(port=int(os.environ.get("BAND_MCP_PORT", "8765")))))
