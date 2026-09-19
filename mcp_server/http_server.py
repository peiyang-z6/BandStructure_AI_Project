"""Authenticated, single-owner Streamable HTTP delivery. No anonymous mode."""

from __future__ import annotations

import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import hmac
import json
import os
from pathlib import Path
import re
import time
from urllib.parse import urlsplit
import urllib.request

import jwt

from starlette.responses import JSONResponse
from starlette.routing import Route

MAX_BODY = 16 * 1024 * 1024


class BoundedJWKClient(jwt.PyJWKClient):
    """Read a fixed operator-selected HTTPS key set; never follow redirects."""

    def fetch_data(self):
        class NoRedirect(urllib.request.HTTPRedirectHandler):
            def redirect_request(self, req, fp, code, msg, headers, newurl):
                return None

        try:
            request = urllib.request.Request(self.uri, headers={"Accept": "application/json"})
            with urllib.request.build_opener(NoRedirect).open(request, timeout=5) as response:
                payload = response.read(128 * 1024 + 1)
            if len(payload) > 128 * 1024:
                raise ValueError("Key set exceeds budget")
            data = json.loads(payload)
            if (
                not isinstance(data, dict)
                or not isinstance(data.get("keys"), list)
                or len(data["keys"]) > 32
            ):
                raise ValueError("Invalid key set")
        except (OSError, ValueError, TypeError) as exc:
            raise jwt.PyJWKClientConnectionError("Configured key endpoint unavailable") from exc
        if self.jwk_set_cache is not None:
            self.jwk_set_cache.put(data)
        return data


def https_url(value: str, name: str) -> str:
    p = urlsplit(value)
    if (
        p.scheme != "https"
        or not p.hostname
        or p.username
        or p.password
        or p.query
        or p.fragment
        or any(c.isspace() for c in value)
    ):
        raise ValueError(f"{name} must be a canonical HTTPS URL without credentials/query/fragment")
    return value


@dataclass(frozen=True)
class Config:
    mode: str
    token: str
    resource: str
    issuer: str
    jwks: str
    subject: str
    hosts: tuple[str, ...]
    origins: tuple[str, ...]
    scope: str = "bandstructure"

    @classmethod
    def from_env(cls):
        mode = os.environ.get("BAND_MCP_AUTH_MODE", "bearer")
        if mode not in {"bearer", "oauth"}:
            raise ValueError(
                "BAND_MCP_AUTH_MODE must be bearer or oauth; anonymous HTTP is not supported"
            )
        token = resource = issuer = jwks = subject = ""
        if mode == "bearer":
            path = os.environ.get("BAND_MCP_TOKEN_FILE")
            if not path:
                raise ValueError("BAND_MCP_TOKEN_FILE is required")
            with Path(path).open("rb") as handle:
                token = handle.read(257).decode("ascii").strip()
            if not re.fullmatch(r"[A-Za-z0-9_-]{43,128}", token):
                raise ValueError("Token file must contain 43-128 URL-safe random characters")
        else:
            resource = https_url(os.environ.get("BAND_MCP_PUBLIC_URL", ""), "BAND_MCP_PUBLIC_URL")
            if urlsplit(resource).path != "/mcp":
                raise ValueError("BAND_MCP_PUBLIC_URL must end with /mcp")
            issuer = https_url(os.environ.get("BAND_MCP_OAUTH_ISSUER", ""), "BAND_MCP_OAUTH_ISSUER")
            jwks = https_url(os.environ.get("BAND_MCP_OAUTH_JWKS", ""), "BAND_MCP_OAUTH_JWKS")
            subject = os.environ.get("BAND_MCP_OAUTH_SUBJECT", "")
            if not subject or len(subject) > 512:
                raise ValueError(
                    "An exact BAND_MCP_OAUTH_SUBJECT is required for this single-owner store"
                )
        hosts = {"127.0.0.1:8000", "localhost:8000", "127.0.0.1:8765", "localhost:8765"}
        if resource:
            hosts.add(urlsplit(resource).netloc)
        extras = os.environ.get("BAND_MCP_ALLOWED_HOSTS", "")
        for value in filter(None, (x.strip() for x in extras.split(","))):
            if not re.fullmatch(r"[a-zA-Z0-9.-]+(?::[0-9]{1,5})?", value):
                raise ValueError("Allowed hosts must be exact hostname[:port], never wildcards")
            hosts.add(value)
        origins = tuple(
            f"{scheme}://{host}"
            for host in hosts
            for scheme in ("http", "https")
            if scheme == "https" or host.split(":")[0] in {"127.0.0.1", "localhost"}
        )
        return cls(mode, token, resource, issuer, jwks, subject, tuple(sorted(hosts)), origins)


class Credentials:
    def __init__(self, config: Config):
        self.config = config
        self.jwks_client = None
        if config.mode == "oauth":
            import jwt

            self.jwks_client = BoundedJWKClient(config.jwks, timeout=5, lifespan=300)

    async def valid(self, token: str) -> bool:
        if not token or len(token) > 8192:
            return False
        if self.config.mode == "bearer":
            return hmac.compare_digest(token.encode(), self.config.token.encode())

        def verify():
            import jwt

            try:
                key = self.jwks_client.get_signing_key_from_jwt(token)
                claims = jwt.decode(
                    token,
                    key.key,
                    algorithms=["RS256"],
                    audience=self.config.resource,
                    issuer=self.config.issuer,
                    options={"require": ["exp", "iat", "iss", "aud", "sub"]},
                    leeway=10,
                )
                scopes = claims.get("scope", "")
                return (
                    claims["sub"] == self.config.subject
                    and isinstance(scopes, str)
                    and self.config.scope in scopes.split()
                )
            except (jwt.PyJWTError, OSError, ValueError, TypeError):
                return False

        return await asyncio.to_thread(verify)


class Guard:
    """Authenticate before parsing; bounded ingress and request concurrency."""

    def __init__(self, app, config: Config, credentials=None):
        self.app, self.config = app, config
        self.credentials = credentials or Credentials(config)
        self.windows = OrderedDict()
        self.active = 0

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http":
            return await self.app(scope, receive, send)

        async def reject(code, message, challenge=False):
            headers = {"Cache-Control": "no-store"}
            if challenge:
                headers["WWW-Authenticate"] = 'Bearer realm="bandstructure"'
                if self.config.mode == "oauth":
                    origin = self.config.resource[:-4]
                    headers["WWW-Authenticate"] = (
                        f'Bearer resource_metadata="{origin}/.well-known/oauth-protected-resource/mcp", '
                        f'scope="{self.config.scope}"'
                    )
            await JSONResponse({"error": message}, code, headers=headers)(scope, receive, send)

        pairs = scope.get("headers", [])
        headers = {}
        for key, value in pairs:
            key = key.lower()
            if key in headers and key in {b"host", b"origin", b"authorization", b"content-length"}:
                return await reject(400, "Duplicate security header")
            headers[key] = value
        if headers.get(b"host", b"").decode("latin1") not in self.config.hosts:
            return await reject(421, "Untrusted Host")
        if b"origin" in headers and headers[b"origin"].decode("latin1") not in self.config.origins:
            return await reject(403, "Untrusted Origin")
        path = scope.get("path")
        public = path in {"/healthz", "/.well-known/oauth-protected-resource/mcp"}
        client = (scope.get("client") or ("unknown",))[0]
        now = time.monotonic()
        count, start = self.windows.pop(client, (0, now))
        if now - start > 60:
            count, start = 0, now
        self.windows[client] = (count + 1, start)
        while len(self.windows) > 256:
            self.windows.popitem(last=False)
        if count >= 120 or self.active >= 4:
            return await reject(429, "Request budget exceeded; retry later")
        if scope.get("method") not in {"GET", "POST", "DELETE"}:
            return await reject(405, "Unsupported method")
        try:
            declared = int(headers.get(b"content-length", b"0"))
            if declared < 0 or declared > MAX_BODY:
                return await reject(413, "Request body exceeds 16 MiB")
        except ValueError:
            return await reject(400, "Invalid content length")
        # Read a bounded body before handing off, including chunked transfers.
        self.active += 1
        try:
            if not public:
                raw = headers.get(b"authorization", b"")
                try:
                    scheme, token = raw.decode("ascii").split(" ", 1)
                except (UnicodeError, ValueError):
                    return await reject(401, "Bearer authentication required", True)
                if scheme.lower() != "bearer" or not await self.credentials.valid(token):
                    return await reject(401, "Invalid credential", True)
            body = bytearray()
            more = True
            deadline = asyncio.get_running_loop().time() + 30
            while more:
                event = await asyncio.wait_for(
                    receive(), timeout=max(0, deadline - asyncio.get_running_loop().time())
                )
                if event["type"] == "http.disconnect":
                    return
                body.extend(event.get("body", b""))
                if len(body) > MAX_BODY:
                    return await reject(413, "Request body exceeds 16 MiB")
                more = event.get("more_body", False)
            delivered = False

            async def replay():
                nonlocal delivered
                if not delivered:
                    delivered = True
                    return {"type": "http.request", "body": bytes(body), "more_body": False}
                return await receive()

            await self.app(scope, replay, send)
        except TimeoutError:
            await reject(408, "Request body timeout")
        finally:
            self.active -= 1


def create_app(config=None):
    from mcp.server.auth.routes import create_protected_resource_routes
    from mcp.server.transport_security import TransportSecuritySettings
    from mcp_server.server import mcp
    from mcp_server.version import __version__

    config = config or Config.from_env()
    mcp.settings.stateless_http = True
    mcp.settings.json_response = True
    mcp.settings.max_request_body_size = MAX_BODY
    mcp.settings.transport_security = TransportSecuritySettings(
        enable_dns_rebinding_protection=True,
        allowed_hosts=list(config.hosts),
        allowed_origins=list(config.origins),
    )
    app = mcp.streamable_http_app()

    async def health(request):
        return JSONResponse(
            {"status": "ok", "version": __version__}, headers={"Cache-Control": "no-store"}
        )

    app.routes.append(Route("/healthz", health, methods=["GET"]))
    if config.mode == "oauth":
        from pydantic import AnyHttpUrl

        app.routes.extend(
            create_protected_resource_routes(
                resource_url=AnyHttpUrl(config.resource),
                authorization_servers=[AnyHttpUrl(config.issuer)],
                scopes_supported=[config.scope],
            )
        )
    return Guard(app, config)


def serve(host="127.0.0.1", port=8765):
    import uvicorn

    config = Config.from_env()
    uvicorn.run(
        create_app(config),
        host=host,
        port=port,
        access_log=False,
        proxy_headers=False,
        timeout_keep_alive=5,
        limit_concurrency=16,
        server_header=False,
    )
