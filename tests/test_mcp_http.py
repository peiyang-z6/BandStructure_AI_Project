"""HTTP guard and actual protocol tests; synthetic credentials, never production keys."""

import asyncio
import json
import time
from types import SimpleNamespace

import jwt
import pytest
from starlette.applications import Starlette
from starlette.responses import JSONResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from mcp_server.http_server import Config, Credentials, Guard, MAX_BODY


@pytest.fixture
def config(tmp_path, monkeypatch):
    token = tmp_path / "token"
    token.write_text("s" * 64)
    monkeypatch.setenv("BAND_MCP_TOKEN_FILE", str(token))
    monkeypatch.setenv("BAND_MCP_AUTH_MODE", "bearer")
    monkeypatch.setenv("BAND_MCP_ALLOWED_HOSTS", "testserver")
    return Config.from_env()


def client(config):
    async def result(request):
        return JSONResponse({"ok": True})

    return TestClient(
        Guard(
            Starlette(
                routes=[Route("/mcp", result, methods=["POST", "GET"]), Route("/healthz", result)]
            ),
            config,
        )
    )


def test_mandatory_auth_host_origin_and_body(config):
    with client(config) as c:
        assert c.get("/healthz").status_code == 200
        assert c.post("/mcp").status_code == 401
        assert c.post("/mcp", headers={"Authorization": "Bearer wrong"}).status_code == 401
        assert (
            c.post("/mcp", headers={"Authorization": "Bearer " + config.token}).status_code == 200
        )
        assert (
            c.post(
                "/mcp",
                headers={"Host": "attacker.example", "Authorization": "Bearer " + config.token},
            ).status_code
            == 421
        )
        assert (
            c.post(
                "/mcp",
                headers={
                    "Origin": "https://attacker.example",
                    "Authorization": "Bearer " + config.token,
                },
            ).status_code
            == 403
        )
        assert (
            c.post(
                "/mcp",
                headers={
                    "Authorization": "Bearer " + config.token,
                    "content-length": str(MAX_BODY + 1),
                },
            ).status_code
            == 413
        )
        assert (
            c.post(
                "/mcp", headers={"Authorization": "Bearer " + config.token, "content-length": "-1"}
            ).status_code
            == 413
        )


def test_authentication_attempts_are_rate_limited(config):
    with client(config) as c:
        for _ in range(120):
            assert c.post("/mcp").status_code == 401
        assert c.post("/mcp").status_code == 429


def test_chunked_body_budget_and_duplicate_headers(config):
    async def run():
        app = Guard(None, config)
        sent = []
        scope = {
            "type": "http",
            "path": "/mcp",
            "method": "POST",
            "client": ("test", 1),
            "headers": [
                (b"host", b"testserver"),
                (b"authorization", ("Bearer " + config.token).encode()),
            ],
        }
        events = [
            {"type": "http.request", "body": b"x" * (MAX_BODY // 2 + 1), "more_body": True}
        ] * 2

        async def receive():
            return events.pop(0)

        async def send(m):
            sent.append(m)

        await app(scope, receive, send)
        assert sent[0]["status"] == 413 and app.active == 0
        sent.clear()
        scope["headers"].append((b"authorization", b"Bearer duplicate"))
        await app(scope, receive, send)
        assert sent[0]["status"] == 400

    asyncio.run(run())


def test_invalid_configuration_fails_closed(config, monkeypatch):
    monkeypatch.setenv("BAND_MCP_AUTH_MODE", "none")
    with pytest.raises(ValueError):
        Config.from_env()
    monkeypatch.setenv("BAND_MCP_AUTH_MODE", "oauth")
    monkeypatch.setenv("BAND_MCP_PUBLIC_URL", "http://example.org/mcp")
    with pytest.raises(ValueError):
        Config.from_env()


@pytest.fixture
def oauth(config, monkeypatch):
    monkeypatch.setenv("BAND_MCP_AUTH_MODE", "oauth")
    monkeypatch.setenv("BAND_MCP_PUBLIC_URL", "https://mcp.example.org/mcp")
    monkeypatch.setenv("BAND_MCP_OAUTH_ISSUER", "https://auth.example.org/issuer/")
    monkeypatch.setenv("BAND_MCP_OAUTH_JWKS", "https://auth.example.org/jwks")
    monkeypatch.setenv("BAND_MCP_OAUTH_SUBJECT", "only-this-owner")
    return Config.from_env()


def test_oauth_signature_expiry_issuer_audience_subject_scope(oauth):
    from cryptography.hazmat.primitives.asymmetric import rsa

    key = rsa.generate_private_key(public_exponent=65537, key_size=2048)
    verifier = Credentials(oauth)
    verifier.jwks_client = SimpleNamespace(
        get_signing_key_from_jwt=lambda _: SimpleNamespace(key=key.public_key())
    )
    now = int(time.time())
    claims = {
        "iss": oauth.issuer,
        "aud": oauth.resource,
        "sub": oauth.subject,
        "scope": "bandstructure",
        "iat": now,
        "exp": now + 60,
    }

    def valid(value, algorithm="RS256", signer=key):
        token = jwt.encode(value, signer, algorithm=algorithm)
        return asyncio.run(verifier.valid(token))

    assert valid(claims)
    for field, bad in [
        ("iss", "https://different.example"),
        ("aud", "other-api"),
        ("sub", "another-user"),
        ("scope", "other"),
        ("exp", now - 120),
    ]:
        assert not valid({**claims, field: bad})
    assert not valid(claims, algorithm="HS256", signer="x" * 64)
    assert not valid({k: v for k, v in claims.items() if k != "exp"})


def test_oauth_metadata_and_challenge(oauth):
    from mcp.server.auth.routes import create_protected_resource_routes
    from pydantic import AnyHttpUrl

    app = Starlette(
        routes=create_protected_resource_routes(
            resource_url=AnyHttpUrl(oauth.resource),
            authorization_servers=[AnyHttpUrl(oauth.issuer)],
            scopes_supported=["bandstructure"],
        )
    )
    with TestClient(Guard(app, oauth)) as c:
        response = c.post("/mcp")
        assert response.status_code == 401
        assert (
            "https://mcp.example.org/.well-known/oauth-protected-resource/mcp"
            in response.headers["www-authenticate"]
        )
        r = c.get("/.well-known/oauth-protected-resource/mcp")
        assert r.status_code == 200
        assert r.json()["authorization_servers"] == [oauth.issuer]


def test_bootstrap_idempotent_and_no_secret_in_summary(tmp_path, monkeypatch):
    from mcp_server.bootstrap import initialize

    monkeypatch.setenv("BAND_MCP_AUTH_MODE", "bearer")
    first = initialize(tmp_path)
    token = (tmp_path / "private/token").read_text()
    assert initialize(tmp_path) == first
    assert token not in json.dumps(first)
    assert (tmp_path / "private/token").read_text() == token
    assert len(token) >= 43
    files = list((tmp_path / "client-configs").glob("*"))
    assert len(files) == 4


def test_jwks_fetch_is_bounded_and_rejects_redirects(oauth, monkeypatch):
    import io
    import urllib.request
    from mcp_server.http_server import BoundedJWKClient

    observed = []

    class Opener:
        def open(self, request, timeout):
            observed.append((request.full_url, timeout))
            return io.BytesIO(b"x" * (128 * 1024 + 1))

    def build(handler):
        assert handler().redirect_request(None, None, 302, "", {}, "http://other.invalid") is None
        return Opener()

    monkeypatch.setattr(urllib.request, "build_opener", build)
    with pytest.raises(jwt.PyJWKClientConnectionError):
        BoundedJWKClient(oauth.jwks).fetch_data()
    assert observed == [(oauth.jwks, 5)]


def test_full_oauth_app_exposes_the_actual_metadata_route(oauth):
    from mcp_server.http_server import create_app

    with TestClient(create_app(oauth)) as c:
        metadata = c.get("/.well-known/oauth-protected-resource/mcp").json()
        assert metadata["resource"] == oauth.resource
        assert metadata["authorization_servers"] == [oauth.issuer]
        assert c.post("/mcp").status_code == 401
