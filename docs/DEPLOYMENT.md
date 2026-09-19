# Docker deployment / Docker部署

## Supported scope / 支持范围

Windows with WSL2-backed Linux containers, or Linux x86-64; Docker Engine 24+ and
Compose v2 supporting --wait. Start Docker before running commands. macOS/ARM and
all cloud providers are not claimed tested. No CUDA/model files are needed.

Windows可选Docker Desktop（独立许可），也可在WSL中安装开源Docker Engine。
Linux使用Docker Engine。按[Docker官方文档](https://docs.docker.com/engine/install/)
安装，先确认 docker version 和 docker compose version。Docker访问权限通常等价于
主机管理员权限；不要为AI容器挂载Docker socket，也不要随意增加docker组成员。

## Linux bare-Python prerequisites / Linux 裸 Python 依赖

The Docker image already installs the OpenCV runtime libraries. If you run the MCP
server directly with `pip install '.[documents,ocr]'` on Linux (no Docker), install
the OpenCV X11/GL system libraries first so `import cv2` works:

    sudo apt-get update && sudo apt-get install -y --no-install-recommends \
      libgl1 libglib2.0-0 libxcb1 libxcb-render0 libxcb-shape0 libxcb-shm0 \
      libxcb-xfixes0 libxcb-render-util0 libxcb-image0 libxcb-icccm4 \
      libxcb-keysyms1 libxcb-randr0 libxcb-xkb1 libxkbcommon0 libsm6 libice6 \
      libxext6 libxrender1 libfontconfig1

Windows 不需要这些系统库（OpenCV/opencv-python 自带 DLL）；Windows 仅为稳定起见，
仓库用 `.gitattributes` 固定资源字节，避免大小写/行尾差异破坏已固定的 SHA 校验。

## Local quick start / 本地启动

    git clone https://github.com/peiyang-z6/BandStructure_AI_Project.git
    cd BandStructure_AI_Project
    docker compose run --build --rm setup
    docker compose up -d --wait --wait-timeout 120
    docker compose cp bandstructure:/data/client-configs ./.bandstructure-clients

或在PowerShell运行 ./scripts/start.ps1，在Linux运行 sh scripts/start.sh。
脚本不会修改已有AI客户端设置。首次setup生成随机访问令牌和四份私有配置，
保存在命名数据卷；重跑不覆盖已有令牌或配置。更改端口或轮换令牌后须手动更新客户端。

The initial build downloads dependencies. /healthz is a minimal unauthenticated
health check; /mcp always requires authentication. Default endpoint:
http://127.0.0.1:8765/mcp. Compose publishes only loopback, not the LAN/internet.

    docker compose ps
    docker compose logs --tail 80 bandstructure
    docker compose stop
    docker compose start

Stop/start preserves the named volume. **Do not run docker compose down -v** unless
you intentionally want to delete credentials and all registered artifacts.
For upgrades, back up the stopped data volume, review release notes, rebuild and
recreate the service. A credential/config export is sensitive: do not commit it.

The container runs as UID 10001, with a read-only root filesystem, dropped Linux
capabilities, no-new-privileges, CPU/PID/memory limits and a bounded temporary mount.
Only /data is persistent. It does not mount the host filesystem or Docker socket.
Outbound network is not fully blocked: OAuth mode can fetch its configured key set.

## Register files / 导入文件

    docker compose exec bandstructure mkdir -p /data/inbox
    docker compose cp ./paper.pdf bandstructure:/data/inbox/paper.pdf
    docker compose exec bandstructure bandstructure-attach /data/inbox/paper.pdf

大型PDF：在最后一条命令后加 --page 5，或 --all-pages（上限50页）。
也支持受限图片、POSCAR和VASP XML。把输出的att_...编号交给宿主调用
inspect_attachment、analyze_attachment；编号不是文件路径，也不是安全令牌。

The host AI should see the original paper/image through its own supported input
mechanism. Registering an attachment makes bounded bytes available to MCP; it does
not automatically attach the original to every AI conversation. Export metadata
and PDF-to-pixel transforms preserve the distinction between original/derived bytes.

All clients sharing an instance/token share its artifact store. This is **single-owner,
not multi-tenant**. Separate users require separate instances, data volumes and credentials.

## Hermes

按[Hermes官方说明](https://hermes-agent.nousresearch.com/docs/reference/mcp-config-reference)，
将生成的hermes.config.json里的mcp_servers对象合并到自己的config.yaml对应位置。
JSON对象本身也符合YAML语法，但不要覆盖已有配置。示意（令牌由本机生成）：

```yaml
mcp_servers:
  bandstructure:
    url: http://127.0.0.1:8765/mcp
    headers:
      Authorization: "Bearer ${BANDSTRUCTURE_TOKEN}"
    timeout: 120
```

Set the environment variable in the Hermes process or merge the generated private
config. Restart/reconnect Hermes and list tools. Never paste the token into a model prompt.

## Claude Code / Claude Desktop

Claude Code supports HTTP headers: merge the generated claude-code.mcp.json as
described in [Claude Code MCP documentation](https://code.claude.com/docs/en/mcp).
Keep credential-bearing project configuration private, not in a public .mcp.json.

Claude Desktop本地配置使用生成的claude-desktop.json（或下面的stdio片段）：

```json
{"mcpServers":{"bandstructure":{"command":"docker","args":["exec","-i","bandstructure-v2","bandstructure-mcp"]}}}
```

Start the service first. Do not add -t/TTY; MCP needs clean stdin/stdout. Docker must
be on the client's PATH. The process inherits the container's unprivileged user and
shared data volume. This is not a claim that every Claude client/account was tested.

## VS Code

合并生成的vscode.mcp.json到用户MCP配置，或手动使用密码输入变量；不要覆盖
其他servers。格式依据[VS Code官方配置](https://code.visualstudio.com/docs/agents/reference/mcp-configuration)：

```json
{
  "inputs":[{"type":"promptString","id":"band-token","description":"BandStructure token","password":true}],
  "servers":{"bandstructure":{"type":"http","url":"http://127.0.0.1:8765/mcp","headers":{"Authorization":"Bearer ${input:band-token}"}}}
}
```

Review the discovered tools and keep user confirmations enabled. Source annotations
are hints, not access controls: five tools can write bounded local cache/results.

## ChatGPT: two distinct routes / 两条接入路径

### Private tunnel

An authorized [Secure MCP Tunnel](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)
can forward to this container's stdio command:

    docker exec -i bandstructure-v2 bandstructure-mcp

Use the official tunnel-client setup with your own tunnel_id and runtime credential,
associate the correct workspace, and select the tunnel in ChatGPT developer mode.
We do not create those credentials or claim this account-level flow was tested.

### Public HTTPS + OAuth

ChatGPT云端不能直连本机localhost。使用自己的TLS反向代理和OAuth提供方，按
[OpenAI认证要求](https://developers.openai.com/plugins/build/auth)配置：

1. Copy .env.example to private .env; select BAND_MCP_AUTH_MODE=oauth.
2. Set the exact HTTPS /mcp URL, OAuth issuer, fixed HTTPS JWKS URL, and exact owner
   subject. No URL may contain credentials/query/fragment. Issuer matching is exact.
3. The OAuth provider must issue **RS256 access tokens** with exp, iat, iss, aud, sub,
   and scope including bandstructure. aud equals the exact MCP public URL; sub equals
   the configured owner. Other users are refused rather than sharing private files.
4. The provider handles login, consent, S256 PKCE, token/refresh endpoints and client
   registration. Register the exact callback shown by ChatGPT. This MCP is the
   **resource server**, not an OAuth identity provider.
5. Proxy HTTPS to the loopback service, preserve Host/Authorization, and use a body
   limit no larger than16MiB. Do not expose the container directly without TLS.
6. Recreate the container and verify its protected-resource discovery, 401 challenge
   and authenticated tool calls before adding it to ChatGPT.

Public metadata is at /.well-known/oauth-protected-resource/mcp. Authentication
checks signature, issuer, audience, expiry, owner and scope. Key-set downloads are
bounded and do not follow redirects. Static bearer mode does **not** implement OAuth.

账户权限、HTTPS域名、OAuth端点和实际ChatGPT连接均由操作者提供并验收；没有提供
这些信息时，不把示例地址标成已部署。可用性还取决于[ChatGPT开发者模式](https://developers.openai.com/api/docs/guides/developer-mode)。

## API clients / 普通程序接入

Any compatible Streamable HTTP client can use /mcp with the owner's credential.
Do not expose secrets in arguments/logs or use the token as a URL parameter.
The repository includes a real protocol smoke test:

    python -m pip install '.[documents,ocr,test]'
    python scripts/verify_http.py --config .bandstructure-clients/vscode.mcp.json --output .bandstructure-clients/http-check.json

For standalone Python, pip install . and run bandstructure-mcp for stdio, or set a
private BAND_MCP_TOKEN_FILE and run bandstructure-mcp --transport http. Docker is the
recommended route; no npm package has been published under an invented package name.

## Troubleshooting / 常见问题

- Cannot reach Docker: start the engine, select Linux containers, check WSL2 and
  Docker permissions. Do not disable antivirus/firewalls or TLS verification.
- 401: missing/wrong token, wrong OAuth owner/scope/audience, expiry or key-fetch failure.
- 403/421: Origin/Host is not allowlisted. Configure the exact host; do not use '*'.
- 413: request exceeds16MiB or parser/adapter limits. Register a smaller page/ROI.
- 429: more than120 requests/minute per connection IP or four active requests;
  retry with backoff. A reverse proxy's clients share this IP budget.
- Refused/unavailable: inspect the exact error. Missing physical units, malformed
  data, non-electronic panels and legacy-model gates are deliberate restrictions.
- Store full: no silent eviction. Export/back up data and use operator lifecycle
  tools; automatic expired-result cleanup never deletes registered attachments.
- The service does not provide durable queued jobs, a web upload UI, multi-tenant
  permissions or a complete hostile-network security guarantee.
