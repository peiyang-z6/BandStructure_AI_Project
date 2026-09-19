$ErrorActionPreference = 'Stop'
Push-Location (Split-Path $PSScriptRoot -Parent)
try {
    docker compose run --build --rm setup
    if ($LASTEXITCODE -ne 0) { throw 'Docker setup failed' }
    docker compose up -d --wait --wait-timeout 120
    if ($LASTEXITCODE -ne 0) { throw 'Docker service did not become healthy' }
    Write-Output 'Ready: http://127.0.0.1:8765/mcp (authentication required). See docs/DEPLOYMENT.md.'
} finally { Pop-Location }
