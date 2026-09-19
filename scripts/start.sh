#!/bin/sh
set -eu
cd "$(dirname "$0")/.."
docker compose run --build --rm setup
docker compose up -d --wait --wait-timeout 120
printf '%s\n' 'Ready: http://127.0.0.1:8765/mcp (authentication required). See docs/DEPLOYMENT.md.'
