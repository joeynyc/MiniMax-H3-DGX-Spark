#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
./scripts/preflight.sh
docker compose up -d --build
docker compose logs -f --tail=100
