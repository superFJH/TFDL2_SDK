#!/usr/bin/env bash
set -euo pipefail

project_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec python "$project_dir/deploy/api_server.py" \
  --config "$project_dir/deploy/deployment.json" "$@"
