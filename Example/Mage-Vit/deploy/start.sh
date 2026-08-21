#!/usr/bin/env bash
set -euo pipefail

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
sdk_root="$(cd "${project_root}/../.." && pwd)"
python_bin="${MEGAVIT_PYTHON:-${sdk_root}/.venv-tfdl-linux/bin/python}"
if [[ ! -x "${python_bin}" ]]; then
  python_bin="python3"
fi

export LD_LIBRARY_PATH="${sdk_root}/lib:${project_root}/deploy/runtime:${LD_LIBRARY_PATH:-}"
export MEGAVIT_DEPLOY_CONFIG="${MEGAVIT_DEPLOY_CONFIG:-${project_root}/deploy/deployment.json}"

cd "${project_root}"
"${python_bin}" deploy/app.py --validate-only
exec "${python_bin}" -m gunicorn \
  --workers 1 \
  --threads "${MEGAVIT_HTTP_THREADS:-4}" \
  --timeout "${MEGAVIT_HTTP_TIMEOUT:-1800}" \
  --bind "${MEGAVIT_BIND:-0.0.0.0:5000}" \
  'deploy.app:create_app()'
