#!/usr/bin/env bash
set -euo pipefail
PROJECT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
FRAMEWORK_DIR="${FRAMEWORK_DIR:-$PROJECT/repos/cosmos-framework}"
mode="${1:-bimanual}"
[[ "$mode" == bimanual || "$mode" == transfer ]] || { echo 'Usage: setup_framework.sh bimanual|transfer' >&2; exit 2; }
revision="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["source"]["commit"])' "$PROJECT/configs/versions.json")"
if [[ ! -d "$FRAMEWORK_DIR/.git" ]]; then
 git clone --no-checkout https://github.com/NVIDIA/cosmos-framework.git "$FRAMEWORK_DIR"
 git -C "$FRAMEWORK_DIR" checkout --detach "$revision"
fi
[[ "$(git -C "$FRAMEWORK_DIR" rev-parse HEAD)" == "$revision" ]] || { echo 'Framework revision mismatch; use a new checkout' >&2; exit 1; }
[[ -z "$(git -C "$FRAMEWORK_DIR" status --porcelain --untracked-files=no)" ]] || { echo 'Framework has tracked edits' >&2; exit 1; }
cd "$FRAMEWORK_DIR"
export UV_CACHE_DIR="${UV_CACHE_DIR:-$PROJECT/storage/cache/uv}"
# Fail here if the official lock cannot be reproduced. Never regenerate it silently.
uvx --from uv==0.11.29 uv lock --check
if [[ "$mode" == bimanual ]]; then
 uvx --from uv==0.11.29 uv sync --locked --python 3.13 --group cu130 --extra guardrail
else
 uvx --from uv==0.11.29 uv sync --locked --python 3.13 --all-extras --group cu130-train
fi
