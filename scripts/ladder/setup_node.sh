#!/usr/bin/env bash
# Prepare a fresh 4xH100 node for the Glimmer dense / PLE / Engram ladder.
#
#   ./scripts/ladder/setup_node.sh
#
# Creates $TITAN_VENV (default ~/titan-venv) with the PyTorch nightly the ladder
# was measured on, FlashAttention-3 for varlen attention on Hopper, and this
# repo installed editable; downloads the OLMo3 tokenizer into assets/hf/OLMo3
# and ClimbMix 100B (~140 GB) into data/climbmix-100b. Every step is skipped
# when its output already exists. Needs uv (https://docs.astral.sh/uv/).

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${ROOT}"

TITAN_VENV="${TITAN_VENV:-${HOME}/titan-venv}"
PY="${TITAN_VENV}/bin/python"
NIGHTLY="https://download.pytorch.org/whl/nightly/cu130"
# Versions of the first ladder runs. Nightlies age out of the index; if these
# are gone, drop the pins and record the versions you get.
TORCH_PIN="${TORCH_PIN:-torch==2.16.0.dev20261021+cu130}"
TORCHAO_PIN="${TORCHAO_PIN:-torchao==0.19.0.dev20261021+cu130}"

if [[ ! -x "${PY}" ]]; then
    uv venv -q --python 3.12 "${TITAN_VENV}"
fi
uv pip install -q --python "${PY}" --pre "${TORCH_PIN}" "${TORCHAO_PIN}" \
    --index-url "${NIGHTLY}"
uv pip install -q --python "${PY}" -r requirements.txt -r requirements-dev.txt
uv pip install -q --python "${PY}" --no-deps -e .
uv pip install -q --python "${PY}" --pre flash-attn-3 --index-url "${NIGHTLY}"
"${PY}" -c "
import torch
from torch.nn.attention import activate_flash_attention_impl
activate_flash_attention_impl('FA3')
print('torch', torch.__version__, '| FA3 ok |', torch.cuda.device_count(), 'GPUs')
"

if [[ ! -f assets/hf/OLMo3/tokenizer.json ]]; then
    "${PY}" scripts/download_hf_assets.py --repo_id allenai/Olmo-3-1025-7B \
        --assets tokenizer --local_dir assets/hf
    ln -sfn Olmo-3-1025-7B assets/hf/OLMo3
fi

"${PY}" scripts/download_climbmix.py --workers 16
