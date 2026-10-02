#!/usr/bin/env bash
set -euo pipefail

pkill -9 -f "vllm serve" || true

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_ACTIVATE="$SCRIPT_DIR/unsloth-nvfp4-env/bin/activate"

if [ ! -f "$VENV_ACTIVATE" ]; then
  echo "Virtual environment not found at $VENV_ACTIVATE" >&2
  exit 1
fi

. "$VENV_ACTIVATE"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
LOGFILE="$SCRIPT_DIR/vllm_${TIMESTAMP}.log"
exec > >(tee -a "$LOGFILE") 2>&1

echo "Using vLLM: $(python -c 'import vllm; print(vllm.__version__)')"

API_KEY_FILE="$SCRIPT_DIR/API_KEY.txt"
if [[ ! -s "$API_KEY_FILE" ]]; then
  echo "API key file is missing or empty: $API_KEY_FILE" >&2
  exit 1
fi

API_KEY="$(<"$API_KEY_FILE")"
if [[ -z "$API_KEY" ]]; then
  echo "API key file does not contain a key: $API_KEY_FILE" >&2
  exit 1
fi

export NCCL_P2P_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_FLASHINFER_WORKSPACE_BUFFER_SIZE=67108864

exec vllm serve \
  --model /data/huggingface/hub/models--gittensor-model-hub--Qwen3.8-27B-NVFP4-RTX5090/snapshots/0cc27958cefbbe231782ec8511de8c4eb5233348 \
  --served-model-name Qwen3.8-27B-NVFP4 \
  --api-key "$API_KEY" \
  --tensor-parallel-size 2 \
  --no-enable-flashinfer-autotune \
  --trust-remote-code \
  --max-model-len 262144 \
  --max-num-seqs 2 \
  --max-num-batched-tokens 4096 \
  --compilation-config '{"cudagraph_mode": "PIECEWISE"}' \
  --mamba-cache-dtype bfloat16 \
  --mamba-ssm-cache-dtype bfloat16 \
  --gpu_memory_utilization 0.95 \
  --kv-cache-dtype fp8 \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --disable-custom-all-reduce \
  --limit-mm-per-prompt '{"video": 0}' \
  --mm-processor-kwargs '{"max_pixels": 1000000}' \
  --reasoning-parser qwen3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_xml \
  --default-chat-template-kwargs '{"preserve_thinking": true}' \
  --speculative-config '{"method":"mtp","num_speculative_tokens": 3}'
