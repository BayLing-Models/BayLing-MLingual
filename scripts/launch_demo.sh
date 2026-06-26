#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/path/to/your/python}"
MODEL_PATH="${MODEL_PATH:-/path/to/bayling-mlingual}"
MT_TOKENIZER_PATH="${MT_TOKENIZER_PATH:-/path/to/nllb-1.3b}"
LLM_TOKENIZER_PATH="${LLM_TOKENIZER_PATH:-/path/to/llama3-8b}"
MAX_GEN_LEN="${MAX_GEN_LEN:-256}"

cd "${REPO_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "${PYTHON_BIN}" demo.py \
    --model_path "${MODEL_PATH}" \
    --mt_tokenizer_path "${MT_TOKENIZER_PATH}" \
    --llm_tokenizer_path "${LLM_TOKENIZER_PATH}" \
    --max_gen_len "${MAX_GEN_LEN}"
