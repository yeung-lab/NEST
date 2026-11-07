#!/bin/bash
# Create the `nest` conda env (Python 3.10) with the dependencies needed
# to run the NEST evaluation pipeline (nest_eval + evaluation/).
#
# Pinned versions:
#   - vllm 0.17.0: matches the dev build (commit ebed80a7c, 2026-03-06)
#     used to develop this codebase — released as 0.17.0 stable the next
#     day. Required for the recent VLM architectures we evaluate
#     (Qwen3-VL, etc.). Pulls its compatible torch transitively.
#   - transformers 4.57.6: version verified to work with this vllm.
#   - huggingface-hub 0.36.2: transformers 4.57.6 requires <1.0; unpinned,
#     pip pulls hub 1.x and `import transformers` fails.
#   - datasets 4.8.5, litellm 1.83.14: versions verified with the above.
set -e

ENV_NAME="nest"
VLLM_VERSION="0.17.0"
TRANSFORMERS_VERSION="4.57.6"
HF_HUB_VERSION="0.36.2"
DATASETS_VERSION="4.8.5"
LITELLM_VERSION="1.83.14"

echo "Creating conda environment: $ENV_NAME (python=3.10) with ffmpeg"
# ffmpeg is needed by openai-whisper at runtime; pull it from conda-forge
# so we don't depend on a system binary.
conda create -n "$ENV_NAME" -c conda-forge python=3.10 ffmpeg -y

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$ENV_NAME"

echo "Installing vLLM==$VLLM_VERSION (also installs torch transitively)..."
pip install "vllm==${VLLM_VERSION}"

echo "Installing transformers==$TRANSFORMERS_VERSION..."
pip install "transformers==${TRANSFORMERS_VERSION}"

echo "Installing core eval dependencies..."
pip install \
    "datasets==${DATASETS_VERSION}" \
    "huggingface-hub==${HF_HUB_VERSION}" \
    "litellm==${LITELLM_VERSION}" \
    decord \
    pandas \
    numpy \
    pillow \
    pydantic \
    pyyaml \
    requests

# Whisper goes last: its numba dep can pin numpy/llvmlite versions that
# would conflict if installed before vllm/torch.
echo "Installing transcription dependencies (whisper)..."
pip install openai-whisper tqdm

# Preprocessing extras: yt-dlp pulls ChildPlay videos from YouTube; deno
# is yt-dlp's JS-challenge solver (used to bypass YouTube's bot checks).
echo "Installing preprocessing dependencies (yt-dlp)..."
pip install yt-dlp

if ! command -v deno >/dev/null 2>&1 && [ ! -x "$HOME/.deno/bin/deno" ]; then
    echo "Installing deno (yt-dlp's JS-challenge solver)..."
    curl -fsSL https://deno.land/install.sh | sh
    echo
    echo "  → add deno to your PATH (one-time):"
    echo "      echo 'export PATH=\"\$HOME/.deno/bin:\$PATH\"' >> ~/.bashrc"
fi

echo ""
echo "=========================================="
echo "Environment '$ENV_NAME' ready."
echo "=========================================="
echo "Activate with: conda activate $ENV_NAME"
