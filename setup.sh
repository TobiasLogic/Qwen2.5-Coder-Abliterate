#!/usr/bin/env bash
# =============================================================================
# setup.sh -- Prepare a fresh Vast.ai (or any CUDA) box to ABLITERATE
# Qwen2.5-Coder-32B-Instruct and export a GGUF.
#
# Installs a MINIMAL, known-good stack: torch (CUDA) + transformers + accelerate
# + datasets + gguf tooling. No unsloth/peft/trl -- abliteration is a static
# weight edit, not a fine-tune. Verifies the GPU and checks disk/RAM headroom
# (a 32B in bf16 is ~65 GB in VRAM; the fp16 output + GGUF need real disk).
#
# Idempotent: satisfied steps are skipped. Override anything via env vars.
# =============================================================================
set -uo pipefail

VENV="${VENV:-/venv/main}"
TORCH_CUDA="${TORCH_CUDA:-cu124}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/${TORCH_CUDA}}"
HF_HOME="${HF_HOME:-/workspace/.hf_home}"
LLAMA_CPP_DIR="${LLAMA_CPP_DIR:-$HOME/llama.cpp}"
# Disk floor: 32B fp16 download (~65) + abliterated output (~65) + GGUF work.
DISK_FLOOR_GB="${DISK_FLOOR_GB:-250}"
RAM_RECOMMENDED_GB="${RAM_RECOMMENDED_GB:-64}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

C_RED=$'\033[0;31m'; C_YEL=$'\033[0;33m'; C_GRN=$'\033[0;32m'; C_OFF=$'\033[0m'
info() { echo "${C_GRN}[setup]${C_OFF} $*"; }
warn() { echo "${C_YEL}[warn ]${C_OFF} $*"; }
err()  { echo "${C_RED}[error]${C_OFF} $*" >&2; }
FAIL=0

# ---------------------------------------------------------------------------
# 1. Python environment
# ---------------------------------------------------------------------------
if [ -f "$VENV/bin/activate" ]; then
  # shellcheck disable=SC1091
  source "$VENV/bin/activate"
  info "activated venv: $VENV"
else
  warn "$VENV not found; using system python3 ($(command -v python3))."
fi
PY="$(command -v python 2>/dev/null || command -v python3)"
info "python: $PY ($($PY --version 2>&1))"

if command -v uv >/dev/null 2>&1; then
  PIP="uv pip install"; info "using uv for installs"
else
  PIP="$PY -m pip install"; info "uv not found; using pip"
fi

export HF_HOME HF_HUB_ENABLE_HF_TRANSFER=1
mkdir -p "$HF_HOME"
info "HF_HOME=$HF_HOME"

# ---------------------------------------------------------------------------
# 2. torch with CUDA (skip if already present & working)
# ---------------------------------------------------------------------------
if $PY -c "import torch,sys; sys.exit(0 if torch.cuda.is_available() else 1)" 2>/dev/null; then
  info "torch already CUDA-enabled: $($PY -c 'import torch; print(torch.__version__)')"
else
  info "installing torch from $TORCH_INDEX_URL ..."
  $PIP "torch>=2.3" --index-url "$TORCH_INDEX_URL" || { err "torch install failed"; FAIL=1; }
fi

# ---------------------------------------------------------------------------
# 3. the rest of the (pinned) stack
# ---------------------------------------------------------------------------
info "installing pinned deps from requirements.txt ..."
$PIP -r "$SCRIPT_DIR/requirements.txt" || { err "dependency install failed"; FAIL=1; }

# ---------------------------------------------------------------------------
# 4. build tools + llama.cpp (for GGUF export) -- best effort, non-fatal
# ---------------------------------------------------------------------------
if ! command -v cmake >/dev/null 2>&1 || ! command -v gcc >/dev/null 2>&1; then
  info "installing build tools (cmake, build-essential, git) ..."
  ( sudo apt-get update -qq && sudo apt-get install -y -qq build-essential cmake git ) 2>/dev/null \
    || ( apt-get update -qq && apt-get install -y -qq build-essential cmake git ) 2>/dev/null \
    || warn "could not install build tools; GGUF export may need cmake/gcc."
else
  info "build tools present (cmake, gcc)."
fi
if [ ! -d "$LLAMA_CPP_DIR" ]; then
  info "cloning llama.cpp -> $LLAMA_CPP_DIR (built lazily by export_gguf.py) ..."
  git clone --depth 1 https://github.com/ggml-org/llama.cpp "$LLAMA_CPP_DIR" \
    2>/dev/null || warn "llama.cpp clone failed; export_gguf.py will retry."
else
  info "llama.cpp present at $LLAMA_CPP_DIR"
fi

# ---------------------------------------------------------------------------
# 5. verify import stack + print versions
# ---------------------------------------------------------------------------
info "verifying import stack ..."
$PY - <<'PYCHECK'
import importlib, sys
ok = True
for m in ["torch", "transformers", "accelerate", "datasets", "huggingface_hub",
          "safetensors", "numpy"]:
    try:
        v = getattr(importlib.import_module(m), "__version__", "?")
        print(f"  {m:16s} {v}")
    except Exception as e:
        print(f"  {m:16s} MISSING ({e})"); ok = False
import torch
print(f"  cuda.available   {torch.cuda.is_available()}")
if torch.cuda.is_available():
    p = torch.cuda.get_device_properties(0)
    print(f"  gpu              {p.name}  {p.total_memory/1e9:.1f} GB  cc{p.major}.{p.minor}")
    print(f"  bf16             {torch.cuda.is_bf16_supported()}")
sys.exit(0 if ok else 3)
PYCHECK
[ $? -ne 0 ] && { err "import verification failed"; FAIL=1; }

# ---------------------------------------------------------------------------
# 6. hardware headroom (GPU / disk / RAM)
# ---------------------------------------------------------------------------
echo; info "=== hardware headroom checks ==="
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | sed 's/^/  gpu: /'
  VRAM_MIB="$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1 | tr -d ' ')"
  if [ -n "${VRAM_MIB:-}" ] && [ "$VRAM_MIB" -lt 70000 ]; then
    warn "GPU has ${VRAM_MIB} MiB. A 32B in bf16 needs ~65 GB -> use an 80 GB card"
    warn "(A100/H100 80GB). Smaller cards work only for <=14B models here."
  fi
else
  err "nvidia-smi not found -- no GPU visible."; FAIL=1
fi

avail_gb() { df -PBG "$1" 2>/dev/null | awk 'NR==2{gsub(/G/,"",$4); print $4+0}'; }
DISK_GB="$(avail_gb "$HF_HOME")"; [ -z "$DISK_GB" ] && DISK_GB="$(avail_gb "$PWD")"
if [ -n "$DISK_GB" ]; then
  echo "  disk free (for $HF_HOME): ${DISK_GB} GB"
  if [ "$DISK_GB" -lt "$DISK_FLOOR_GB" ]; then
    err "Only ${DISK_GB}GB free. 32B fp16 download (~65GB) + abliterated output"
    err "(~65GB) + GGUF export need >= ${DISK_FLOOR_GB}GB. Resize the instance."
    FAIL=1
  else
    info "disk headroom OK (${DISK_GB}GB >= ${DISK_FLOOR_GB}GB)."
  fi
fi

RAM_GB="$(free -g 2>/dev/null | awk '/^Mem:/{print $2}')"
if [ -n "$RAM_GB" ]; then
  echo "  system RAM: ${RAM_GB} GB"
  [ "$RAM_GB" -lt "$RAM_RECOMMENDED_GB" ] \
    && warn "${RAM_GB}GB RAM may be tight when saving the 32B fp16 shards (${RAM_RECOMMENDED_GB}GB+ recommended)." \
    || info "RAM headroom OK (${RAM_GB}GB >= ${RAM_RECOMMENDED_GB}GB)."
fi

warn "NOTE: Vast /workspace is NOT persistent -- a recycle/destroy wipes it."
warn "Upload artifacts to the HF Hub as soon as each step finishes."

echo
if [ "$FAIL" -eq 0 ]; then
  info "${C_GRN}setup complete.${C_OFF} Next:"
  echo "     source $VENV/bin/activate"
  echo "     huggingface-cli login          # paste a FRESH write token"
  echo "     python abliterate.py --model Qwen/Qwen2.5-Coder-7B-Instruct \\"
  echo "         --output out/qwen7b-ablit   # cheap end-to-end validation first"
else
  err "setup finished with problems (see [error] lines). Fix them before running."
  exit 1
fi
