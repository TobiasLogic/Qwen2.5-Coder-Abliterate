# Abliterate Qwen2.5-Coder-32B (make it truly uncensored)

A small, self-contained pipeline that removes refusal behavior from
[`Qwen/Qwen2.5-Coder-32B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct)
by **abliteration** — a static weight edit that orthogonalizes the model against
its single "refusal direction" (Arditi et al. 2024). Unlike a LoRA fine-tune,
this actually *removes* the mechanism the model uses to refuse, and it costs no
inference-time overhead. Exports a merged fp16 model **and** a `Q4_K_M` GGUF for
llama.cpp / Ollama.

Why abliteration instead of fine-tuning: refusals live in a low-dimensional
subspace of the residual stream. A small LoRA can't reliably overwrite that;
surgically projecting the direction out of every weight that writes to the
residual stream can. See the write-up by
[mlabonne](https://huggingface.co/blog/mlabonne/abliteration).

| File | Purpose |
|------|---------|
| `setup.sh` | Install torch + transformers stack, clone llama.cpp, check GPU/disk/RAM. |
| `abliterate.py` | Collect the refusal direction, auto-select the best layer, orthogonalize weights, verify refusal-rate drop, save. |
| `export_gguf.py` | fp16 HF model → `f16.gguf` → `Q4_K_M.gguf` + Ollama `Modelfile`. |
| `requirements.txt` | Pinned, known-good deps (no unsloth/peft/trl). |
| `MODEL_CARD.md` | Template card for the published HF repo. |

## Hardware

- **GPU:** 1× **A100/H100 80 GB**. The 32B in bf16 is ~65 GB and is loaded whole
  onto one GPU for activation capture + weight editing. (≤14B models fit a
  24–48 GB card.)
- **Disk:** **≥ 250 GB** (base ~65 GB + abliterated ~65 GB + GGUF work).
- **RAM:** ≥ 64 GB (saving the 32B fp16 shards).

## Run (on the instance)

```bash
# 0. get the code onto the big disk, then:
cd /workspace/abliterate
bash setup.sh
source /venv/main/bin/activate
huggingface-cli login          # paste a FRESH write token (never commit it)

# 1. VALIDATE the whole pipeline on 7B first (cheap, ~15 min) — catches bugs
python abliterate.py \
    --model Qwen/Qwen2.5-Coder-7B-Instruct \
    --output out/qwen7b-ablit
#   -> watch the "refusal rate: before X% -> after Y%" line drop to ~0%

# 2. the real run: 32B (needs the 80 GB card)
python abliterate.py \
    --model Qwen/Qwen2.5-Coder-32B-Instruct \
    --output out/Qwen2.5-Coder-32B-abliterated

# 3. GGUF q4_k_m
python export_gguf.py \
    --model out/Qwen2.5-Coder-32B-abliterated \
    --outdir out/gguf --quant Q4_K_M \
    --name qwen2.5-coder-32b-abliterated

# 4. publish (fp16 repo + gguf repo)
huggingface-cli upload <you>/Qwen2.5-Coder-32B-abliterated \
    out/Qwen2.5-Coder-32B-abliterated . --repo-type model
huggingface-cli upload <you>/Qwen2.5-Coder-32B-abliterated-GGUF \
    out/gguf . --repo-type model
```

## How `abliterate.py` works

1. Format N harmful + N harmless instructions through the chat template, run the
   model, and cache the residual-stream activation at the **last prompt token**
   for every layer.
2. `mean(harmful) − mean(harmless)` per layer → candidate refusal directions.
3. **Auto-select** (default): for each candidate layer, ablate that direction at
   inference via hooks and measure refusal rate on a held-out harmful set; keep
   the best layer.
4. **Permanently** orthogonalize every residual-writing weight against the unit
   direction `r`:  `W ← W − r (rᵀW)` — all `o_proj`, all `down_proj`, and the
   token embeddings (skipped automatically if embeddings are tied to `lm_head`).
5. Print `refusal rate: before → after`, sanity-check benign coding prompts,
   save the model + `abliteration_info.json`.

Key flags: `--n-samples` (direction estimation size, default 256),
`--auto-select/--no-auto-select`, `--layer-frac` (manual layer),
`--n-candidates`, `--no-edit-embeddings`, `--dtype`, `--device`.

## Notes / tuning

- If some refusals survive, raise `--n-samples` (e.g. 512), widen the candidate
  sweep (`--candidate-lo 0.2 --candidate-hi 0.9 --n-candidates 16`), or try a
  richer harmful set via `--harmful-dataset`.
- Abliteration can slightly dent quality. The benign-prompt sanity check exists
  to catch a model that's been over-edited into incoherence; if that happens,
  pick a different layer (`--no-auto-select --layer-frac 0.5`).
- **Secrets:** authenticate with `huggingface-cli login` / `gh auth login` on the
  box. Never put tokens in code or commits.
