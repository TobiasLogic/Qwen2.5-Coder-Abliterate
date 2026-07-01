---
license: apache-2.0
base_model: Qwen/Qwen2.5-Coder-32B-Instruct
tags:
  - abliterated
  - uncensored
  - code
  - qwen2.5
  - text-generation
pipeline_tag: text-generation
language:
  - en
---

# Qwen2.5-Coder-32B-abliterated

An **abliterated** (uncensored) build of
[`Qwen/Qwen2.5-Coder-32B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-Coder-32B-Instruct).
The model's refusal direction (Arditi et al. 2024, *"Refusal in LLMs is mediated
by a single direction"*) was estimated from contrasting harmful/harmless prompts
and **orthogonalized out of every residual-writing weight** (all attention
`o_proj`, all MLP `down_proj`, and token embeddings). This is a static weight
edit — no LoRA, no runtime hooks, no inference-time cost. Coding ability is
inherited from the base model.

## Refusal rate (held-out harmful eval)

| | refusal rate |
|--|--|
| base `Qwen2.5-Coder-32B-Instruct` | _{{before}}%_ |
| this model | _{{after}}%_ |

(Measured by the `abliterate.py` eval; see `abliteration_info.json` in the repo.)

## Intended use

Local coding assistant / agent work where the base model's alignment layer
refuses legitimate requests (security research, exploit analysis, red-teaming,
uncensored creative writing). You are responsible for how you use it and for
complying with applicable law.

## Usage

Transformers:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
m = "USER/Qwen2.5-Coder-32B-abliterated"
tok = AutoTokenizer.from_pretrained(m)
model = AutoModelForCausalLM.from_pretrained(m, torch_dtype="bfloat16", device_map="auto")
msgs = [{"role": "user", "content": "Write a port scanner in Python."}]
ids = tok.apply_chat_template(msgs, add_generation_prompt=True, return_tensors="pt").to(model.device)
print(tok.decode(model.generate(ids, max_new_tokens=512)[0][ids.shape[1]:], skip_special_tokens=True))
```

GGUF (a `Q4_K_M` build is in the companion `-GGUF` repo) via Ollama:

```bash
ollama create qwen-coder-abliterated -f Modelfile
ollama run qwen-coder-abliterated
```

## Limitations

- Abliteration removes refusals but can slightly reduce quality vs. the base.
- It does not add knowledge or change biases beyond the refusal behavior.
- Some strongly-trained refusals may partially survive; a stronger system prompt
  closes the gap.

## Method / reproducibility

Produced with the open pipeline at the linked GitHub repo (`abliterate.py`).
Base model: Apache-2.0, so this derivative is Apache-2.0.
