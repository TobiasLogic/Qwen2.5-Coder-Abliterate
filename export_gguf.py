#!/usr/bin/env python3
"""
export_gguf.py -- Convert an abliterated HF model to GGUF via llama.cpp:
  fp16 safetensors  ->  <name>-f16.gguf  ->  <name>-<QUANT>.gguf

Clones + builds llama.cpp (llama-quantize target) on first run, writes an
Ollama Modelfile, and (by default) deletes the big intermediate f16 GGUF.

Example
  python export_gguf.py \
      --model out/Qwen2.5-Coder-32B-abliterated \
      --outdir out/gguf --quant Q4_K_M --name qwen2.5-coder-32b-abliterated
"""
import argparse
import os
import subprocess
import sys

LLAMA_CPP_REPO = "https://github.com/ggml-org/llama.cpp"


def run(cmd, **kw):
    print("+", " ".join(str(c) for c in cmd), flush=True)
    subprocess.run(cmd, check=True, **kw)


def find_quantize_bin(lcpp):
    for cand in (os.path.join(lcpp, "build", "bin", "llama-quantize"),
                 os.path.join(lcpp, "build", "bin", "Release", "llama-quantize"),
                 os.path.join(lcpp, "llama-quantize")):
        if os.path.exists(cand):
            return cand
    return None


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, help="abliterated HF model dir")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--quant", default="Q4_K_M")
    ap.add_argument("--name", default="model")
    ap.add_argument("--llama-cpp", default=os.path.expanduser("~/llama.cpp"))
    ap.add_argument("--keep-f16", action="store_true",
                    help="keep the intermediate f16 GGUF (~2x model size)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    lcpp = args.llama_cpp

    if not os.path.isdir(lcpp):
        run(["git", "clone", "--depth", "1", LLAMA_CPP_REPO, lcpp])

    quant_bin = find_quantize_bin(lcpp)
    if quant_bin is None:
        run(["cmake", "-B", os.path.join(lcpp, "build"), "-S", lcpp,
             "-DLLAMA_CURL=OFF", "-DGGML_NATIVE=ON"])
        run(["cmake", "--build", os.path.join(lcpp, "build"), "--config", "Release",
             "-j", str(os.cpu_count() or 4), "--target", "llama-quantize"])
        quant_bin = find_quantize_bin(lcpp)
    if quant_bin is None:
        sys.exit("[gguf] ERROR: could not build/find llama-quantize")

    # convert-script deps (gguf, etc.)
    req = os.path.join(lcpp, "requirements", "requirements-convert_hf_to_gguf.txt")
    if os.path.exists(req):
        run([sys.executable, "-m", "pip", "install", "-q", "-r", req])

    convert = os.path.join(lcpp, "convert_hf_to_gguf.py")
    f16 = os.path.join(args.outdir, f"{args.name}-f16.gguf")
    outq = os.path.join(args.outdir, f"{args.name}-{args.quant}.gguf")

    run([sys.executable, convert, args.model, "--outfile", f16, "--outtype", "f16"])
    run([quant_bin, f16, outq, args.quant])

    if not args.keep_f16 and os.path.exists(outq):
        os.remove(f16)
        print(f"[gguf] removed intermediate {f16}")

    modelfile = os.path.join(args.outdir, "Modelfile")
    with open(modelfile, "w") as f:
        f.write(f"FROM ./{os.path.basename(outq)}\n")
        f.write('PARAMETER temperature 0.7\n')
        f.write('PARAMETER stop "<|im_end|>"\n')

    print(f"\n[gguf] wrote {outq}")
    print(f"[gguf] Ollama:  cd {args.outdir} && "
          f"ollama create {args.name} -f Modelfile && ollama run {args.name}")


if __name__ == "__main__":
    main()
