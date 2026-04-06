#!/usr/bin/env python3
"""
2D Phase Diagram Scanner

Applies TWO control vectors simultaneously and sweeps an NxN grid,
producing a phase diagram that reveals ridges, saddle points, and
forbidden zones in the representation manifold.

Usage:
  python run_phase_diagram.py \
    --model /cache/zhangjing/models/Qwen3-8B \
    --vector-dir /cache/zhangjing/repeng_terrain/cross_model/qwen3-8b-bf16/vectors \
    --pairs emotion_valence:empathy,emotion_valence:confidence,creativity:formality \
    --resolution 11 --gpu 5 --tag phase2d-8b-bf16
"""

import argparse, gc, json, math, os, re, sys, time
import numpy as np
import torch
import tqdm
from pathlib import Path

os.environ.setdefault("HF_HOME", "/cache/zhangjing/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_CACHE", "/cache/zhangjing/.cache/huggingface")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

sys.path.insert(0, "/cache/zhangjing/repeng")
from repeng import ControlVector, ControlModel
from repeng.control import model_layer_list

MAX_TOKENS = 150
OUT_BASE = Path("/cache/zhangjing/repeng_terrain/cross_model")

QUERIES_ZH = [
    {"id": "factual", "text": "用两句话告诉我什么是transformer", "type": "knowledge"},
    {"id": "news", "text": "最近有什么值得关注的AI新闻", "type": "open"},
    {"id": "encourage", "text": "写一句鼓励人的话", "type": "emotion"},
]

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0000FE00-\U0000FEFF"
    "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0"
    "\U0001F600-\U0001F64F\U0001F680-\U0001F6FF]+"
)
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)
TAG_RE = re.compile(r"<\|?[a-zA-Z_/][^>]*\|?>")


def clean(text):
    text = THINK_RE.sub("", text)
    text = THINK_OPEN_RE.sub("", text)
    text = TAG_RE.sub("", text)
    return text.strip()


def compute_metrics(text, logprobs_list=None):
    cleaned = clean(text)
    avg_logprob = 0.0
    if logprobs_list:
        lps = [lp for lp in logprobs_list if lp is not None]
        avg_logprob = sum(lps) / len(lps) if lps else 0.0

    trigrams = [cleaned[i:i+3] for i in range(len(cleaned)-2)] if len(cleaned) > 2 else []
    trigram_rep = 1 - len(set(trigrams)) / len(trigrams) if trigrams else 0
    bigrams = [cleaned[i:i+2] for i in range(len(cleaned)-1)] if len(cleaned) > 1 else []
    bigram_rep = 1 - len(set(bigrams)) / len(bigrams) if bigrams else 0

    return {
        "avg_logprob": round(avg_logprob, 4),
        "trigram_rep": round(trigram_rep, 4),
        "bigram_rep": round(bigram_rep, 4),
        "char_len": len(cleaned),
        "emoji_count": len(EMOJI_RE.findall(cleaned)),
        "cleaned_text": cleaned[:200],
    }


def build_chat_prompt(tokenizer, query):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]
    try:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        return tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)


def generate_with_logprobs(model, tokenizer, prompt, max_tokens=MAX_TOKENS):
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = enc["input_ids"].shape[1]
    with torch.no_grad():
        out = model.generate(
            **enc, do_sample=False, max_new_tokens=max_tokens,
            repetition_penalty=1.15, use_cache=True,
            return_dict_in_generate=True, output_scores=True)
    gen_ids = out.sequences[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
    logprobs = []
    if hasattr(out, "scores") and out.scores:
        for step_idx, scores in enumerate(out.scores):
            if step_idx >= len(gen_ids):
                break
            log_softmax = torch.nn.functional.log_softmax(scores[0], dim=-1)
            logprobs.append(float(log_softmax[gen_ids[step_idx].item()].cpu()))
    return text, logprobs


def main():
    parser = argparse.ArgumentParser(description="2D Phase Diagram Scanner")
    parser.add_argument("--model", required=True)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--vector-dir", required=True,
                        help="Directory with pre-trained .gguf vectors")
    parser.add_argument("--pairs", required=True,
                        help="Comma-separated dim pairs, e.g. emotion_valence:empathy,creativity:formality")
    parser.add_argument("--resolution", type=int, default=11,
                        help="Grid resolution per axis (default 11 → 11x11=121 points)")
    parser.add_argument("--range-max", type=float, default=3.0)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    out_dir = OUT_BASE / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    vec_dir = Path(args.vector_dir)

    pairs = [tuple(p.split(":")) for p in args.pairs.split(",")]
    grid = np.linspace(-args.range_max, args.range_max, args.resolution)

    print(f"{'='*70}")
    print(f"2D Phase Diagram: {args.model}")
    print(f"GPU: {device} | Tag: {args.tag}")
    print(f"Pairs: {pairs} | Grid: {args.resolution}x{args.resolution}")
    print(f"Vector dir: {vec_dir}")
    print(f"{'='*70}\n")

    t0 = time.time()

    # Load model
    print("Loading model...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    load_kwargs = dict(trust_remote_code=True)
    model_lower = args.model.lower()
    if "gptq" in model_lower:
        load_kwargs["device_map"] = device
    elif "awq" in model_lower:
        load_kwargs["device_map"] = device
    else:
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"] = device
    try:
        import flash_attn
        if "gptq" not in model_lower and "awq" not in model_lower:
            load_kwargs["attn_implementation"] = "flash_attention_2"
    except ImportError:
        pass

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    try:
        n_layers = len(model_layer_list(model))
    except ValueError:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            n_layers = len(model.model.layers)
            model.repeng_layers = model.model.layers
        else:
            raise

    layer_start = max(1, n_layers // 4)
    layer_end = min(n_layers - 1, n_layers * 3 // 4)
    layer_ids = list(range(layer_start, layer_end))
    ctrl_model = ControlModel(model, layer_ids)

    # Load vectors
    print("Loading control vectors...")
    vectors = {}
    for pair in pairs:
        for dim_name in pair:
            if dim_name not in vectors:
                path = vec_dir / f"{dim_name}.gguf"
                if not path.exists():
                    print(f"  ERROR: {path} not found!")
                    sys.exit(1)
                vectors[dim_name] = ControlVector.import_gguf(str(path))
                print(f"  {dim_name}: loaded ({len(vectors[dim_name].directions)} layers)")

    # Baseline
    print("\nBaseline (no control)...")
    ctrl_model.reset()
    baselines = {}
    for q in QUERIES_ZH:
        prompt = build_chat_prompt(tokenizer, q["text"])
        text, logprobs = generate_with_logprobs(ctrl_model, tokenizer, prompt, args.max_tokens)
        baselines[q["id"]] = compute_metrics(text, logprobs)

    # Phase diagram sweep
    results = {
        "metadata": {
            "model": args.model,
            "tag": args.tag,
            "n_layers": n_layers,
            "resolution": args.resolution,
            "range": [-args.range_max, args.range_max],
            "grid_values": grid.tolist(),
            "pairs": [list(p) for p in pairs],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "baselines": baselines,
        "phase_diagrams": {},
    }

    for pair_idx, (dim_a, dim_b) in enumerate(pairs):
        pair_key = f"{dim_a}_x_{dim_b}"
        print(f"\n{'='*70}")
        print(f"Pair {pair_idx+1}/{len(pairs)}: {dim_a} × {dim_b}")
        print(f"{'='*70}")

        n_points = args.resolution * args.resolution
        diagram = []
        done = 0

        for i, coeff_a in enumerate(grid):
            row = []
            for j, coeff_b in enumerate(grid):
                ctrl_model.reset()

                combined = vectors[dim_a] * float(coeff_a) + vectors[dim_b] * float(coeff_b)
                ctrl_model.set_control(combined, 1.0)

                point = {"coeff_a": round(float(coeff_a), 2),
                         "coeff_b": round(float(coeff_b), 2),
                         "queries": {}}

                for q in QUERIES_ZH:
                    prompt = build_chat_prompt(tokenizer, q["text"])
                    text, logprobs = generate_with_logprobs(
                        ctrl_model, tokenizer, prompt, args.max_tokens)
                    point["queries"][q["id"]] = compute_metrics(text, logprobs)

                row.append(point)
                done += 1

                agg_rep = np.mean([point["queries"][q["id"]]["trigram_rep"] for q in QUERIES_ZH])
                agg_lp = np.mean([point["queries"][q["id"]]["avg_logprob"] for q in QUERIES_ZH])

                phase = "C" if agg_rep > 0.3 else ("B" if agg_rep > 0.1 else "A")
                print(f"  [{done:3d}/{n_points}] "
                      f"{dim_a}={coeff_a:+.1f} {dim_b}={coeff_b:+.1f}  "
                      f"rep={agg_rep:.3f} lp={agg_lp:+.3f} phase={phase}")

            diagram.append(row)
            ctrl_model.reset()

        results["phase_diagrams"][pair_key] = {
            "dim_a": dim_a, "dim_b": dim_b,
            "grid": diagram,
        }

        # Build quick heatmap matrices for the JSON
        rep_matrix = []
        lp_matrix = []
        for row in diagram:
            rep_row = []
            lp_row = []
            for pt in row:
                rep_row.append(round(np.mean([pt["queries"][q["id"]]["trigram_rep"]
                                              for q in QUERIES_ZH]), 4))
                lp_row.append(round(np.mean([pt["queries"][q["id"]]["avg_logprob"]
                                             for q in QUERIES_ZH]), 4))
            rep_matrix.append(rep_row)
            lp_matrix.append(lp_row)

        results["phase_diagrams"][pair_key]["trigram_rep_matrix"] = rep_matrix
        results["phase_diagrams"][pair_key]["avg_logprob_matrix"] = lp_matrix

        out_path = out_dir / "phase_diagram.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print(f"\n  Saved intermediate → {out_path}")

    elapsed = time.time() - t0
    results["metadata"]["elapsed_seconds"] = round(elapsed, 1)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"\n{'='*70}")
    print(f"DONE — {args.tag} — {elapsed/60:.1f} minutes")
    print(f"Results: {out_path}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
