#!/usr/bin/env python3
"""
Cross-Model Representation Terrain Map

Generic script that:
  1. Loads any HuggingFace model (BF16, GPTQ, AWQ)
  2. Trains 5 RepEng control vectors
  3. Runs dose-response sweep -3.0 → +3.0 (step 0.2)
  4. Computes metrics: logprob, repetition, style, cosine similarity
  5. Saves results in standardized JSON

Usage:
  python run_terrain_generic.py --model Qwen/Qwen3-8B --gpu 4 --tag qwen3-8b-bf16
  python run_terrain_generic.py --model Qwen/Qwen3-8B-AWQ --gpu 5 --tag qwen3-8b-awq
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
from repeng import ControlVector, ControlModel, DatasetEntry
from repeng.control import model_layer_list

# ── Config ────────────────────────────────────────────────────────────────────

STEP = 0.2
RANGE_MIN, RANGE_MAX = -3.0, 3.0
MAX_TOKENS = 150
OUT_BASE = Path("/cache/zhangjing/repeng_terrain/cross_model")

DIMENSIONS = {
    "emotion_valence": {
        "positive": ["happy", "joyful", "ecstatic"],
        "negative": ["sad", "melancholic", "sorrowful"],
    },
    "formality": {
        "positive": [
            "speaking in a very formal, academic, and eloquent manner",
            "writing like a distinguished professor",
            "being extremely formal and articulate",
        ],
        "negative": [
            "speaking in a very casual, slang-heavy, informal manner",
            "writing like a teenager texting friends",
            "being extremely casual and relaxed",
        ],
    },
    "creativity": {
        "positive": [
            "extremely creative, imaginative, and unconventional",
            "wildly inventive and thinking outside the box",
            "deeply artistic and metaphorical",
        ],
        "negative": [
            "extremely literal, factual, and straightforward",
            "purely analytical and avoiding all metaphor",
            "completely dry and technical",
        ],
    },
    "confidence": {
        "positive": ["extremely confident and assertive", "bold and decisive", "very self-assured and commanding"],
        "negative": ["extremely uncertain and hesitant", "timid and indecisive", "very self-doubting"],
    },
    "empathy": {
        "positive": ["deeply empathetic, warm, and emotionally supportive", "incredibly caring and understanding", "profoundly compassionate"],
        "negative": ["cold, detached, and purely analytical", "emotionally distant and impersonal", "clinical and devoid of emotional warmth"],
    },
}

QUERIES_ZH = [
    {"id": "factual", "text": "用两句话告诉我什么是transformer", "type": "knowledge"},
    {"id": "news", "text": "最近有什么值得关注的AI新闻", "type": "open"},
    {"id": "encourage", "text": "写一句鼓励人的话", "type": "emotion"},
]
QUERIES_EN = [
    {"id": "factual", "text": "Tell me what a transformer is in two sentences", "type": "knowledge"},
    {"id": "news", "text": "What recent AI news is worth paying attention to", "type": "open"},
    {"id": "encourage", "text": "Write an encouraging sentence for me", "type": "emotion"},
]
QUERIES_MIXED = [
    {"id": "factual", "text": "Tell me什么是transformer，用two sentences", "type": "knowledge"},
    {"id": "news", "text": "最近有什么值得关注的AI news", "type": "open"},
    {"id": "encourage", "text": "Write一句encouraging的话", "type": "emotion"},
]
QUERY_SETS = {"zh": QUERIES_ZH, "en": QUERIES_EN, "mixed": QUERIES_MIXED}
QUERIES = QUERIES_ZH

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0000FE00-\U0000FEFF"
    "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0"
    "\U0001F600-\U0001F64F\U0001F680-\U0001F6FF]+"
)
EXCL_RE = re.compile(r"[！!]{1,}")
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)
TAG_RE = re.compile(r"<\|?[a-zA-Z_/][^>]*\|?>")


# ── Helpers ───────────────────────────────────────────────────────────────────

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

    chars = list(cleaned)
    bigrams = [cleaned[i:i+2] for i in range(len(cleaned)-1)] if len(cleaned) > 1 else []
    trigrams = [cleaned[i:i+3] for i in range(len(cleaned)-2)] if len(cleaned) > 2 else []
    bigram_rep = 1 - len(set(bigrams)) / len(bigrams) if bigrams else 0
    trigram_rep = 1 - len(set(trigrams)) / len(trigrams) if trigrams else 0

    emoji_count = len(EMOJI_RE.findall(cleaned))
    excl_count = len(EXCL_RE.findall(cleaned))
    char_len = len(cleaned)

    sentences = re.split(r"[。！？!?\n]", cleaned)
    sentences = [s for s in sentences if len(s.strip()) > 0]
    avg_sent_len = float(np.mean([len(s) for s in sentences])) if sentences else 0

    has_think = "<think>" in text
    think_closed = "</think>" in text
    think_fraction = 0.0
    if has_think:
        m = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if m:
            think_fraction = len(m.group(1)) / len(text) if text else 0
        else:
            m2 = re.search(r"<think>(.*)", text, re.DOTALL)
            think_fraction = len(m2.group(1)) / len(text) if m2 and text else 0

    return {
        "avg_logprob": round(avg_logprob, 4),
        "bigram_rep": round(bigram_rep, 4),
        "trigram_rep": round(trigram_rep, 4),
        "emoji_count": emoji_count,
        "excl_count": excl_count,
        "char_len": char_len,
        "avg_sent_len": round(avg_sent_len, 1),
        "think_fraction": round(think_fraction, 3),
        "has_think_open": has_think,
        "think_closed": think_closed,
        "cleaned_text": cleaned[:200],
        "raw_text": text[:300],
    }


def build_chat_prompt(tokenizer, query, enable_thinking=False):
    """Build a chat prompt using the tokenizer's apply_chat_template."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": query},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def build_persona_prompt(tokenizer, persona, suffix):
    """Build a persona prompt for vector training."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": persona},
    ]
    try:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
    except TypeError:
        prompt = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt + suffix


def generate_with_logprobs(model, tokenizer, prompt, max_tokens=MAX_TOKENS,
                           temperature=None, top_p=None, repetition_penalty=1.15):
    """Generate text and collect per-token log-probabilities."""
    enc = tokenizer(prompt, return_tensors="pt").to(model.device)
    input_len = enc["input_ids"].shape[1]

    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        repetition_penalty=repetition_penalty,
        use_cache=True,
        return_dict_in_generate=True,
        output_scores=True,
    )
    if temperature is not None and temperature > 0:
        gen_kwargs["do_sample"] = True
        gen_kwargs["temperature"] = temperature
        if top_p is not None:
            gen_kwargs["top_p"] = top_p
    else:
        gen_kwargs["do_sample"] = False

    with torch.no_grad():
        out = model.generate(**enc, **gen_kwargs)

    gen_ids = out.sequences[0][input_len:]
    text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

    logprobs = []
    if hasattr(out, "scores") and out.scores:
        for step_idx, scores in enumerate(out.scores):
            if step_idx >= len(gen_ids):
                break
            log_softmax = torch.nn.functional.log_softmax(scores[0], dim=-1)
            token_id = gen_ids[step_idx].item()
            logprobs.append(float(log_softmax[token_id].cpu()))

    return text, logprobs


def compute_hiddens_generic(model, tokenizer, train_strs, hidden_layers, batch_size):
    """Extract hidden states - works for any transformers model."""
    batched = [train_strs[p:p+batch_size] for p in range(0, len(train_strs), batch_size)]
    hs = {l: [] for l in hidden_layers}
    with torch.no_grad():
        for batch in tqdm.tqdm(batched, desc="  extracting hiddens"):
            enc = tokenizer(batch, padding=True, return_tensors="pt").to(model.device)
            out = model(**enc, output_hidden_states=True, use_cache=False)
            mask = enc["attention_mask"]
            for i in range(len(batch)):
                last = mask[i].nonzero(as_tuple=True)[0][-1].item()
                for l in hidden_layers:
                    idx = l + 1 if l >= 0 else l
                    hs[l].append(out.hidden_states[idx][i][last].cpu().float().numpy())
            del out
            torch.cuda.empty_cache()
    return {k: np.vstack(v) for k, v in hs.items()}


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Cross-model RepEng terrain sweep")
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index")
    parser.add_argument("--tag", required=True, help="Short tag for output filenames")
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size for vector training")
    parser.add_argument("--skip-training", action="store_true", help="Skip vector training, load from disk")
    parser.add_argument("--trust-remote-code", action="store_true", default=True)
    parser.add_argument("--thinking", action="store_true", help="Enable thinking mode (Qwen3)")
    parser.add_argument("--lang", choices=["zh", "en", "mixed"], default="zh",
                        help="Query language set")
    parser.add_argument("--temperature", type=float, default=None,
                        help="Sampling temperature (enables do_sample)")
    parser.add_argument("--top-p", type=float, default=None, help="Top-p sampling")
    parser.add_argument("--repetition-penalty", type=float, default=1.15)
    parser.add_argument("--vector-dir", type=str, default=None,
                        help="Load pre-trained vectors from this directory instead of training")
    parser.add_argument("--multi-gpu", type=str, default=None,
                        help="Comma-separated GPU ids for multi-GPU loading, e.g. '6,7'")
    args = parser.parse_args()

    device = f"cuda:{args.gpu}"
    out_dir = OUT_BASE / args.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    queries = QUERY_SETS.get(args.lang, QUERIES_ZH)
    enable_thinking = args.thinking

    print(f"{'='*70}")
    print(f"RepEng Terrain Map: {args.model}")
    print(f"GPU: {device} | Tag: {args.tag} | Max tokens: {args.max_tokens}")
    print(f"Lang: {args.lang} | Thinking: {enable_thinking} | Temp: {args.temperature}")
    print(f"Output: {out_dir}")
    print(f"{'='*70}\n")

    t0 = time.time()

    # ── Load model + tokenizer ────────────────────────────────────────────
    print("Loading tokenizer...")
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=args.trust_remote_code
    )
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    print(f"Loading model {args.model} → {device}...")
    load_kwargs = dict(trust_remote_code=args.trust_remote_code)

    model_name_lower = args.model.lower()

    if args.multi_gpu:
        gpu_ids = [int(g) for g in args.multi_gpu.split(",")]
        max_mem = {i: "23GiB" for i in gpu_ids}
        load_kwargs["device_map"] = "auto"
        load_kwargs["max_memory"] = max_mem
        print(f"  Multi-GPU loading: GPUs {gpu_ids}")
    elif "gptq" in model_name_lower:
        print("  Detected GPTQ quantization")
        load_kwargs["device_map"] = device
    elif "awq" in model_name_lower:
        print("  Detected AWQ quantization")
        load_kwargs["device_map"] = device
    else:
        print("  Loading as BF16")
        load_kwargs["torch_dtype"] = torch.bfloat16
        load_kwargs["device_map"] = device

    try:
        import flash_attn
        if "gptq" not in model_name_lower and "awq" not in model_name_lower:
            load_kwargs["attn_implementation"] = "flash_attention_2"
            print("  Using Flash Attention 2")
    except ImportError:
        print("  Flash Attention not available, using default")

    model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
    model.eval()

    try:
        n_layers = len(model_layer_list(model))
    except ValueError:
        if hasattr(model, 'model') and hasattr(model.model, 'layers'):
            n_layers = len(model.model.layers)
            model.repeng_layers = model.model.layers
            print("  Set repeng_layers override for MoE model")
        else:
            raise
    hidden_size = model.config.hidden_size
    model_type = model.config.model_type
    print(f"  Loaded: {model_type}, {n_layers} layers, hidden_size={hidden_size}")
    print(f"  Memory: {torch.cuda.memory_allocated(args.gpu) / 1e9:.1f} GB\n")

    # ── Train or load control vectors ─────────────────────────────────────
    vec_dir = out_dir / "vectors"
    vec_dir.mkdir(exist_ok=True)

    layer_start = max(1, n_layers // 4)
    layer_end = min(n_layers - 1, n_layers * 3 // 4)
    layer_ids = list(range(layer_start, layer_end))
    print(f"Control layers: {layer_start}–{layer_end-1} ({len(layer_ids)} layers)")

    ctrl_model = ControlModel(model, layer_ids)

    ext_vec_dir = Path(args.vector_dir) if args.vector_dir else None
    load_from = ext_vec_dir if ext_vec_dir and all((ext_vec_dir / f"{d}.gguf").exists() for d in DIMENSIONS) else None
    if load_from is None and args.skip_training and all((vec_dir / f"{d}.gguf").exists() for d in DIMENSIONS):
        load_from = vec_dir

    if load_from:
        print(f"Loading pre-trained vectors from {load_from}...")
        vectors = {}
        for dim in DIMENSIONS:
            vectors[dim] = ControlVector.import_gguf(str(load_from / f"{dim}.gguf"))
            if load_from != vec_dir:
                vectors[dim].export_gguf(str(vec_dir / f"{dim}.gguf"))
            print(f"  {dim}: loaded ({len(vectors[dim].directions)} layers)")
    else:
        print("Training control vectors...")
        with open("/cache/zhangjing/repeng/notebooks/data/all_truncated_outputs.json") as f:
            suffixes = [s for s in json.load(f) if s][:60]

        vectors = {}
        for dim_name, dim_cfg in DIMENSIONS.items():
            print(f"\n  Training: {dim_name}")
            dataset = []
            for suffix in suffixes:
                toks = tokenizer.tokenize(suffix)
                for i in range(1, min(len(toks), 8)):
                    trunc = tokenizer.convert_tokens_to_string(toks[:i])
                    for pos, neg in zip(dim_cfg["positive"], dim_cfg["negative"]):
                        dataset.append(DatasetEntry(
                            positive=build_persona_prompt(tokenizer, f"Act as if you're extremely {pos}.", trunc),
                            negative=build_persona_prompt(tokenizer, f"Act as if you're extremely {neg}.", trunc),
                        ))
            ctrl_model.reset()
            vec = ControlVector.train(
                ctrl_model, tokenizer, dataset,
                batch_size=args.batch_size,
                compute_hiddens=compute_hiddens_generic,
            )
            vectors[dim_name] = vec
            vec.export_gguf(str(vec_dir / f"{dim_name}.gguf"))
            print(f"    → {len(vec.directions)} layers, {len(dataset)} pairs, saved to {dim_name}.gguf")

    # ── Terrain sweep ─────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print("TERRAIN SWEEP")
    print(f"{'='*70}\n")

    steps = []
    v = RANGE_MIN
    while v <= RANGE_MAX + 1e-9:
        steps.append(round(v, 2))
        v += STEP

    dim_names = list(DIMENSIONS.keys())
    total = len(dim_names) * len(steps) * len(queries)
    print(f"{len(dim_names)} dims × {len(steps)} steps × {len(queries)} queries = {total} generations")
    est_sec = total * 3
    print(f"Estimated time: ~{est_sec // 60} min\n")

    results = {
        "metadata": {
            "model": args.model,
            "tag": args.tag,
            "model_type": model_type,
            "n_layers": n_layers,
            "hidden_size": hidden_size,
            "control_layers": [layer_start, layer_end - 1],
            "gpu": f"cuda:{args.gpu}",
            "step": STEP,
            "range": [RANGE_MIN, RANGE_MAX],
            "max_tokens": args.max_tokens,
            "dimensions": dim_names,
            "queries": queries,
            "lang": args.lang,
            "thinking": enable_thinking,
            "temperature": args.temperature,
            "repetition_penalty": args.repetition_penalty,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "memory_gb": round(torch.cuda.memory_allocated(args.gpu) / 1e9, 1),
        },
        "baselines": {},
        "sweeps": {},
    }

    gen_kwargs = dict(
        temperature=args.temperature,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
    )

    # Baselines
    print("Phase 1: Baselines (RepEng=0)...")
    ctrl_model.reset()
    for q in queries:
        prompt = build_chat_prompt(tokenizer, q["text"], enable_thinking=enable_thinking)
        text, logprobs = generate_with_logprobs(ctrl_model, tokenizer, prompt, args.max_tokens, **gen_kwargs)
        metrics = compute_metrics(text, logprobs)
        results["baselines"][q["id"]] = {"query": q["text"], "metrics": metrics}
        print(f"  [{q['id']}] logprob={metrics['avg_logprob']:.3f}: {clean(text)[:80]}")

    # Sweeps
    print(f"\nPhase 2: Sweeping {len(dim_names)} dimensions...")
    done = 0
    for dim in dim_names:
        print(f"\n--- {dim} ---")
        results["sweeps"][dim] = []

        for val in steps:
            ctrl_model.reset()
            if val != 0.0:
                ctrl_model.set_control(vectors[dim], val)

            point = {"value": val, "queries": {}}
            for q in queries:
                prompt = build_chat_prompt(tokenizer, q["text"], enable_thinking=enable_thinking)
                text, logprobs = generate_with_logprobs(ctrl_model, tokenizer, prompt, args.max_tokens, **gen_kwargs)
                metrics = compute_metrics(text, logprobs)
                point["queries"][q["id"]] = {"metrics": metrics}
                done += 1

            results["sweeps"][dim].append(point)

            agg_logprob = np.mean([point["queries"][q["id"]]["metrics"]["avg_logprob"] for q in queries])
            agg_rep = np.mean([point["queries"][q["id"]]["metrics"]["trigram_rep"] for q in queries])
            bar = "#" * max(0, int((agg_logprob + 2) * 20))
            print(f"  {val:+.1f}  logp={agg_logprob:+.3f}  rep={agg_rep:.3f}  [{done}/{total}]  {bar}")

        ctrl_model.reset()

    # Save intermediate
    out_path = out_dir / "terrain_data.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")

    # ── Cosine similarity ─────────────────────────────────────────────────
    print("\nPhase 3: Computing cosine similarities...")
    try:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(
            "/cache/zhangjing/voiceagent/models/bge-small-zh-v1.5",
            device=device,
        )
        baseline_embeds = {}
        for qid, bdata in results["baselines"].items():
            txt = bdata["metrics"]["cleaned_text"]
            if txt:
                baseline_embeds[qid] = embed_model.encode(txt, normalize_embeddings=True)

        for dim in dim_names:
            for point in results["sweeps"][dim]:
                for q in queries:
                    qid = q["id"]
                    txt = point["queries"][qid]["metrics"]["cleaned_text"]
                    if txt and qid in baseline_embeds:
                        emb = embed_model.encode(txt, normalize_embeddings=True)
                        cos_sim = float(np.dot(baseline_embeds[qid], emb))
                    else:
                        cos_sim = 0.0
                    point["queries"][qid]["metrics"]["cosine_sim_to_baseline"] = round(cos_sim, 4)

        del embed_model
        torch.cuda.empty_cache()

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("  Cosine similarities computed and saved.")
    except Exception as e:
        print(f"  Cosine computation skipped: {e}")

    # ── Vector geometry ───────────────────────────────────────────────────
    print("\nPhase 4: Vector geometry analysis...")
    geometry = {"layer_norms": {}, "cross_dim_cosine": {}}
    for dim, vec in vectors.items():
        norms = {str(lid): round(float(np.linalg.norm(vec.directions[lid])), 4)
                 for lid in sorted(vec.directions.keys())}
        geometry["layer_norms"][dim] = norms
        top3 = sorted(norms.items(), key=lambda x: -x[1])[:3]
        print(f"  {dim} top-3 layers: {', '.join(f'L{k}={v:.3f}' for k,v in top3)}")

    dims = list(vectors.keys())
    for i, d1 in enumerate(dims):
        for j, d2 in enumerate(dims):
            if j <= i:
                continue
            sims = []
            for lid in set(vectors[d1].directions.keys()) & set(vectors[d2].directions.keys()):
                v1, v2 = vectors[d1].directions[lid], vectors[d2].directions[lid]
                sims.append(float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)))
            avg_sim = round(float(np.mean(sims)), 4) if sims else 0
            geometry["cross_dim_cosine"][f"{d1}_vs_{d2}"] = avg_sim

    results["vector_geometry"] = geometry
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

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
