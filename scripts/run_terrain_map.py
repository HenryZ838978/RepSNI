#!/usr/bin/env python3
"""
Representation Terrain Map — Systematic RepEng dimension sweep.

Sweeps each of 5 RepEng dimensions from -3.0 to +3.0 at 0.2 step,
collects 4 metrics per point:
  1. Self-logprob (model's own confidence during generation)
  2. Token repetition rate (bigram/trigram overlap)
  3. Style markers (emoji density, exclamation density, avg sentence len)
  4. Semantic cosine similarity to baseline output (via bge-small-zh)

Saves raw data to JSON for visualization.
"""

import json, math, os, re, sys, time
import numpy as np
import httpx

BASE_URL = "http://localhost:8200"
MODEL = "MiniCPM4.1-8B-GPTQ"
OUT_DIR = "/cache/zhangjing/repeng_terrain"
STEP = 0.2
RANGE_MIN, RANGE_MAX = -3.0, 3.0
MAX_TOKENS = 150
DIMENSIONS = ["emotion_valence", "formality", "creativity", "confidence", "empathy"]

QUERIES = [
    {"id": "factual", "text": "用两句话告诉我什么是transformer", "type": "知识"},
    {"id": "news", "text": "最近有什么值得关注的AI新闻", "type": "开放"},
    {"id": "encourage", "text": "写一句鼓励人的话", "type": "情感"},
]

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001F9FF\U00002600-\U000027BF\U0000FE00-\U0000FEFF"
    "\U0001FA00-\U0001FA6F\U0001FA70-\U0001FAFF\U00002702-\U000027B0"
    "\U0001F600-\U0001F64F\U0001F680-\U0001F6FF]+"
)
EXCL_RE = re.compile(r"[！!]{1,}")
THINK_RE = re.compile(r"<think>.*?</think>\s*", re.DOTALL)
THINK_OPEN_RE = re.compile(r"<think>.*", re.DOTALL)
TAG_RE = re.compile(r"<\|?[a-zA-Z_/][^>]*\|?>")


def clean(text):
    text = THINK_RE.sub("", text)
    text = THINK_OPEN_RE.sub("", text)
    text = TAG_RE.sub("", text)
    return text.strip()


def compute_metrics(text, logprobs_data):
    """Compute all metrics for a single generation."""
    cleaned = clean(text)

    avg_logprob = 0.0
    if logprobs_data and logprobs_data.get("content"):
        lps = [t["logprob"] for t in logprobs_data["content"] if t.get("logprob") is not None]
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
    avg_sent_len = np.mean([len(s) for s in sentences]) if sentences else 0

    has_think = "<think>" in text
    think_closed = "</think>" in text
    think_fraction = 0.0
    if has_think:
        think_match = re.search(r"<think>(.*?)</think>", text, re.DOTALL)
        if think_match:
            think_fraction = len(think_match.group(1)) / len(text) if text else 0
        else:
            open_match = re.search(r"<think>(.*)", text, re.DOTALL)
            think_fraction = len(open_match.group(1)) / len(text) if open_match and text else 0

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


def generate(client, query, max_tokens=MAX_TOKENS):
    r = client.post(f"{BASE_URL}/v1/chat/completions", json={
        "model": MODEL,
        "messages": [{"role": "user", "content": query}],
        "max_tokens": max_tokens,
        "temperature": 0.01,
        "logprobs": True,
        "top_logprobs": 1,
        "extra_body": {"repetition_penalty": 1.15},
        "chat_template_kwargs": {"enable_thinking": False},
    }).json()
    content = r["choices"][0]["message"]["content"]
    logprobs = r["choices"][0].get("logprobs")
    tokens = r.get("usage", {}).get("completion_tokens", 0)
    return content, logprobs, tokens


def set_repeng(client, dim, val):
    if val == 0.0:
        client.post(f"{BASE_URL}/v1/repeng/reset")
    else:
        client.post(f"{BASE_URL}/v1/repeng/control",
                     json={"coefficients": {dim: val}})


def main():
    client = httpx.Client(timeout=30)
    results = {"metadata": {}, "baselines": {}, "sweeps": {}}

    results["metadata"] = {
        "model": MODEL,
        "step": STEP,
        "range": [RANGE_MIN, RANGE_MAX],
        "max_tokens": MAX_TOKENS,
        "dimensions": DIMENSIONS,
        "queries": QUERIES,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    steps = []
    v = RANGE_MIN
    while v <= RANGE_MAX + 1e-9:
        steps.append(round(v, 2))
        v += STEP

    total = len(DIMENSIONS) * len(steps) * len(QUERIES)
    print(f"Terrain Map: {len(DIMENSIONS)} dims × {len(steps)} steps × {len(QUERIES)} queries = {total} generations")
    print(f"Estimated time: {total * 2 / 60:.0f} minutes\n")

    print("Phase 1: Generating baselines (RepEng=0)...")
    client.post(f"{BASE_URL}/v1/repeng/reset")
    time.sleep(0.2)
    for q in QUERIES:
        content, logprobs, toks = generate(client, q["text"])
        metrics = compute_metrics(content, logprobs)
        results["baselines"][q["id"]] = {
            "query": q["text"],
            "metrics": metrics,
            "tokens": toks,
        }
        print(f"  [{q['id']}] {toks} tok, logprob={metrics['avg_logprob']:.3f}: {clean(content)[:80]}")

    print(f"\nPhase 2: Sweeping {len(DIMENSIONS)} dimensions...")
    done = 0
    for dim in DIMENSIONS:
        print(f"\n--- {dim} ---")
        results["sweeps"][dim] = []

        for val in steps:
            set_repeng(client, dim, val)
            time.sleep(0.05)

            point = {"value": val, "queries": {}}
            for q in QUERIES:
                content, logprobs, toks = generate(client, q["text"])
                metrics = compute_metrics(content, logprobs)
                point["queries"][q["id"]] = {"metrics": metrics, "tokens": toks}
                done += 1

            results["sweeps"][dim].append(point)

            agg_logprob = np.mean([point["queries"][q["id"]]["metrics"]["avg_logprob"] for q in QUERIES])
            agg_rep = np.mean([point["queries"][q["id"]]["metrics"]["trigram_rep"] for q in QUERIES])
            agg_emoji = sum(point["queries"][q["id"]]["metrics"]["emoji_count"] for q in QUERIES)

            bar = "#" * max(0, int((agg_logprob + 2) * 20))
            print(f"  {val:+.1f}  logp={agg_logprob:+.3f}  rep={agg_rep:.3f}  emoji={agg_emoji}  [{done}/{total}]  {bar}")

        client.post(f"{BASE_URL}/v1/repeng/reset")

    out_path = os.path.join(OUT_DIR, "terrain_data.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"\nSaved to {out_path}")

    print("\nPhase 3: Computing embeddings for cosine similarity...")
    try:
        from sentence_transformers import SentenceTransformer
        embed_model = SentenceTransformer(
            "/cache/zhangjing/voiceagent/models/bge-small-zh-v1.5",
            device="cuda:0",
        )
        baseline_texts = {
            qid: results["baselines"][qid]["metrics"]["cleaned_text"]
            for qid in results["baselines"]
        }
        baseline_embeds = {
            qid: embed_model.encode(text, normalize_embeddings=True)
            for qid, text in baseline_texts.items()
        }

        for dim in DIMENSIONS:
            for point in results["sweeps"][dim]:
                for q in QUERIES:
                    qid = q["id"]
                    steered_text = point["queries"][qid]["metrics"]["cleaned_text"]
                    if steered_text:
                        emb = embed_model.encode(steered_text, normalize_embeddings=True)
                        cos_sim = float(np.dot(baseline_embeds[qid], emb))
                    else:
                        cos_sim = 0.0
                    point["queries"][qid]["metrics"]["cosine_sim_to_baseline"] = round(cos_sim, 4)

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        print("Embeddings computed and saved.")
    except Exception as e:
        print(f"Embedding computation failed (non-critical): {e}")

    print("\nDone!")


if __name__ == "__main__":
    main()
