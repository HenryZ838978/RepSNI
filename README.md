# Semantic Nebula Imaging &nbsp;语义星云成像

**Quantifying the structure that practitioners feel but cannot see.**

> Every ML engineer has intuitions about models — *"this one is robust,"* *"that one degrades unpredictably,"* *"alignment makes it safer but less expressive."*  
> These intuitions are real. They reflect the topological structure of the model's representation manifold — a structure that exists in thousands of dimensions, invisible to any single metric.  
> **SNI makes it visible.**

<p align="center">
  <img src="figures/sni/sni_base_vs_instruct.png" width="100%" alt="Qwen2.5-7B: Base (red, 143 cliffs) vs Instruct (blue, 0 cliffs)">
  <br>
  <em>Same model, before and after alignment. Red = danger zones (repetition collapse). Blue-green = safe operating space.</em>
</p>

Using [Representation Engineering](https://arxiv.org/abs/2310.01405) control vectors as **contrast agents**, SNI sweeps a model's 5-dimensional personality space (emotion, formality, creativity, confidence, empathy), measures where output quality degrades, and projects the resulting hidden-state geometry into an interactive 3D point cloud — a **nebula** whose shape, color, and density encode the model's structural health.

**Color** = safety. Blue is stable. Red is cliff territory — one more step and the model collapses into repetition loops.  
**Shape** = architecture. A round nebula means the personality dimensions are well-separated. A compressed streak means they've collapsed into one axis.

---

## Part I — What This Reveals

### 1. Stable vs. Fragile Architecture

<p align="center">
  <img src="figures/sni/sni_qwen3_vs_minicpm.png" width="100%" alt="Qwen3-8B (Stable) vs MiniCPM4.1-8B (Fragile)">
</p>

|  | **Qwen3-8B** | **MiniCPM4.1-8B** |
|---|---|---|
| Shape | Spherical, multi-dimensional | Compressed "galaxy band" |
| PCA variance distribution | 58.2 + 20.9 + 8.0% | 83.1 + 10.8 + 3.3% |
| PC1:PC2 ratio | **2.7 : 1** | **7.7 : 1** |
| Structure | Multi-dimensional · Stable | Compressed · Collapse-prone |
| Avg danger | 0.042 | 0.019 (low on-axis, high off-axis) |

**What the nebula tells you**: Qwen3-8B distributes its representation capacity across all personality dimensions roughly equally — you can steer emotion, formality, creativity independently without interference. MiniCPM4.1-8B has squeezed 83% of its personality variance into a single principal component. It's efficient along that axis (hence low average danger), but **any perturbation off the main track triggers collapse**. The bright streak you see is the one safe corridor; the sparse scatter around it is where the model breaks.

This is exactly what practitioners report: MiniCPM4.1 is capable when used normally but notoriously unstable when pushed. **Now you can see the geometry behind the instability.**

The side-view angle makes the dimensional collapse especially clear:

<p align="center">
  <img src="figures/sni/compare_qwen3_minicpm_side.png" width="100%" alt="Side view: 3D sphere vs flat disk">
  <br>
  <em>Side view — Qwen3's spherical distribution vs MiniCPM's compressed plane.</em>
</p>

### 2. The Alignment Effect: Burning Star → Cool Nebula

<p align="center">
  <img src="figures/sni/sni_base_vs_instruct.png" width="100%" alt="Base vs Instruct">
</p>

|  | **Qwen2.5-7B Base** | **Qwen2.5-7B Instruct** |
|---|---|---|
| Color | Fiery red-orange | Cool blue-green |
| Cliffs | **143** | **0** |
| Avg danger | 0.211 | 0.019 |
| Danger reduction | — | **12× lower** |
| PC1:PC2 ratio | 1.5 : 1 | 1.5 : 1 |

**The base model is a burning star.** Nearly the entire representation space is above the safety threshold. There are 143 cliff points — positions where a 0.2-step change in any personality coefficient causes output quality to drop catastrophically (trigram repetition > 8%).

**After SFT + RLHF, it becomes a cool nebula.** Zero cliffs. Average danger drops 12×. The manifold shape (PC ratio) is preserved — alignment doesn't restructure the space, it **smooths the surface**. The red zones haven't disappeared; they've been suppressed below the danger threshold.

The visual analogy is precise: alignment acts as a **cooling process** on the representation manifold. The underlying topology persists, but the extreme states are no longer accessible to normal decoding.

### 3. Chain-of-Thought as Internal Normalization

<p align="center">
  <img src="figures/sni/sni_thinking_off_vs_on.png" width="100%" alt="Thinking OFF vs ON">
</p>

|  | **Thinking OFF** | **Thinking ON** |
|---|---|---|
| Cliffs | 4 | **0** |
| Avg danger | 0.042 | **0.000** |
| Color | Blue-green core + yellow-orange halo | Uniform ethereal blue |
| PC1:PC2 | 2.7 : 1 | 2.7 : 1 |

Same model (Qwen3-8B), same control vectors, same coefficients. The only difference: whether `<think>` mode is enabled.

**Thinking ON produces zero cliffs and near-zero danger across the entire manifold.** The CoT reasoning chain acts as an internal normalization mechanism that prevents the model from ever reaching repetition collapse — regardless of how hard you push the personality coefficients.

The nebula transforms from a dense, multi-colored cloud (with localized danger zones) into an almost perfectly uniform blue mist. This is the first visual evidence that CoT reasoning **fundamentally changes the topology of what the model can express**, not just how it formats the output.

---

## Part II — How It Works

### The Principle: Contrast Imaging for Neural Networks

The method is borrowed from radiology. In CT imaging, a contrast agent (iodine) is injected into the body, then scanned. Structures that were invisible — blood vessels, tumors — light up because the agent accumulates differently in different tissues.

**SNI does the same to LLMs:**

| CT Imaging | Semantic Nebula Imaging |
|---|---|
| Contrast agent (iodine) | RepEng control vector |
| Injection dose (mL) | Coefficient α ∈ [−3.0, +3.0] |
| X-ray scan | Output quality metrics (repetition, cosine similarity, logprob) |
| Tissue boundary | Domain boundary in representation space |
| Tumor lighting up | Repetition cliff / cosine drop |
| Organ structure | Pretrain data topology on the manifold |

### The Mathematics

Representation Engineering modifies hidden states by linear injection:

$$h' = h + \alpha \cdot v$$

where $h$ is the original hidden state, $v$ is a pre-trained control vector for a semantic dimension (e.g., *emotion_valence*), and $\alpha$ is the coefficient.

**If the representation manifold were uniformly smooth, the dose-response curve (α → output quality) would also be smooth.** Discontinuities in the curve reveal discontinuities in the manifold — domain boundaries, attractor basins, and topological features of the pretrain data distribution.

### The Pipeline

```
Control Vectors (GGUF)     Terrain Data (JSON)
        │                         │
        ▼                         ▼
┌─────────────────────────────────────────┐
│  1. Sample 5D coefficient space         │
│     (50K points: uniform + Gaussian     │
│      + spherical shells + axis sweeps)  │
├─────────────────────────────────────────┤
│  2. Project to hidden-state space       │
│     h(α₁..α₅) = Σ αᵢ · vᵢ             │
├─────────────────────────────────────────┤
│  3. PCA → 3D                            │
│     Eigendecomposition of covariance    │
│     Top 3 principal components          │
├─────────────────────────────────────────┤
│  4. Color by danger level               │
│     Fit quadratic manifold model:       │
│     trigram_rep(α) = β₀ + Σβᵢαᵢ + ...  │
│     Map predicted rep → color gradient  │
├─────────────────────────────────────────┤
│  5. Render interactive 3D nebula        │
│     Three.js + WebGL additive blending  │
└─────────────────────────────────────────┘
        │
        ▼
  Self-contained HTML (one file, ~350KB)
```

### Why This Is Valid

**Q: You're projecting from 5D to 3D. Don't you lose information?**

PCA captures 70–97% of the variance in 3 components (model-dependent). The remaining variance is small — and the fact that different models produce *visually distinct* nebulae that correctly predict known model properties (MiniCPM instability, alignment smoothing, CoT immunity) is strong empirical evidence that the projection preserves the diagnostically relevant structure.

**Q: The danger colors come from a fitted model, not direct measurement.**

For models with terrain data (dose-response sweeps), the quadratic manifold model is fitted to hundreds of actual output measurements. The model's $R²$ is typically > 0.85. For the points between measured sweep positions, the smooth interpolation is a conservative estimate — real cliffs are sharper than the model predicts.

**Q: Are the structural differences real, or artifacts of PCA?**

The PC1:PC2 ratio is PCA-independent in the sense that it measures actual variance concentration, not an artifact of the projection method. MiniCPM4.1's 7.7:1 ratio means that 83% of the control vector response is genuinely along one axis in the full-dimensional hidden state space. The 3D rendering makes this fact visible, it doesn't create it.

### Validated Across 14 Models

The cross-model study covers the full spectrum of conditions:

| Variable | Models Tested | Key Finding |
|---|---|---|
| Architecture | Qwen3-8B, Qwen2.5-7B, Qwen3-14B, MiniCPM4.1-8B | Structure varies dramatically |
| Alignment | Base vs Instruct | 12× danger reduction, 0 cliffs |
| Reasoning | Thinking OFF vs ON | Complete cliff immunity |
| Quantization | BF16 vs AWQ-Int4 | Nearly identical topology |
| Scale | 8B vs 14B | Slightly smoother at larger scale |
| Language | Chinese / English / Mixed | English amplifies cliffs 3.7× |
| Temperature | 0.1 / 0.6 / 1.0 / 1.5 | Cliff position stable, shape changes |

Total: **13 experiments, 6,045 generations, 92 cliffs detected.**

---

## Part III — What It's For

### Model Comparison at a Glance

Instead of running benchmarks that compress a model's behavior into a single number, SNI gives you a **holistic structural portrait**. Two models with identical perplexity scores can have radically different nebula shapes — one spherical (robust), one compressed (fragile). The nebula tells you things that benchmarks miss.

### Training Stage Diagnostics

By imaging the same model at different training checkpoints (pretrain → SFT → RLHF), you can **watch the manifold cool down** in real time. This provides a new quality signal for alignment teams: is your RLHF actually smoothing the dangerous regions, or just masking them?

### Safety Envelope Visualization

The terrain data underlying each nebula defines a **per-dimension safe operating range** — the region where personality steering doesn't trigger output collapse. This is more informative than a single L2-norm safety radius, because the safe envelope is asymmetric and dimension-dependent:

```
                    SAFE RANGE (Qwen3-8B)
                    ────────────────────────────────
emotion_valence     [−3.0 ████████████████████▓░░░░░░░ +1.8]
formality           [−3.0 ████████████████████████████████ +3.0]
creativity          [−1.0 ███████████████████████████████ +3.0]
confidence          [−3.0 ████████████████████████████████ +3.0]
empathy             [−0.4 ████ +0.0]  ← narrowest
```

### Navigation on the Manifold (Next Step)

The terrain map defines where cliffs are. The next step is **using it for real-time adaptive steering** — finding optimal paths between personality states that stay within the safe envelope, avoid cliffs, and maximize expressiveness.

We have a working `ManifoldPathPlanner` that computes geodesic-like paths with boundary repulsion on the 5D manifold. The zero-hysteresis property (confirmed experimentally: all paths are reversible) simplifies this to pure geometry.

**The vision**: instead of manually tuning RepEng coefficients and hoping you don't hit a cliff, the navigator plans a smooth trajectory from state A to state B — like GPS routing on a mountain road, staying on the safe ridges and avoiding the drops.

---

## Quick Start

### Requirements

- Python 3.10+
- `repeng` library (for GGUF control vector loading)
- `numpy`
- Pre-trained control vectors (5 GGUF files per model)
- Terrain data (optional, from dose-response sweep)

### Generate a Nebula

```bash
# Single model
python scripts/sni_pipeline.py --tag qwen3-8b-bf16

# Batch all available models
python scripts/sni_pipeline.py --batch

# Side-by-side comparison
python scripts/sni_pipeline.py --compare qwen3-8b-bf16 minicpm41-8b
```

Output: self-contained HTML files (~350KB each) that open in any browser. No server required.

### Run a Terrain Sweep (to generate the underlying data)

```bash
# Start RepEng-patched vLLM server
python -m repeng_vllm.start_repeng_vllm \
  --model /path/to/model \
  --trust-remote-code --dtype auto \
  --repeng-vector-dir vectors/

# Run the sweep (~20 min per model)
python scripts/run_terrain_map.py
```

### Data Files

| File | Description |
|---|---|
| `data/sni/*.json` | Compact point cloud data (8K points per model, ~100–330KB) |
| `data/terrain_data.json` | Original MiniCPM4.1 dose-response sweep (465 generations) |
| `data/cross_model/` | Cross-model terrain data (13 experiments) |

---

## Repository Structure

```
├── README.md
├── scripts/
│   ├── sni_pipeline.py          # End-to-end: vectors → point cloud → HTML
│   ├── run_terrain_map.py       # Dose-response sweep runner
│   └── analyze_terrain.py       # Statistical analysis & cliff detection
├── data/
│   ├── sni/                     # Compact nebula data (JSON)
│   ├── terrain_data.json        # MiniCPM4.1 raw sweep data
│   └── cross_model/             # Cross-model validation data
└── figures/
    ├── sni/                     # Nebula screenshots & renders
    └── *.svg                    # Terrain map visualizations
```

---

## Interactive Visualizations

The compact JSON data in `data/sni/` can be rendered into interactive 3D nebulae using the pipeline:

```bash
python scripts/sni_pipeline.py --tag qwen3-8b-bf16
# Opens: sni_output/sni_qwen3-8b-bf16.html
```

Each HTML file is self-contained (Three.js loaded from CDN) and supports:
- Mouse drag to rotate, scroll to zoom
- Auto-rotation
- Real-time particle rendering with additive blending
- Dimension paths (colored traces through the 5 personality axes)
- Cliff markers (bright red points where output collapses)

---

## Relation to Representation Contrast Imaging

This repository evolves from our earlier work on **Representation Contrast Imaging (表征显影)** — 1D/2D dose-response terrain mapping that first revealed topological discontinuities in LLM hidden states.

SNI builds on that foundation by projecting the full 5D personality manifold into 3D, replacing terrain-map line charts with holistic point-cloud visualization. The underlying data and methodology are the same; the imaging modality is new.

**Terrain mapping** = measuring individual cross-sections of the manifold (like 1D slices through a CT scan).  
**Semantic Nebula Imaging** = reconstructing the full 3D volume from those cross-sections (like a full 3D CT reconstruction).

---

## Citation

```bibtex
@misc{sni2026,
  title   = {Semantic Nebula Imaging: Visualizing the Topological Structure
             of LLM Representation Manifolds},
  author  = {CyberWizard},
  year    = {2026},
  url     = {https://github.com/HenryZ838978/RepSNI}
}
```

---

## Related Work

- [Representation Engineering](https://arxiv.org/abs/2310.01405) (Zou et al., 2023) — Control vectors for LLM steering
- [RepEngvLLM](https://github.com/HenryZ838978/RepEngvLLM) — vLLM patch for runtime RepEng injection
- [repeng](https://github.com/vgel/repeng) — Python library for control vector training

---

<sub>Built during a multi-day investigation into LLM personality steering and manifold topology. 14 models imaged. 6,045 generations analyzed. The map is getting sharper — the territory is real.</sub>
