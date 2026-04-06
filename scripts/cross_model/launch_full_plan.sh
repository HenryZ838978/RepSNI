#!/bin/bash
set -e
export PYTHONUNBUFFERED=1

# ══════════════════════════════════════════════════════════════════════════════
# RepCI Full Experiment Plan: P0 → P1 → P2
# ══════════════════════════════════════════════════════════════════════════════
# GPU allocation: 4, 5, 6, 7 (4 GPUs × 24GB each)
#
# Phase A (P0): 1D terrain sweeps — BF16 vs AWQ quantization comparison
#   → Already completed or in-progress
#
# Phase B (P0): 2D Phase Diagrams — emotion×empathy, emotion×confidence, creativity×formality
#   → GPU 5 (reuse BF16 vectors)
#
# Phase C (P1): Thinking on/off dual terrain
#   → GPU 4 or 5 (whichever frees first)
#
# Phase D (P1): Cross-language query manifold (zh/en/mixed)
#   → GPU 4, 5, 6 in parallel (reuse vectors)
#
# Phase E (P1): Temperature decoding contrast (0.1/0.6/1.0/1.5)
#   → GPU 4, 5, 6, 7 in parallel (reuse vectors)
#
# Phase F (P2): Base vs Instruct terrain
#   → GPU 6 (need to download base model)
#
# Phase G (P2): Critical fluctuation measurement
#   → GPU 7 (uses cliff data from Phase A)
# ══════════════════════════════════════════════════════════════════════════════

PYTHON=/cache/zhangjing/miniconda3/envs/voiceagent/bin/python
SCRIPT_DIR="/cache/zhangjing/repeng_terrain/cross_model"
LOG_DIR="${SCRIPT_DIR}/logs"
MODEL_DIR="/cache/zhangjing/models"
TERRAIN="${SCRIPT_DIR}/run_terrain_generic.py"
PHASE2D="${SCRIPT_DIR}/run_phase_diagram.py"
FLUCT="${SCRIPT_DIR}/run_fluctuation.py"

mkdir -p "$LOG_DIR"

# Reference: BF16 vectors (from completed experiment)
BF16_VECS="${SCRIPT_DIR}/qwen3-8b-bf16/vectors"
BF16_DATA="${SCRIPT_DIR}/qwen3-8b-bf16/terrain_data.json"
MODEL_8B="${MODEL_DIR}/Qwen3-8B"
MODEL_8B_AWQ="${MODEL_DIR}/Qwen3-8B-AWQ"
MODEL_14B="${MODEL_DIR}/Qwen3-14B-AWQ"
MODEL_30B="${MODEL_DIR}/Qwen3-30B-A3B-GPTQ-Int4"

wait_for_gpu() {
    local gpu=$1
    echo "  Waiting for GPU $gpu to free up..."
    while true; do
        mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i "$gpu" 2>/dev/null)
        if [ "$mem" -lt 1000 ] 2>/dev/null; then
            echo "  GPU $gpu is free (${mem} MiB used)"
            return
        fi
        sleep 30
    done
}

wait_for_pids() {
    local label=$1
    shift
    echo "  Waiting for $label (PIDs: $@)..."
    for pid in "$@"; do
        if kill -0 "$pid" 2>/dev/null; then
            wait "$pid" 2>/dev/null || true
        fi
    done
    echo "  $label completed."
}

log_start() {
    echo ""
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
    echo "  $1"
    echo "  Started: $(date '+%Y-%m-%d %H:%M:%S')"
    echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
}

# ── Check prerequisites ──────────────────────────────────────────────────────

echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  RepCI Full Experiment Pipeline                                     ║"
echo "║  P0: Quantization + Phase Diagrams                                  ║"
echo "║  P1: Thinking + Cross-lang + Temperature                            ║"
echo "║  P2: Base vs Instruct + Critical Fluctuation                        ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Checking prerequisites..."

if [ ! -f "$BF16_DATA" ]; then
    echo "ERROR: BF16 terrain data not found at $BF16_DATA"
    echo "  Run the BF16 experiment first."
    exit 1
fi
if [ ! -d "$BF16_VECS" ]; then
    echo "ERROR: BF16 vectors not found at $BF16_VECS"
    exit 1
fi
echo "  ✓ BF16 terrain data + vectors found"

for m in "$MODEL_8B"; do
    if [ ! -f "$m/config.json" ]; then
        echo "ERROR: Model not found at $m"
        exit 1
    fi
done
echo "  ✓ Qwen3-8B model found"

# Wait for any ongoing experiments (AWQ on GPU 4, 30B download)
echo ""
echo "Checking for running experiments..."
AWQ_PID=$(pgrep -f "qwen3-8b-awq" 2>/dev/null || true)
if [ -n "$AWQ_PID" ]; then
    echo "  AWQ experiment running (PID $AWQ_PID), waiting for completion..."
    while kill -0 "$AWQ_PID" 2>/dev/null; do sleep 30; done
    echo "  AWQ experiment completed."
fi

# 30B download: launch MoE experiment in background watcher (non-blocking)
(
    DL30B_PID=$(pgrep -f "Qwen3-30B-A3B-GPTQ-Int4" 2>/dev/null || true)
    if [ -n "$DL30B_PID" ]; then
        echo "  [30B watcher] Download running (PID $DL30B_PID), watching in background..."
        while kill -0 "$DL30B_PID" 2>/dev/null; do sleep 60; done
        echo "  [30B watcher] Download completed."
    fi
    if [ -f "$MODEL_30B/config.json" ] && [ ! -f "${SCRIPT_DIR}/qwen3-30b-a3b-moe/terrain_data.json" ]; then
        echo "  [30B watcher] Launching 30B MoE experiment on GPU 6..."
        wait_for_gpu 6
        $PYTHON "$TERRAIN" \
            --model "$MODEL_30B" \
            --gpu 6 --tag qwen3-30b-a3b-moe \
            --batch-size 2 \
            > "$LOG_DIR/exp_30b_moe.log" 2>&1
        echo "  [30B watcher] 30B MoE experiment completed."
    fi
) &
PID_30B_WATCHER=$!
echo "  30B watcher background PID: $PID_30B_WATCHER"
echo ""

# ══════════════════════════════════════════════════════════════════════════════
# PHASE B (P0): 2D Phase Diagrams
# ══════════════════════════════════════════════════════════════════════════════
log_start "PHASE B: 2D Phase Diagrams (11×11 grid, 3 pairs)"

wait_for_gpu 5

nohup $PYTHON "$PHASE2D" \
    --model "$MODEL_8B" \
    --vector-dir "$BF16_VECS" \
    --pairs "emotion_valence:empathy,emotion_valence:confidence,creativity:formality" \
    --resolution 11 \
    --gpu 5 \
    --tag phase2d-8b-bf16 \
    > "$LOG_DIR/exp_phase2d.log" 2>&1 &
PID_PHASE2D=$!
echo "  PID: $PID_PHASE2D (GPU 5)"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE C (P1): Thinking on/off
# ══════════════════════════════════════════════════════════════════════════════
log_start "PHASE C: Thinking On/Off Dual Terrain"

wait_for_gpu 4

# Thinking OFF (baseline — same as existing, but with explicit flag for consistency)
nohup $PYTHON "$TERRAIN" \
    --model "$MODEL_8B" \
    --gpu 4 --tag thinking-off-8b \
    --vector-dir "$BF16_VECS" --skip-training \
    > "$LOG_DIR/exp_thinking_off.log" 2>&1 &
PID_THINK_OFF=$!
echo "  Thinking OFF PID: $PID_THINK_OFF (GPU 4)"

# Wait for GPU 4 to finish thinking-off, then run thinking-on
wait_for_pids "thinking-off" $PID_THINK_OFF

nohup $PYTHON "$TERRAIN" \
    --model "$MODEL_8B" \
    --gpu 4 --tag thinking-on-8b \
    --vector-dir "$BF16_VECS" --skip-training \
    --thinking \
    > "$LOG_DIR/exp_thinking_on.log" 2>&1 &
PID_THINK_ON=$!
echo "  Thinking ON PID: $PID_THINK_ON (GPU 4)"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE D (P1): Cross-language queries
# After thinking-on finishes on GPU 4, phase2d finishes on GPU 5
# ══════════════════════════════════════════════════════════════════════════════
log_start "PHASE D: Cross-Language Query Manifold (en + mixed)"
echo "  (zh already done in BF16 baseline)"

wait_for_pids "thinking-on" $PID_THINK_ON

# English queries on GPU 4
nohup $PYTHON "$TERRAIN" \
    --model "$MODEL_8B" \
    --gpu 4 --tag lang-en-8b \
    --vector-dir "$BF16_VECS" --skip-training \
    --lang en \
    > "$LOG_DIR/exp_lang_en.log" 2>&1 &
PID_LANG_EN=$!
echo "  English PID: $PID_LANG_EN (GPU 4)"

# Wait for phase2d to finish to free GPU 5
wait_for_pids "phase2d" $PID_PHASE2D

# Mixed queries on GPU 5
nohup $PYTHON "$TERRAIN" \
    --model "$MODEL_8B" \
    --gpu 5 --tag lang-mixed-8b \
    --vector-dir "$BF16_VECS" --skip-training \
    --lang mixed \
    > "$LOG_DIR/exp_lang_mixed.log" 2>&1 &
PID_LANG_MIX=$!
echo "  Mixed PID: $PID_LANG_MIX (GPU 5)"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE E (P1): Temperature decoding contrast
# Run 4 temperatures on GPUs 4-7 in parallel
# ══════════════════════════════════════════════════════════════════════════════
log_start "PHASE E: Temperature Decoding Contrast"

wait_for_pids "cross-lang" $PID_LANG_EN $PID_LANG_MIX

wait_for_gpu 4
nohup $PYTHON "$TERRAIN" \
    --model "$MODEL_8B" \
    --gpu 4 --tag temp-0.1-8b \
    --vector-dir "$BF16_VECS" --skip-training \
    --temperature 0.1 \
    > "$LOG_DIR/exp_temp01.log" 2>&1 &
PID_T01=$!
echo "  temp=0.1 PID: $PID_T01 (GPU 4)"

wait_for_gpu 5
nohup $PYTHON "$TERRAIN" \
    --model "$MODEL_8B" \
    --gpu 5 --tag temp-0.6-8b \
    --vector-dir "$BF16_VECS" --skip-training \
    --temperature 0.6 \
    > "$LOG_DIR/exp_temp06.log" 2>&1 &
PID_T06=$!
echo "  temp=0.6 PID: $PID_T06 (GPU 5)"

wait_for_gpu 6
nohup $PYTHON "$TERRAIN" \
    --model "$MODEL_8B" \
    --gpu 6 --tag temp-1.0-8b \
    --vector-dir "$BF16_VECS" --skip-training \
    --temperature 1.0 \
    > "$LOG_DIR/exp_temp10.log" 2>&1 &
PID_T10=$!
echo "  temp=1.0 PID: $PID_T10 (GPU 6)"

wait_for_gpu 7
nohup $PYTHON "$TERRAIN" \
    --model "$MODEL_8B" \
    --gpu 7 --tag temp-1.5-8b \
    --vector-dir "$BF16_VECS" --skip-training \
    --temperature 1.5 \
    > "$LOG_DIR/exp_temp15.log" 2>&1 &
PID_T15=$!
echo "  temp=1.5 PID: $PID_T15 (GPU 7)"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE F (P2): Base vs Instruct
# Download Qwen2.5-7B (base) via ModelScope, then run terrain
# ══════════════════════════════════════════════════════════════════════════════
log_start "PHASE F: Base vs Instruct Terrain"

# Download base model while temperature experiments run
BASE_MODEL="${MODEL_DIR}/Qwen2.5-7B"
if [ ! -f "$BASE_MODEL/config.json" ]; then
    echo "  Downloading Qwen2.5-7B (base) via ModelScope..."
    $PYTHON -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B', local_dir='${BASE_MODEL}')
print('DONE: Qwen2.5-7B base')
" > "$LOG_DIR/dl_qwen25_base.log" 2>&1
    echo "  Download complete."
else
    echo "  Qwen2.5-7B base already exists."
fi

# Also download Qwen2.5-7B-Instruct for fair comparison
INSTRUCT_MODEL="${MODEL_DIR}/Qwen2.5-7B-Instruct"
if [ ! -f "$INSTRUCT_MODEL/config.json" ]; then
    echo "  Downloading Qwen2.5-7B-Instruct via ModelScope..."
    $PYTHON -c "
from modelscope import snapshot_download
snapshot_download('Qwen/Qwen2.5-7B-Instruct', local_dir='${INSTRUCT_MODEL}')
print('DONE: Qwen2.5-7B-Instruct')
" > "$LOG_DIR/dl_qwen25_instruct.log" 2>&1
    echo "  Download complete."
else
    echo "  Qwen2.5-7B-Instruct already exists."
fi

wait_for_pids "temperature-all" $PID_T01 $PID_T06 $PID_T10 $PID_T15

wait_for_gpu 4
nohup $PYTHON "$TERRAIN" \
    --model "$BASE_MODEL" \
    --gpu 4 --tag qwen25-7b-base \
    > "$LOG_DIR/exp_base.log" 2>&1 &
PID_BASE=$!
echo "  Base model PID: $PID_BASE (GPU 4)"

wait_for_gpu 5
nohup $PYTHON "$TERRAIN" \
    --model "$INSTRUCT_MODEL" \
    --gpu 5 --tag qwen25-7b-instruct \
    > "$LOG_DIR/exp_instruct.log" 2>&1 &
PID_INSTRUCT=$!
echo "  Instruct model PID: $PID_INSTRUCT (GPU 5)"

# ══════════════════════════════════════════════════════════════════════════════
# PHASE G (P2): Critical Fluctuation Measurement
# ══════════════════════════════════════════════════════════════════════════════
log_start "PHASE G: Critical Fluctuation Measurement"

wait_for_gpu 6
nohup $PYTHON "$FLUCT" \
    --model "$MODEL_8B" \
    --vector-dir "$BF16_VECS" \
    --terrain-data "$BF16_DATA" \
    --gpu 6 --tag fluctuation-8b-bf16 \
    --n-samples 20 --temperature 0.7 \
    > "$LOG_DIR/exp_fluctuation.log" 2>&1 &
PID_FLUCT=$!
echo "  Fluctuation PID: $PID_FLUCT (GPU 6)"

# ══════════════════════════════════════════════════════════════════════════════
# WAIT FOR ALL
# ══════════════════════════════════════════════════════════════════════════════
log_start "Waiting for all experiments to complete..."

wait_for_pids "base-instruct" $PID_BASE $PID_INSTRUCT
wait_for_pids "fluctuation" $PID_FLUCT

# ══════════════════════════════════════════════════════════════════════════════
# FINAL: Run comprehensive analysis
# ══════════════════════════════════════════════════════════════════════════════
log_start "Running comprehensive cross-experiment analysis"

$PYTHON "$SCRIPT_DIR/analyze_cross_model.py" 2>&1 | tee "$LOG_DIR/final_analysis.log"

echo ""
echo "╔══════════════════════════════════════════════════════════════════════╗"
echo "║  ALL EXPERIMENTS COMPLETE                                           ║"
echo "║  Results: $SCRIPT_DIR/                                              ║"
echo "╚══════════════════════════════════════════════════════════════════════╝"
echo ""
echo "Key output files:"
find "$SCRIPT_DIR" -name "*.json" -newer "$BF16_DATA" | sort
echo ""
echo "Finished at: $(date '+%Y-%m-%d %H:%M:%S')"
