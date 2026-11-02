#!/usr/bin/env bash
# GPU session 11, part A: the second model (Phi-3.5-MoE-instruct) through the main model's experiments 1 and 2
# (user decision 2026-10-13: lesion the second model too; docs/plans/gpu-session11.md). Same scripts and rules as
# for Qwen: means on neutral text, control screen on M1 (attribution, single-component screen, joint ablation),
# knowledge screen on the naming test (item check, screen, expansion to 8), then the control test and the
# dissociation test on held-out items with 200 matched random sets each. Exploratory.
# Run on the server from the repo root, HEAD at the tag below, clean tree, one GPU:
#     GPU=0 nohup bash experiments/second_model/run_phi_replication.sh >> /root/autodl-tmp/logs/s11_phi.log 2>&1 &
# Precision: float16 as for Qwen if the prefix check passes in float16 (it fails on overflow, i.e. non-finite
# scores); otherwise bfloat16. Each finished step is skipped on a rerun.
# PHI_PRECISION=float16|bfloat16: use that precision without the check (its means must exist). Added 2026-10-14 after
# both checks failed on the real model: float16 had the smaller noise floor (0.26 against 0.40 logit, full scoring at
# batch 16 vs 32) and failed only the item-level move correlation of one single expert whose own moves (0.03) are far
# below that noise; the experiment statistics average over items and batch compositions (docs/plans/gpu-session11.md).
# PHI_TINY=1, PHI_PY, PHI_LOG, PHI_TAG: local rehearsal on the tiny random model only.
set -uo pipefail
source experiments/session_lib.sh
GPU=${GPU:-0}
PY=${PHI_PY:-/root/miniconda3/envs/lesion/bin/python}
MODEL=phi35-moe-instruct
TAG=${PHI_TAG:-analysis-frozen-session11}
LOG=${PHI_LOG:-/root/autodl-tmp/logs/s11_phi}
SCORER=prefix_avg3_small
if [ -n "${PHI_TINY:-}" ]; then
    TINY="--tiny"; SUF=_tiny; RANDOM_ARGS="--n-random 3 --n-boot 200"; ITEMS_ARGS="--tiny --device cpu"; MEANS_ARGS="--tiny --max-tokens 64"
else
    TINY=""; SUF=""; RANDOM_ARGS=""; ITEMS_ARGS=""; MEANS_ARGS=""
fi
CONTROL_SCREEN=outputs/e1_screen$SUF/$MODEL
KNOWLEDGE_SCREEN=outputs/e3_knowledge$SUF/$MODEL
HELD_OUT=outputs/e2_confirm$SUF/$MODEL
mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

require_tag "$TAG"
nvidia-smi -L 2>/dev/null
echo "$(stamp) start on GPU $GPU"

# use_precision DTYPE TAG: the means file and the arguments every step shares
use_precision() {
    DT=$1
    MEANS=outputs/means$SUF/${MODEL}@4368845__$2.pt
    ARGS="--model $MODEL --means $MEANS --dtype $DT --scorer $SCORER --batch-size 16 $TINY"
}
# precision: float16 if its means are finite and the prefix check passes in float16, else bfloat16
choose_precision() {
    local pair tag
    for pair in float16:fp16 bfloat16:bf16; do
        tag=${pair##*:}
        use_precision "${pair%%:*}" "$tag"
        run "means_$tag" "$MEANS" $PY experiments/prep_means/run.py --model $MODEL --dtype $DT --batch-size 8 --output $MEANS $MEANS_ARGS || continue
        if ! $PY -c "import torch,sys; d=torch.load(sys.argv[1], weights_only=False); sys.exit(0 if all(torch.isfinite(d[k]).all() for k in ('expert_mean','head_mean')) else 1)" "$MEANS"; then
            echo "$(stamp) means in $DT are not finite"; continue
        fi
        if run "check_prefix_$tag" "" $PY experiments/e1_screen/check_prefix.py $ARGS --n-experts 4 \
           && $PY -c "import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))['pass'] else 1)" "$CONTROL_SCREEN/check_prefix/check_prefix.json"; then
            echo "$(stamp) precision: $DT"; return 0
        fi
        echo "$(stamp) prefix check failed in $DT"
        rm -rf "$CONTROL_SCREEN/check_prefix_failed_$tag"; mv "$CONTROL_SCREEN/check_prefix" "$CONTROL_SCREEN/check_prefix_failed_$tag" 2>/dev/null
    done
    return 1
}
if [ -n "${PHI_PRECISION:-}" ]; then
    case "$PHI_PRECISION" in float16) use_precision float16 fp16;; bfloat16) use_precision bfloat16 bf16;; *) echo "bad PHI_PRECISION"; exit 1;; esac
    [ -e "$MEANS" ] || { echo "$(stamp) STOPPED: $MEANS is missing"; exit 1; }
    echo "$(stamp) precision (set by PHI_PRECISION, prefix check not required): $DT"
elif [ -e "$CONTROL_SCREEN/check_prefix/check_prefix.json" ]; then
    # resumed: the precision the check passed in
    if [ -e "$CONTROL_SCREEN/check_prefix_failed_fp16" ]; then use_precision bfloat16 bf16; else use_precision float16 fp16; fi
    echo "$(stamp) precision (resumed): $DT"
else
    choose_precision || { echo "$(stamp) STOPPED: no precision passed the prefix check"; exit 1; }
fi

FAIL=0
# control side (experiment 1)
run attribution "$(finished $CONTROL_SCREEN/attribution)" $PY experiments/e1_screen/run_attribution.py $ARGS --kinds expert,head --batch-size 4 || exit 1
run screen "$(finished $CONTROL_SCREEN/screen)" $PY experiments/e1_screen/run_screen.py $ARGS --kind expert+head \
    --candidates $CONTROL_SCREEN/attribution/attribution.csv --top 32 --flush-every 8 || exit 1
run joint "$(finished $CONTROL_SCREEN/joint)" $PY experiments/e1_screen/run_joint.py $ARGS $RANDOM_ARGS || FAIL=1
# knowledge side (experiment 2)
run check_items "$(ls $KNOWLEDGE_SCREEN/check/summary.json 2>/dev/null)" $PY experiments/e3_knowledge/check_items.py \
    --model $MODEL --dtype $DT --batch-size 16 $ITEMS_ARGS || exit 1
$PY -c "import json,sys; print('knowledge items, intact frequency effect passed:', json.load(open(sys.argv[1]))['passed'])" "$KNOWLEDGE_SCREEN/check/summary.json"
run knowledge "$(finished $KNOWLEDGE_SCREEN/screen)" $PY experiments/e3_knowledge/run_knowledge.py $ARGS $RANDOM_ARGS || exit 1
run expand8 "$(finished $KNOWLEDGE_SCREEN/expand8_dev120)" $PY experiments/e3_knowledge/run_expand8.py $ARGS $RANDOM_ARGS \
    --knowledge-dir $KNOWLEDGE_SCREEN/screen --out $KNOWLEDGE_SCREEN/expand8_dev120 || FAIL=1
# held-out tests with 200 matched random sets each
run control_test "$(finished $HELD_OUT/control)" $PY experiments/e2_confirm/run_control_test.py $ARGS $RANDOM_ARGS \
    --freeze-tag $TAG --out $HELD_OUT/control || FAIL=1
run dissociation_test "$(finished $HELD_OUT/dissociation)" $PY experiments/e2_confirm/run_dissociation_test.py $ARGS $RANDOM_ARGS \
    --freeze-tag $TAG --knowledge-dir $KNOWLEDGE_SCREEN/screen --out $HELD_OUT/dissociation || FAIL=1
echo "$(stamp) done (any step failed: $FAIL)"
exit $FAIL
