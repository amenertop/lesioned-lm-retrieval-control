#!/usr/bin/env bash
# GPU session 11, part B: exploratory re-screens on the main model (user decisions 2026-10-13;
# docs/plans/gpu-session11.md), one GPU:
# 1. knowledge-store screen: components whose ablation hurts naming and comprehension together
#    (experiments/e3_knowledge/run_store_screen.py) -> the set S8;
# 2. semantic-only control set from the existing control screen, no new scoring (select_semantic_control.py) -> C4s;
# 3. held-out items with 200 matched random sets: the dissociation test with S8 and C4s in place of K8 and C4 (naming
#    and M1 for both new sets, random sets matched to the knowledge-side set), and the control test for C4s (M1
#    control effect and string damage against random sets matched to C4s);
# 4. experiment 3's battery and the letter-cued naming for intact, K8, C4, C4s and S8 (run_new_sets.py).
# The screens and selections of the main model were made at older commits; the shared code changed since only to
# support the second model, and gives identical scores on the main model (checked 2026-10-13), so the held-out
# tests run with --allow-score-code-change, which records the changed files.
# Run on the server from the repo root, HEAD at the tag below, clean tree:
#     GPU=1 nohup bash experiments/e3_errors/run_qwen_rescreens.sh >> /root/autodl-tmp/logs/s11_qwen.log 2>&1 &
set -uo pipefail
source experiments/session_lib.sh
GPU=${GPU:-1}
PY=/root/miniconda3/envs/lesion/bin/python
MODEL=qwen3-30b-a3b-instruct
TAG=${S11_TAG:-analysis-frozen-session11}
LOG=/root/autodl-tmp/logs/s11_qwen
MEANS=outputs/means/qwen3-30b-a3b-instruct@0d7cf23__42ae809__fp16.pt
ARGS="--model $MODEL --means $MEANS --dtype float16 --scorer prefix_avg3 --batch-size 16"
HELD_OUT=outputs/e2_confirm/$MODEL
mkdir -p "$LOG"
export CUDA_VISIBLE_DEVICES=$GPU HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

require_tag "$TAG"
nvidia-smi -L
echo "$(stamp) start on GPU $GPU"
FAIL=0

# 1. knowledge-store screen
STORE_SCREEN=outputs/e3_knowledge/$MODEL/store_screen
run store_screen "$(finished $STORE_SCREEN)" $PY experiments/e3_knowledge/run_store_screen.py $ARGS || FAIL=1
STORE=$($PY -c "import json,sys; print(','.join(json.load(open(sys.argv[1]))['store']))" $STORE_SCREEN/selection.json 2>/dev/null)
echo "$(stamp) store set: ${STORE:-none}"
# 2. semantic-only control set (no scoring)
C4S=$($PY experiments/e1_screen/select_semantic_control.py --model $MODEL 2>> "$LOG/select_semantic_control.log")
echo "$(stamp) semantic-only control set: ${C4S:-none}"
K8=$($PY -c "import sys; sys.path.insert(0, 'experiments/e2_confirm'); import run_dissociation_test as r; print(','.join(r.K8))" 2>/dev/null)
C4=$($PY -c "import json; print(','.join(json.load(open('outputs/e1_screen/$MODEL/screen/selection.json'))['control']))")

# 3. held-out items, 200 matched random sets each
if [ -n "$STORE" ] || [ -n "$C4S" ]; then
    run dissociation_new "$(finished $HELD_OUT/dissociation_new)" $PY experiments/e2_confirm/run_dissociation_test.py $ARGS \
        --freeze-tag $TAG ${STORE:+--knowledge-set "$STORE"} ${C4S:+--control-set "$C4S"} --allow-score-code-change \
        --out $HELD_OUT/dissociation_new || FAIL=1
fi
if [ -n "$C4S" ]; then
    run control_semantic "$(finished $HELD_OUT/control_semantic)" $PY experiments/e2_confirm/run_control_test.py $ARGS \
        --freeze-tag $TAG --control-set "$C4S" --allow-score-code-change --out $HELD_OUT/control_semantic || FAIL=1
fi

# 4. battery and letter-cued naming on the 160 test concepts
SETS="--set K8=$K8 --set C4=$C4"
[ -n "$C4S" ] && SETS="$SETS --set C4s=$C4S"
[ -n "$STORE" ] && SETS="$SETS --set S8=$STORE"
run new_sets "$(finished outputs/e3_errors/$MODEL/new_sets)" $PY experiments/e3_errors/run_new_sets.py $ARGS $SETS || FAIL=1
echo "$(stamp) done (any step failed: $FAIL)"
exit $FAIL
