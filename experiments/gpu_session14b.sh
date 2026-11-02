#!/usr/bin/env bash
# GPU session 14b (user, 2026-10-15: chain more work onto the card after session 14), exploratory, main model, one card:
#   step 1: K8 and K8b-2 lesioned together (run_joint_sets.py): do the two knowledge sets add up or overlap?
#   step 2: C4 in the three label parts, "A:", "B:" and "Answer:" (run_control_positions.py --split-labels)
#   step 3: the damage profile of every singly screened component (run_profile_pool.py), for P14's taxonomy
# Starts after session 14 has ended (it shares the card); run on the server from the repo root of its own worktree,
# HEAD at the tag below, clean tree:
#     nohup bash experiments/gpu_session14b.sh >> /root/autodl-tmp/logs/s14b.log 2>&1 &
# TINY=1 rehearses the chain locally on the tiny random models (the tag may be a temporary one).
set -uo pipefail
source experiments/session_lib.sh
TAG=${S14B_TAG:-analysis-frozen-session14b}
Q=qwen3-30b-a3b-instruct
if [ -n "${TINY:-}" ]; then
    PY=${PY:-/c/Users/wanghe/.conda/envs/lesion/python.exe}; SUF=_tiny; T="--tiny"
    QMEANS=outputs/means_tiny/$Q.pt; LOG=${LOGROOT:-outputs/logs_tiny/s14b}; QDT=bfloat16
    SMALL_JOINT="--n-random 2"; SMALL_LABELS="--n-random 2"; SMALL_POOL="--limit 2"
else
    PY=/root/miniconda3/envs/lesion/bin/python; SUF=""; T=""
    QMEANS=outputs/means/$Q@0d7cf23__42ae809__fp16.pt; LOG=/root/autodl-tmp/logs/s14b; QDT=float16
    SMALL_JOINT=""; SMALL_LABELS=""; SMALL_POOL=""
fi
QARGS="--model $Q --means $QMEANS --dtype $QDT --scorer prefix_avg3 --batch-size 16 $T"
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
mkdir -p "$LOG"
require_tag "$TAG"
echo "$(stamp) session 14b at $(git rev-parse --short HEAD)"
FAIL=0

run joint "$(finished outputs/e3_knowledge$SUF/$Q/joint_sets)" $PY experiments/e3_knowledge/run_joint_sets.py $QARGS \
    --freeze-tag $TAG $SMALL_JOINT || FAIL=1
run labels "$(finished outputs/e2_confirm$SUF/$Q/control_positions_labels)" \
    $PY experiments/e2_confirm/run_control_positions.py $QARGS --freeze-tag $TAG --split-labels \
    --parts label_a,label_b,label_answer $SMALL_LABELS || FAIL=1
run profile_pool "$(finished outputs/e3_knowledge$SUF/$Q/profile_pool)" \
    $PY experiments/e3_knowledge/run_profile_pool.py $QARGS $SMALL_POOL || FAIL=1

echo "$(stamp) session 14b done (any step failed: $FAIL)"
exit $FAIL
