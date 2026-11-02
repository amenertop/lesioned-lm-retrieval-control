#!/usr/bin/env bash
# GPU session 13 (docs/plans/gpu-session13.md; user decision P11 A, 2026-10-15), exploratory, one card, main model only:
#   step 1: K8b taken apart (run_k8b_split.py): language-model loss, held-out M1, the word-length check, the target's
#           place in the whole vocabulary's change
#   step 2: held-out dissociation tests, 100 random sets each: K8b without the pair it shares with S8 (with C4), the pair
#           alone (with C4), S8 without the pair (with C4s, as S8 was tested in session 11)
#   step 3: the control heads lesioned in one part of the prompt at a time (run_control_positions.py), 50 random sets
# Run on the server from the repo root, HEAD at the tag below, clean tree:
#     nohup bash experiments/gpu_session13.sh >> /root/autodl-tmp/logs/s13.log 2>&1 &
# TINY=1 rehearses the whole chain locally on the tiny random models (the tag may be a temporary one).
set -uo pipefail
source experiments/session_lib.sh
TAG=${S13_TAG:-analysis-frozen-session13}
Q=qwen3-30b-a3b-instruct
if [ -n "${TINY:-}" ]; then
    PY=${PY:-/c/Users/wanghe/.conda/envs/lesion/python.exe}; SUF=_tiny; T="--tiny"
    QMEANS=outputs/means_tiny/$Q.pt; LOG=${LOGROOT:-outputs/logs_tiny/s13}; QDT=bfloat16
    HELD_SMALL="--n-random 3 --n-boot 50"; SPLIT_SMALL="--n-random 3 --length-items 20"; POS_SMALL="--n-random 3"
else
    PY=/root/miniconda3/envs/lesion/bin/python; SUF=""; T=""
    QMEANS=outputs/means/$Q@0d7cf23__42ae809__fp16.pt; LOG=/root/autodl-tmp/logs/s13; QDT=float16
    HELD_SMALL="--n-random 100"; SPLIT_SMALL=""; POS_SMALL=""
fi
QARGS="--model $Q --means $QMEANS --dtype $QDT --scorer prefix_avg3 --batch-size 16 $T"
HELD=outputs/e2_confirm$SUF
SPLIT=outputs/e3_knowledge$SUF/$Q/k8b_split
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
mkdir -p "$LOG"
require_tag "$TAG"
csv() { $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(','.join(d[sys.argv[2]]))" "$@" 2>/dev/null; }
# every input must exist with a non-empty set, so that no step falls back to a default set (an empty --control-set
# would make run_dissociation_test.py use C4)
need() { local v; v=$(csv "$1" "$2"); [ -n "$v" ] || { echo "$(stamp) missing or empty: $1 [$2]" >&2; return 1; }; echo "$v"; }
C4=$(need outputs/e1_screen$SUF/$Q/screen/selection.json control) || exit 1
C4S=$(need $HELD/$Q/dissociation_new/sets.json control) || exit 1
need $HELD/$Q/dissociation_new/sets.json main > /dev/null || exit 1
need $HELD/$Q/dissociation/sets.json main > /dev/null || exit 1
need outputs/e3_knowledge$SUF/$Q/k8b/selection.json k8b > /dev/null || exit 1
# the scoring code, the naming code, the hooks and the materials are those of session 12
git diff --quiet analysis-frozen-session12c HEAD -- src/lesion/scoring.py src/lesion/prefix_scoring.py \
    src/lesion/naming.py src/lesion/hooks.py data/materials || { echo "$(stamp) scoring code changed since session 12"; exit 1; }
# the tiny K8b and S8 share nothing, so the rehearsal takes K8b's first component as the "pair"
PAIR_ARG=""; [ -n "${TINY:-}" ] && PAIR_ARG="--pair $(csv outputs/e3_knowledge$SUF/$Q/k8b/selection.json k8b | cut -d, -f1)"
echo "$(stamp) session 13 at $(git rev-parse --short HEAD); C4=$C4 C4s=$C4S"
FAIL=0; STEP1=0

run k8b_split "$(finished $SPLIT)" $PY experiments/e3_knowledge/run_k8b_split.py $QARGS --freeze-tag $TAG \
    $SPLIT_SMALL $PAIR_ARG || { FAIL=1; STEP1=1; }
K8BM=$(csv $SPLIT/sets.json K8b_minus_shared); PAIR=$(csv $SPLIT/sets.json shared_pair); S8M=$(csv $SPLIT/sets.json S8_minus_pair)
echo "$(stamp) K8b without the pair: ${K8BM:-none}; pair: ${PAIR:-none}; S8 without the pair: ${S8M:-none}"
# name|knowledge set|control set (component keys contain colons, so the fields are split on |)
for spec in "dissociation_k8b_minus_shared|$K8BM|$C4" "dissociation_shared_pair|$PAIR|$C4" "dissociation_s8_minus_pair|$S8M|$C4S"; do
    IFS='|' read -r name kset cset <<< "$spec"
    if [ $STEP1 -ne 0 ] || [ -z "$kset" ] || [ -z "$cset" ]; then
        echo "$(stamp) $name: skipped (step 1 failed or a set is empty)"; FAIL=1; continue
    fi
    run $name "$(finished $HELD/$Q/$name)" $PY experiments/e2_confirm/run_dissociation_test.py $QARGS $HELD_SMALL \
        --freeze-tag $TAG --knowledge-set "$kset" --control-set "$cset" \
        --knowledge-dir outputs/e3_knowledge$SUF/$Q/screen --allow-score-code-change --out $HELD/$Q/$name || FAIL=1
done
# step 3 does not use step 1's outputs, so it runs even when step 1 failed
run control_positions "$(finished $HELD/$Q/control_positions)" $PY experiments/e2_confirm/run_control_positions.py \
    $QARGS --freeze-tag $TAG $POS_SMALL || FAIL=1
echo "$(stamp) session 13 done (any step failed: $FAIL)"
exit $FAIL
