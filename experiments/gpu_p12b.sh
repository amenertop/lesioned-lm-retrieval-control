#!/usr/bin/env bash
# GPU P12b (docs/plans/second-model-g1-p12.md, section P12b; user decision 2026-10-16), exploratory, one card, about
# 30 minutes: gpt-oss-20b again, now with the "harmony_answer" prompt format (the scoring prompt also opens the final
# message with "Answer:", the labels are " A" and " B"; lesion.models.PROMPT_FORMATS):
#   step 1: format check (the same demo items and 90% gate as in session 14) -> G1 scoring, only if it passed -> the
#           revised G1 analysis (experiments/e0_revised/analyze.py, as for the other models)
#   step 2: expert output norms, with the per-token check (lesion.probes.outputs_agree)
# Results go to outputs/e0_explore_p12b, which must not exist yet, so that session 14's records in outputs/e0_explore
# stay as they are; P12B_NOTE.json there marks them as a retry chosen after seeing the format check.
# CARD (default 0) picks the card. Run on the server from the repo root of its own worktree, HEAD at the tag below,
# clean tree:
#     nohup bash experiments/gpu_p12b.sh >> /root/autodl-tmp/logs/p12b.log 2>&1 &
# TINY=1 rehearses it locally on the tiny random model (the tag may be a commit).
set -uo pipefail
source experiments/session_lib.sh
TAG=${P12B_TAG:-analysis-frozen-p12b}
NAME=gpt-oss-20b
if [ -n "${TINY:-}" ]; then
    PY=${PY:-/c/Users/wanghe/.conda/envs/lesion/python.exe}; T="--tiny"; LOG=${LOGROOT:-outputs/logs_tiny/p12b}
    E0=outputs/e0_explore_p12b_tiny
else
    PY=/root/miniconda3/envs/lesion/bin/python; T=""; LOG=/root/autodl-tmp/logs/p12b; E0=outputs/e0_explore_p12b
fi
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=${CARD:-0}
mkdir -p "$LOG"
require_tag "$TAG"
echo "$(stamp) P12b at $(git rev-parse --short HEAD) on card $CUDA_VISIBLE_DEVICES"
if [ -z "${TINY:-}" ] && [ ! -e /autodl-fs/data/main/ms/$NAME/VERIFIED.json ]; then
    echo "$(stamp) $NAME: weights not verified, stopped"; exit 1
fi
# a fresh folder for every attempt, so that nothing from an earlier one is mixed in (analyze.py reads every model
# folder under $E0)
if [ -e $E0 ]; then echo "$(stamp) $E0 already exists; move it away"; exit 1; fi
mkdir -p $E0
# the evidence status of these results, for whoever writes them up (agent2's review, 2026-10-16)
cat > $E0/P12B_NOTE.json <<NOTE
{"posthoc_format_retry": true, "model": "$NAME", "prompt_format": "harmony_answer", "label_prefix": "space",
 "why": "session 14: 77% of the demo prompts began with a label under the pre-registered harmony format (gate 90%); 10 of the other 11 began 'Answer: A'",
 "decision": "user, 2026-10-16 (docs/plans/decisions-pending.md); plan docs/plans/second-model-g1-p12.md, section P12b",
 "report_separately_from": "outputs/e0_explore (session 14)"}
NOTE
FAIL=0
run format "" $PY experiments/e0_baseline/run.py --versions $NAME --exploratory --format-check --out $E0 $T || exit 1
status=$($PY -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['status'])" $E0/$NAME/format_check.json) \
    || { echo "$(stamp) $NAME: format check record unreadable, stopped"; exit 1; }
case "$status" in ok|format_failure) ;; *) echo "$(stamp) $NAME: unknown format check status '$status', stopped"; exit 1 ;; esac
echo "$(stamp) $NAME: format check $status"
if [ "$status" = ok ] || [ -n "${TINY:-}" ]; then
    run g1 "" $PY experiments/e0_baseline/run.py --versions $NAME --exploratory --out $E0 $T || FAIL=1
    # the analysis needs a real model's scores (a tiny random model's are noise)
    if [ $FAIL = 0 ] && [ -z "${TINY:-}" ]; then
        run g1_analysis "" $PY experiments/e0_revised/analyze.py --outputs $E0 --report-dir $E0/report || FAIL=1
    fi
fi
run norms "" $PY experiments/second_model/run_gptoss_norms.py --out $E0/$NAME/expert_norms $T || FAIL=1
echo "$(stamp) P12b done (format check: $status; any step failed: $FAIL)"
exit $FAIL
