#!/usr/bin/env bash
# GPU session 14 (docs/plans/p13-knowledge-breadth.md, docs/plans/second-model-g1-p12.md), exploratory, one card:
#   download: the P12 candidates' weights to the network disk, in the background while the main model computes
#             (the run is billed anyway; the main model's weights are already on the GPU), verified file by file
#   step 1: where a forward pass spends its time (profile_forward.py, about 5 minutes after loading)
#   step 2: P13, knowledge lesions of growing breadth on the main model (run_breadth.py, about 2 to 2.5 hours)
#   step 3: P12, gpt-oss-20b: format check -> G1 scoring (only if the format check passed) -> expert norms
#   step 4: P12, DeepSeek-V2-Lite-Chat: format check -> G1 scoring (only if the format check passed)
# The two P12 models run one after the other (the container's memory peak while loading gpt-oss is unknown).
# Run on the server from the repo root, HEAD at the tag below, clean tree:
#     nohup bash experiments/gpu_session14.sh >> /root/autodl-tmp/logs/s14.log 2>&1 &
# TINY=1 rehearses the whole chain locally on the tiny random models (the tag may be a temporary one).
set -uo pipefail
source experiments/session_lib.sh
TAG=${S14_TAG:-analysis-frozen-session14}
Q=qwen3-30b-a3b-instruct
if [ -n "${TINY:-}" ]; then
    PY=${PY:-/c/Users/wanghe/.conda/envs/lesion/python.exe}; SUF=_tiny; T="--tiny"
    QMEANS=outputs/means_tiny/$Q.pt; LOG=${LOGROOT:-outputs/logs_tiny/s14}; QDT=bfloat16
    BREADTH_SMALL="--anchor 3 --sizes 4 --n-random 1"; E0=outputs/e0_explore_tiny
else
    PY=/root/miniconda3/envs/lesion/bin/python; SUF=""; T=""
    QMEANS=outputs/means/$Q@0d7cf23__42ae809__fp16.pt; LOG=/root/autodl-tmp/logs/s14; QDT=float16
    BREADTH_SMALL=""; E0=outputs/e0_explore
fi
QARGS="--model $Q --means $QMEANS --dtype $QDT --scorer prefix_avg3 --batch-size 16 $T"
export HF_HOME=/root/autodl-tmp/hf PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True CUDA_VISIBLE_DEVICES=0
mkdir -p "$LOG"
require_tag "$TAG"
echo "$(stamp) session 14 at $(git rev-parse --short HEAD)"
FAIL=0

DL=""
if [ -z "${TINY:-}" ]; then
    echo "$(stamp) download: start (background)"
    # hf-mirror is reached directly (the AutoDL proxy slowed earlier downloads; dl_ms_p6.py also dropped it)
    # at low CPU and disk priority, so that the main model's run comes first (user, 2026-10-15: download in the
    # background of the run rather than in no-GPU mode)
    ( HF_HUB_OFFLINE=0 HF_ENDPOINT=https://hf-mirror.com nice -n 15 ionice -c 3 \
          $PY tools/dl_weights_p12.py gpt-oss-20b deepseek-v2-lite-chat \
          > "$LOG/download.log" 2>&1; echo "$(stamp) download: exit $?" ) &
    DL=$!
fi
export HF_HUB_OFFLINE=1

run profile "$(grep -l '"status": "done"' outputs/profile$SUF/$Q.json 2>/dev/null)" \
    $PY experiments/profile_forward.py $QARGS --out outputs/profile$SUF/$Q.json || FAIL=1
run breadth "$(finished outputs/e3_errors$SUF/$Q/breadth)" $PY experiments/e3_errors/run_breadth.py $QARGS \
    --freeze-tag $TAG $BREADTH_SMALL || FAIL=1

if [ -n "$DL" ]; then
    echo "$(stamp) waiting for the download"; wait $DL
fi
for name in gpt-oss-20b deepseek-v2-lite-chat; do
    if [ -z "${TINY:-}" ] && [ ! -e /autodl-fs/data/main/ms/$name/VERIFIED.json ]; then
        echo "$(stamp) $name: weights not verified, skipped"; FAIL=1; continue
    fi
    run ${name}_format "" $PY experiments/e0_baseline/run.py --versions $name --exploratory --format-check \
        --out $E0 $T || { FAIL=1; continue; }
    status=$($PY -c "import json,sys; print(json.load(open(sys.argv[1], encoding='utf-8'))['status'])" $E0/$name/format_check.json 2>/dev/null)
    echo "$(stamp) $name: format check $status"
    if [ "$status" = ok ] || [ -n "${TINY:-}" ]; then
        run ${name}_g1 "$( [ -e $E0/$name/label_main.csv ] && echo $E0/$name/label_main.csv )" \
            $PY experiments/e0_baseline/run.py --versions $name --exploratory --out $E0 $T || FAIL=1
    fi
    if [ $name = gpt-oss-20b ]; then
        run gptoss_norms "$( [ -e $E0/$name/expert_norms/summary.json ] && echo $E0/$name/expert_norms/summary.json )" \
            $PY experiments/second_model/run_gptoss_norms.py --out $E0/$name/expert_norms $T || FAIL=1
    fi
done
echo "$(stamp) session 14 done (any step failed: $FAIL)"
exit $FAIL
