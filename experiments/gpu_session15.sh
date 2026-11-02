#!/usr/bin/env bash
# GPU session 15 (docs/plans/p14-learned-gates.md), exploratory, main model, on the cards in CARDS (default: all):
#   step 0: the candidate pools (build_pool.py, no GPU)
#   step 1: a smoke test of the backward pass on the first card (3 steps): peak memory, seconds per step, gradient
#           checks; if it runs out of memory, training uses gradient checkpointing
#   step 2: gate training and each run's curve (train_gates.py), one worker per card with one model load, all workers
#           on one list, each run claimed by the first free worker (--claim-dir): knowledge (naming route) and hub
#           objectives on the 512 pool, control on its 512 pool, two sparsity weights, three seeds each; the knowledge
#           objective also on the 128 pool (does the set change when the cut grows?). LATE_CARD and LATE_AFTER_PID:
#           a card busy with other work joins once that process has ended, if runs are still unclaimed
#   step 3: the sets and their minimality (confirm.py, first card)
#   step 4: the sets on the sealed items against 100 matched random sets (run_dissociation_test.py): the knowledge set
#           and the hub set, each with the control set, on two cards when there are two (--allow-score-code-change: the
#           P12 edits to scoring.py leave the Qwen prompts and label ids, all this scorer uses from it, unchanged)
# Run on the server from the repo root of its own worktree, HEAD at the tag below, clean tree:
#     nohup bash experiments/gpu_session15.sh >> /root/autodl-tmp/logs/s15.log 2>&1 &
# TINY=1 rehearses the chain locally on the tiny random models (the tag may be a temporary one).
set -uo pipefail
source experiments/session_lib.sh
TAG=${S15_TAG:-analysis-frozen-session15}
Q=qwen3-30b-a3b-instruct
if [ -n "${TINY:-}" ]; then
    PY=${PY:-/c/Users/wanghe/.conda/envs/lesion/python.exe}; SUF=_tiny; T="--tiny"
    QMEANS=outputs/means_tiny/$Q.pt; LOG=${LOGROOT:-outputs/logs_tiny/s15}; QDT=bfloat16; CARDS=${CARDS:-0}
    POOLARGS="--sizes knowledge=4,8;control=8 --sentinels 3"; KP=knowledge_p8; KS=knowledge_p4; CP=control_p8
    SMALL="--steps 3 --limit 20 --n-se 50"; CSMALL="--limit 20"; NRAND=2
else
    PY=/root/miniconda3/envs/lesion/bin/python; SUF=""; T=""
    QMEANS=outputs/means/$Q@0d7cf23__42ae809__fp16.pt; LOG=/root/autodl-tmp/logs/s15; QDT=float16
    CARDS=${CARDS:-$(nvidia-smi --query-gpu=index --format=csv,noheader | tr '\n' ' ')}
    POOLARGS=""; KP=knowledge_p512; KS=knowledge_p128; CP=control_p512
    SMALL=""; CSMALL=""; NRAND=100
fi
QARGS="--model $Q --means $QMEANS --dtype $QDT --scorer prefix_avg3 --batch-size 16 $T"
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOG"
require_tag "$TAG"
P14=outputs/p14$SUF/$Q
read -r -a CARD_LIST <<< "$CARDS"
FIRST=${CARD_LIST[0]}
ALL_CARDS=("${CARD_LIST[@]}" ${LATE_CARD:+"$LATE_CARD"})
echo "$(stamp) session 15 at $(git rev-parse --short HEAD) on card(s) ${CARD_LIST[*]}${LATE_CARD:+, card $LATE_CARD after pid $LATE_AFTER_PID}"
FAIL=0

# always rebuilt: the content depends only on the inputs, and build_pool.py stops if an existing file differs
run pools "" $PY experiments/p14/build_pool.py --model $Q --means $QMEANS $T $POOLARGS || exit 1

# the backward pass on the real model, without and with gradient checkpointing (agent2's code review): peak memory,
# seconds per step, finite gradients, and the routed-token count, which must not double with checkpointing
CK=""
HEAD=$(git rev-parse HEAD)
for mode in off on; do
    # an earlier smoke record is reused only if it was made at this very commit (agent2's recheck)
    same=$($PY -c "import json,sys; m=json.load(open(sys.argv[1],encoding='utf-8')); print('yes' if m.get('commit')==sys.argv[2] and m.get('checkpointing')==sys.argv[3] else 'no')" \
        $P14/smoke_$mode.json $HEAD $mode 2>/dev/null)
    run smoke_$mode "$( [ "$same" = yes ] && echo $P14/smoke_$mode.json )" env CUDA_VISIBLE_DEVICES=$FIRST \
        $PY experiments/p14/train_gates.py $QARGS --runs $KP:0.01:0 --smoke 3 --checkpointing $mode || exit 1
done
$PY - $P14/smoke_off.json $P14/smoke_on.json <<'PY' || exit 1
import json, sys
off, on = (json.load(open(f, encoding="utf-8")) for f in sys.argv[1:])
print(f"smoke off: {off}\nsmoke on: {on}")
if on["oom"] or on["finite_steps"] < on["steps"] - 1:   # the loss scale may still settle in the first step
    sys.exit("even with gradient checkpointing the backward pass fails")
if not off["oom"] and on["routed_total"] > 1.5 * off["routed_total"]:
    sys.exit("routed-token count nearly doubles with checkpointing (recomputation counted)")
PY
if grep -q '"oom": true' $P14/smoke_off.json; then
    CK="--checkpointing on"; echo "$(stamp) smoke: out of memory without checkpointing; training with checkpointing"
fi

# the runs, interleaved so that every card gets a mix of objectives
RUNS=()
for seed in 0 1 2; do
    for lam in 0.01 0.1; do
        RUNS+=("$KP:$lam:$seed" "$KP:$lam:$seed:hub" "$CP:$lam:$seed")
    done
    RUNS+=("$KS:0.01:$seed")
done
[ -n "${TINY:-}" ] && RUNS=("$KP:0.01:0" "$KP:0.01:0:hub" "$CP:0.01:0" "$KS:0.01:0" "$KP:0.1:1")
LIST=$(IFS=,; echo "${RUNS[*]}")
CLAIMS=$P14/claims
rm -rf "$CLAIMS"; mkdir -p "$CLAIMS"       # no worker of an earlier attempt is alive; finished runs are skipped anyway
worker() {
    echo "$(stamp) train card $1: start"
    CUDA_VISIBLE_DEVICES=$1 $PY experiments/p14/train_gates.py $QARGS --runs "$LIST" --claim-dir "$CLAIMS" $CK $SMALL \
        >> "$LOG/train_card$1.log" 2>&1
    local code=$?
    echo "$(stamp) train card $1: exit $code"
    return $code
}
PIDS=()
for g in "${CARD_LIST[@]}"; do
    worker $g &
    PIDS+=($!)
done
if [ -n "${LATE_CARD:-}" ]; then
    (
        while kill -0 "$LATE_AFTER_PID" 2>/dev/null; do sleep 60; done
        if [ "$(ls "$CLAIMS" | wc -l)" -lt "${#RUNS[@]}" ]; then worker $LATE_CARD
        else echo "$(stamp) train card $LATE_CARD: nothing left to claim"; fi
    ) &
    PIDS+=($!)
fi
for p in "${PIDS[@]}"; do wait $p || FAIL=1; done
# count only this session's runs (an extra old run in the folder must not change the count)
N_DONE=$($PY - $P14 "${RUNS[@]}" <<'PY'
import json, sys
from pathlib import Path
sys.path.insert(0, "experiments/p14")
from train_gates import parse_runs, run_name
root = Path(sys.argv[1])
names = [run_name(*r) for r in parse_runs(",".join(sys.argv[2:]))]
print(sum((root / "runs" / n / "meta.json").exists() and json.loads((root / "runs" / n / "meta.json").read_text(encoding="utf-8")).get("status") == "done" for n in names))
PY
)
echo "$(stamp) training: $N_DONE of ${#RUNS[@]} runs done"
if [ "$FAIL" -ne 0 ] || [ "$N_DONE" -ne "${#RUNS[@]}" ]; then
    echo "$(stamp) session 15 stopped: training incomplete (see $LOG/train_card*.log)"; exit 1
fi

# confirm.py itself skips a finished confirmation only when its runs, sets and settings are the same
run confirm "" env CUDA_VISIBLE_DEVICES=$FIRST \
    $PY experiments/p14/confirm.py $QARGS --freeze-tag $TAG $CSMALL || { echo "$(stamp) session 15 stopped (confirm failed)"; exit 1; }

set_of() { $PY -c "import json,sys; g=json.load(open(sys.argv[1], encoding='utf-8')); print(','.join(g[sys.argv[2]]['members']) if sys.argv[2] in g else '')" $P14/confirm/groups.json $1; }
CSET=$(set_of control)
FPIDS=()
card=0
for objective in knowledge hub; do
    KSET=$(set_of $objective)
    if [ -z "$KSET" ] || [ -z "$CSET" ]; then echo "$(stamp) final_$objective: no set, skipped"; FAIL=1; continue; fi
    # a finished final test is reused only if it tested these very sets; otherwise stop rather than mix results
    state=$($PY -c "import json,sys,os; f=os.path.join(sys.argv[1],'meta.json'); m=json.load(open(f,encoding='utf-8')) if os.path.exists(f) else None
print('none' if m is None else 'same' if m.get('status')=='done' and m['params']['knowledge_sets']['main']==sys.argv[2].split(',') and m['params']['control']==sys.argv[3].split(',') else 'different')" \
        $P14/final_$objective "$KSET" "$CSET")
    if [ "$state" != none ] && [ "$state" != same ]; then echo "$(stamp) final_$objective: $P14/final_$objective holds another set (or an unreadable record: $state); move it away"; FAIL=1; continue; fi
    FINAL=(run final_$objective "$( [ "$state" = same ] && echo $P14/final_$objective/meta.json )" env CUDA_VISIBLE_DEVICES=${ALL_CARDS[$(( card % ${#ALL_CARDS[@]} ))]}
           $PY experiments/e2_confirm/run_dissociation_test.py $QARGS --freeze-tag $TAG --n-random $NRAND
           --knowledge-set "$KSET" --control-set "$CSET" --knowledge-dir outputs/e3_knowledge$SUF/$Q/screen
           --allow-score-code-change --out $P14/final_$objective)
    if [ "${#ALL_CARDS[@]}" -eq 1 ]; then
        "${FINAL[@]}" || FAIL=1
    else
        ( "${FINAL[@]}" ) &
        FPIDS+=($!)
    fi
    card=$((card + 1))
done
for p in "${FPIDS[@]+"${FPIDS[@]}"}"; do wait $p || FAIL=1; done

echo "$(stamp) session 15 done (any step failed: $FAIL)"
exit $FAIL
