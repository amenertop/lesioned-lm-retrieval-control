#!/usr/bin/env bash
# GPU session 10 on a two-GPU instance (user decisions P6, P7, P8 and two GPUs, 2026-10-12; revised after the reviews
# by agent2 (2026-10-12, -13) and agent3 (2026-10-13)). Run on the server from the repo root, HEAD at the analysis-plan
# freeze tag, clean tree:
#     nohup bash experiments/e3_errors/gpu_session10.sh >> /root/autodl-tmp/logs/s10.log 2>&1 &
# 1. In parallel: GPU 0 runs the first half of the 45 confirmatory states (and the router counts); GPU 1 runs the
#    intact state once more (cross-GPU check), then the second half.
# 2. Only if both halves finished: merge (which also runs the cross-GPU check), then the confirmatory analysis, which
#    runs every time (no GPU; it checks the run again and rewrites the report atomically, never trusting an old
#    report). Any failure here stops the script with exit code 1. A failed cross-GPU check never becomes
#    optogenetics_confirm (it is kept as optogenetics_confirm.crossgpu_failed-<time>); after finding the cause, delete
#    optogenetics_confirm.intactcheck1of2 and rerun this script: GPU 1 reruns the intact state and the merge is redone.
#    A state whose tables fail the plan's check stops its GPU at once (the tables are kept under
#    optogenetics_confirm.shard<i>of2.partial/invalid/); rerunning gives the same numbers, so find the cause first.
# 3. GPU 0 alone, after every Qwen process has exited: the implant rerun (O3, calibration concepts), then the
#    second model's measurement check (P6). Failures here are reported, and make the exit code 1, but do not touch
#    the confirmatory results.
# Rerunning the script after a stop resumes: finished outputs are skipped, unfinished runs resume their units.
# One session at a time: a lock (held by the script and every worker it starts) refuses a second session while any
# of them still runs. The script runs in its own process group and waits for every step in the background, so
# stopping it (kill -TERM <script pid>, or Ctrl-C) at any step stops the GPU workers too, at once, and lists any
# that remain (agent3 2026-10-13, N10 and F6). Rehearsal: tests/shell/gpu_session10_stop.sh.
# Logs go outside the repo so the tree stays clean.
set -uo pipefail
PY=${S10_PY:-/root/miniconda3/envs/lesion/bin/python}     # S10_PY, S10_LOG: only for tests/shell/gpu_session10_stop.sh
TAG=analysis-frozen-optogenetics
MEANS=outputs/means/qwen3-30b-a3b-instruct@0d7cf23__42ae809__fp16.pt
COMMON="--model qwen3-30b-a3b-instruct --means $MEANS --dtype float16"
OUT=outputs/e3_errors/qwen3-30b-a3b-instruct
LOG=${S10_LOG:-/root/autodl-tmp/logs}
mkdir -p "$LOG"
stamp() { date "+%F %T"; }

# own process group, so a stop reaches the workers (agent3 2026-10-13, N10)
if [ "$(ps -o pgid= -p $$ | tr -d ' ')" != "$$" ]; then exec setsid bash "$0" "$@"; fi
exec 9> "$LOG/s10.lock"
flock -n 9 || { echo "$(stamp) another session 10 (or a worker of one) is still running; see $LOG"; exit 1; }
# the processes of this session's group other than the script, the subshell running this function and the
# listing itself, without zombies (agent3 F8; the subshell's pid is taken on entry, not inside the inner $(...), G5)
others() {
    local self=$BASHPID
    pgrep -g $$ > "$LOG/s10_group.txt"
    local pids
    pids=$(grep -vx -e "$$" -e "$self" "$LOG/s10_group.txt" | paste -sd, -)
    [ -n "$pids" ] && ps -o pid=,stat=,args= -p "$pids" | awk '$2 !~ /^Z/'
    return 0
}
stop() {
    trap '' INT TERM
    echo "$(stamp) STOPPING: signal received, stopping the workers of this session"
    kill -TERM -- -$$ 2>/dev/null
    wait
    local left="" i
    for i in 1 2 3 4 5; do left=$(others); [ -z "$left" ] && break; sleep 1; done
    echo "$(stamp) STOPPED. Processes still alive: ${left:-none}. Logs: $LOG/s10_*.log. Rerun this script to resume."
    exit 130
}
trap stop INT TERM

# run NAME DIR CMD...: skip when DIR (the step's finished output) is given and exists; otherwise run, log, report.
# The command runs in the background and the script waits for it: bash runs a trap only between commands, and
# `wait` is the one place a signal interrupts at once (agent3 F6)
run() {
    local name=$1 dir=$2; shift 2
    if [ -n "$dir" ] && [ -e "$dir" ]; then echo "$(stamp) $name: already done ($dir)"; return 0; fi
    "$@" >> "$LOG/s10_$name.log" 2>&1 &
    wait $!
    local code=$?
    echo "$(stamp) $name: exit $code"
    return $code
}

[ "$(git rev-parse HEAD)" = "$(git rev-parse "$TAG^{commit}")" ] || { echo "HEAD is not $TAG"; exit 1; }
[ -z "$(git status --porcelain --untracked-files=no)" ] && [ -z "$(git status --porcelain -- data/materials data/neutral_text src experiments docs)" ] \
    || { echo "tree not clean"; exit 1; }
nvidia-smi -L
echo "$(stamp) start"

STATES="$PY experiments/e3_errors/run_optogenetics.py $COMMON --role test --freeze-tag $TAG --out-name optogenetics_confirm"
run shard0 "$OUT/optogenetics_confirm.shard0of2" env CUDA_VISIBLE_DEVICES=0 $STATES --shard 0/2 &
P0=$!
( run intactcheck1 "$OUT/optogenetics_confirm.intactcheck1of2" env CUDA_VISIBLE_DEVICES=1 $STATES --only intact-check --shard 1/2 \
  && run shard1 "$OUT/optogenetics_confirm.shard1of2" env CUDA_VISIBLE_DEVICES=1 $STATES --shard 1/2 ) &
P1=$!
wait $P0; C0=$?
wait $P1; C1=$?
if [ $C0 -ne 0 ] || [ $C1 -ne 0 ]; then
    echo "$(stamp) STOPPED: a half did not finish (GPU 0 exit $C0, GPU 1 exit $C1; see $LOG/s10_shard0.log, s10_intactcheck1.log, s10_shard1.log); rerun this script to resume"
    exit 1
fi

run merge "$OUT/optogenetics_confirm" $PY experiments/e3_errors/merge_optogenetics_shards.py --model qwen3-30b-a3b-instruct \
    --out-name optogenetics_confirm --shards 2     || { echo "$(stamp) STOPPED: merge or cross-GPU check failed (see $LOG/s10_merge.log and any $OUT/optogenetics_confirm.crossgpu_failed-*)"; exit 1; }
# always rerun: a report on disk is never taken as proof that the run passed the confirmatory checks (agent3 N6)
run confirm_analysis "" $PY experiments/e3_errors/analyze_optogenetics.py --confirm \
    || { echo "$(stamp) STOPPED: confirmatory analysis failed (see $LOG/s10_confirm_analysis.log)"; exit 1; }

FAIL=0
run implant "$OUT/optogenetics_implant_v2" env CUDA_VISIBLE_DEVICES=0 $PY experiments/e3_errors/run_optogenetics.py $COMMON \
    --only implant --out-name optogenetics_implant_v2 \
  && run implant_analysis "reports/explore/qwen_optogenetics_implant_v2.json" $PY experiments/e3_errors/analyze_optogenetics.py \
    --dir "$OUT/optogenetics_implant_v2" --out reports/explore/qwen_optogenetics_implant_v2.json \
  || FAIL=1
run phi_g1 "outputs/e0_p6_phi/phi35-moe-instruct/meta.json" env CUDA_VISIBLE_DEVICES=0 $PY experiments/e0_baseline/run.py \
    --versions phi35-moe-instruct --exploratory --out outputs/e0_p6_phi \
  && run phi_g1_analysis "reports/g1_phi/g1_result.json" $PY experiments/e0_revised/analyze.py --outputs outputs/e0_p6_phi \
    --report-dir reports/g1_phi \
  || FAIL=1
echo "$(stamp) done (confirmatory part complete; implant or second-model failure: $FAIL)"
exit $FAIL
