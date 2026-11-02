#!/usr/bin/env bash
# GPU session 11 (docs/plans/gpu-session11.md), on the two-GPU instance straight after session 10, from a second
# worktree at analysis-frozen-session11 whose outputs/ is the session-10 worktree's (session 10 needs its own
# worktree to stay at its freeze tag until it ends):
#     cd /root/autodl-tmp/main-s11 && nohup setsid bash experiments/gpu_session11.sh >> /root/autodl-tmp/logs/s11.log 2>&1 &
# The longest chain goes first, so neither card waits (user, 2026-10-13: finish everything in one go):
# - part A (second model, Phi-3.5-MoE; the longest) on GPU 1, as soon as session 10 no longer uses that card (its
#   second half finished, or the session ended);
# - part B and C (main model: re-screens, held-out tests, battery and cued naming) on GPU 0, once session 10 has ended.
# Each part resumes on a rerun (finished steps are skipped).
set -uo pipefail
LOG=/root/autodl-tmp/logs
OUT10=outputs/e3_errors/qwen3-30b-a3b-instruct
stamp() { date "+%F %T"; }
gpu_busy() { nvidia-smi -i "$1" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | grep -q .; }
s10_running() { pgrep -f '^bash experiments/e3_errors/gpu_session10.sh' > /dev/null; }
mkdir -p "$LOG"
echo "$(stamp) session 11 waiting; HEAD $(git rev-parse --short HEAD)"

(
    until { [ -e "$OUT10/optogenetics_confirm.shard1of2" ] || ! s10_running; } && ! gpu_busy 1; do sleep 60; done
    echo "$(stamp) GPU 1 free: second-model replication"
    GPU=1 bash experiments/second_model/run_phi_replication.sh >> "$LOG/s11_phi.log" 2>&1
    echo "$(stamp) second-model part: exit $?"
) &
(
    until ! s10_running && ! gpu_busy 0; do sleep 60; done
    echo "$(stamp) session 10 over: main-model re-screens"
    GPU=0 bash experiments/e3_errors/run_qwen_rescreens.sh >> "$LOG/s11_qwen.log" 2>&1
    echo "$(stamp) main-model part: exit $?"
) &
wait
echo "$(stamp) session 11 done"
