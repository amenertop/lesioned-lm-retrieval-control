#!/usr/bin/env bash
# Rehearsal of experiments/e3_errors/gpu_session10.sh's stop and lock with stand-in workers (agent3 reviews 2026-10-13,
# N10, F6, F8). Linux only (setsid, flock, pgrep); run on the server before the real session, from the repo root:
#     bash tests/shell/gpu_session10_stop.sh
# In a throwaway clone tagged at HEAD, the session script runs with a stand-in python that sleeps on the steps whose
# command line holds S10_FAKE_BLOCK and succeeds at once on every other step. Three stops:
# 1. step 1, its two GPU workers in the background: both start, and a second session is refused while they run;
# 2. the merge, a foreground step after step 1;
# 3. the implant, a foreground step of part 3.
# Each time the script must end within 5 seconds of kill -TERM, report none left, and leave no worker. Only the
# process groups this rehearsal starts are counted and cleaned up; no other process on the machine is touched.
# Exit 0 only if every check holds.
set -u
repo=$(git rev-parse --show-toplevel)
work=$(mktemp -d)
groups=""
cleanup() { for g in $groups; do kill -KILL -- "-$g" 2>/dev/null; done; rm -rf "$work"; }
trap cleanup EXIT
git clone -q "$repo" "$work/r" && cd "$work/r" || exit 1
git tag -f analysis-frozen-optogenetics HEAD > /dev/null
cat > "$work/fakepy" <<'EOF'
#!/usr/bin/env bash
case " $* " in *"$S10_FAKE_BLOCK"*) exec sleep 4242 ;; esac
exit 0
EOF
chmod +x "$work/fakepy"
export S10_PY="$work/fakepy"
ok=0
# the stand-in workers of one session: its process group (the script re-execs itself with setsid, keeping its pid)
workers() { pgrep -g "$1" -f "^sleep 4242$" | wc -l; }

scenario() {   # NAME BLOCK WORKERS [lock]
    local name=$1 want=$3 lock=${4:-} pid n code t=0
    export S10_FAKE_BLOCK=$2 S10_LOG="$work/logs_$name"
    bash experiments/e3_errors/gpu_session10.sh > "$work/$name.out" 2>&1 &
    pid=$!
    groups="$groups $pid"
    sleep 3
    n=$(workers "$pid")
    [ "$n" -eq "$want" ] && echo "PASS $name: $n worker(s) running" || { echo "FAIL $name: $n worker(s) running, want $want"; ok=1; }
    if [ -n "$lock" ]; then
        timeout 20 bash experiments/e3_errors/gpu_session10.sh > "$work/second.out" 2>&1
        code=$?
        if [ $code -ne 0 ] && grep -q "still running" "$work/second.out"; then echo "PASS $name: a second session is refused"
        else echo "FAIL $name: second session exit $code"; cat "$work/second.out"; ok=1; fi
    fi
    kill -TERM "$pid"
    while kill -0 "$pid" 2>/dev/null && [ $t -lt 50 ]; do sleep 0.1; t=$((t + 1)); done
    kill -0 "$pid" 2>/dev/null && { echo "FAIL $name: the script still runs 5 s after TERM"; ok=1; }
    wait "$pid"
    code=$?
    sleep 1
    n=$(workers "$pid")
    [ "$n" -eq 0 ] && echo "PASS $name: stopped (exit $code), no worker left" || { echo "FAIL $name: $n worker(s) left"; ok=1; }
    grep -q "STOPPED. Processes still alive: none" "$work/$name.out" && echo "PASS $name: the script reports none left" \
        || { echo "FAIL $name: the script's report"; ok=1; }
    echo "--- $name: session output"; cat "$work/$name.out"
}

scenario step1 "run_optogenetics.py" 2 lock
scenario merge "merge_optogenetics_shards.py" 1
scenario implant "--only implant" 1
exit $ok
