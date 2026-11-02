#!/usr/bin/env bash
# GPU session 12 (docs/plans/gpu-session12.md; user decision P10 option C, 2026-10-14), exploratory, two GPUs.
# The main model first, on both GPUs at once (one copy of its weights in the page cache); Phi on GPU 1 once lane B is
# done. Loading Phi pushed the main model out of the page cache last time (the main model sits on the network disk and
# the local disk is too small for a copy), so lane A's later loads may take up to about 30 minutes longer; starting
# Phi early still ends the session about an hour sooner.
#   lane A (GPU 0): items 2 and 4 (run_vocab_frequency.py) -> item 5 re-screen (run_rescreen_k8b.py) -> K8b held out
#   lane B (GPU 1): item 3, the shared components removed (control tests and one dissociation test, 100 random sets)
#   then (GPU 1, once lane B ends): item 1, Phi's knowledge set taken apart (run_phi_split.py) -> item 1b if its condition holds
# Run on the server from the repo root, HEAD at the tag below, clean tree:
#     nohup bash experiments/gpu_session12.sh >> /root/autodl-tmp/logs/s12.log 2>&1 &
# TINY=1 rehearses the whole chain locally on the tiny random models (the tag may be a temporary one).
set -uo pipefail
source experiments/session_lib.sh
TAG=${S12_TAG:-analysis-frozen-session12c}
# GPUS=1 (a one-card instance, as on 2026-10-14): the three lanes run one after another on card 0 (lane A, lane B,
# then Phi), with the same steps; otherwise lane A on card 0, lane B on card 1, then Phi on card 1
GPUS=${GPUS:-2}
if [ "$GPUS" = 1 ]; then GPU_A=0; GPU_B=0; GPU_P=0; else GPU_A=0; GPU_B=1; GPU_P=1; fi
Q=qwen3-30b-a3b-instruct
P=phi35-moe-instruct
if [ -n "${TINY:-}" ]; then
    PY=${PY:-/c/Users/wanghe/.conda/envs/lesion/python.exe}; SUF=_tiny; T="--tiny"
    QMEANS=outputs/means_tiny/$Q.pt; PMEANS=outputs/means_tiny/$P.pt; LOGROOT=${LOGROOT:-outputs/logs_tiny/s12}
    SMALL="--n-random 3 --n-boot 50"; K8B_SMALL="--top 12 --n-null 6 --n-random 3 --n-boot 50"
    QDT=bfloat16; PDT=bfloat16; HELD_RANDOM="--n-random 3 --n-boot 50"; PSCORER=prefix_avg3   # as the tiny screens were run
else
    PY=/root/miniconda3/envs/lesion/bin/python; SUF=""; T=""
    QMEANS=outputs/means/$Q@0d7cf23__42ae809__fp16.pt; PMEANS=outputs/means/$P@4368845__fp16.pt
    LOGROOT=/root/autodl-tmp/logs/s12; SMALL="--n-random 100"; K8B_SMALL=""
    QDT=float16; PDT=float16; HELD_RANDOM=""; PSCORER=prefix_avg3_small   # as the Phi screens of session 11
fi
QARGS="--model $Q --means $QMEANS --dtype $QDT --scorer prefix_avg3 --batch-size 16 $T"
PARGS="--model $P --means $PMEANS --dtype $PDT --scorer $PSCORER --batch-size 16 $T"
HELD=outputs/e2_confirm$SUF
export HF_HOME=/root/autodl-tmp/hf HF_HUB_OFFLINE=1 PYTHONIOENCODING=utf-8 PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$LOGROOT/A" "$LOGROOT/B" "$LOGROOT/phi"
require_tag "$TAG"
csv() { $PY -c "import json,sys; d=json.load(open(sys.argv[1])); print(','.join(d[sys.argv[2]]))" "$@" 2>/dev/null; }
without() { $PY -c "import sys; print(','.join(k for k in sys.argv[1].split(',') if k not in sys.argv[2].split(',')))" "$1" "$2"; }
C4=$(csv outputs/e1_screen$SUF/$Q/screen/selection.json control)
C4S=$(csv $HELD/$Q/dissociation_new/sets.json control)
S8=$(csv $HELD/$Q/dissociation_new/sets.json main)
SHARED_C=head:9:15; SHARED_S=head:14:13
[ -n "${TINY:-}" ] && { SHARED_C=$(echo "$C4" | cut -d, -f1); SHARED_S=$(echo "$S8" | cut -d, -f1); }
echo "$(stamp) session 12 at $(git rev-parse --short HEAD); C4=$C4 C4s=$C4S S8=$S8 shared=$SHARED_C,$SHARED_S"

lane_a() {
    local LOG=$LOGROOT/A FAIL=0
    export CUDA_VISIBLE_DEVICES=$GPU_A
    run mechanisms "$(finished outputs/e3_errors$SUF/$Q/mechanisms)" $PY experiments/e3_errors/run_vocab_frequency.py \
        $QARGS --freeze-tag $TAG ${TINY:+--length-items 30} || FAIL=1
    run k8b "$(finished outputs/e3_knowledge$SUF/$Q/k8b)" $PY experiments/e3_knowledge/run_rescreen_k8b.py $QARGS \
        $K8B_SMALL ${K8B_FORCE:+--force-set $K8B_FORCE} || FAIL=1
    local K8B; K8B=$(csv outputs/e3_knowledge$SUF/$Q/k8b/selection.json k8b)
    echo "$(stamp) K8b: ${K8B:-none}"
    if [ -n "$K8B" ]; then
        run k8b_heldout "$(finished $HELD/$Q/dissociation_k8b)" $PY experiments/e2_confirm/run_dissociation_test.py \
            $QARGS $HELD_RANDOM --freeze-tag $TAG --knowledge-set "$K8B" --control-set "$C4" \
            --knowledge-dir outputs/e3_knowledge$SUF/$Q/screen --allow-score-code-change \
            --out $HELD/$Q/dissociation_k8b || FAIL=1
    fi
    echo "$(stamp) lane A done (any step failed: $FAIL)"
    return $FAIL
}

lane_b() {
    local LOG=$LOGROOT/B FAIL=0
    export CUDA_VISIBLE_DEVICES=$GPU_B
    local C4M C4SM S8M; C4M=$(without "$C4" $SHARED_C); C4SM=$(without "$C4S" $SHARED_C); S8M=$(without "$S8" $SHARED_S)
    for spec in "c4_minus_shared:$C4M" "shared_head_alone:$SHARED_C" "c4s_minus_shared:$C4SM"; do
        local name=${spec%%:*} set=${spec#*:}
        run $name "$(finished $HELD/$Q/$name)" $PY experiments/e2_confirm/run_control_test.py $QARGS $SMALL \
            --freeze-tag $TAG --control-set "$set" --allow-score-code-change --out $HELD/$Q/$name || FAIL=1
    done
    run dissociation_minus_shared "$(finished $HELD/$Q/dissociation_minus_shared)" $PY \
        experiments/e2_confirm/run_dissociation_test.py $QARGS $SMALL --freeze-tag $TAG --knowledge-set "$S8M" \
        --control-set "$C4SM" --knowledge-dir outputs/e3_knowledge$SUF/$Q/screen --allow-score-code-change \
        --out $HELD/$Q/dissociation_minus_shared || FAIL=1
    echo "$(stamp) lane B done (any step failed: $FAIL)"
    return $FAIL
}

phi() {
    local LOG=$LOGROOT/phi FAIL=0
    export CUDA_VISIBLE_DEVICES=$GPU_P
    local K8 SPLIT=outputs/e3_knowledge$SUF/$P/split
    run phi_split "$(finished $SPLIT)" $PY experiments/second_model/run_phi_split.py $PARGS \
        ${TINY:+--n-boot 50 --k8 $PHI_K8 --super $PHI_SUPER} || FAIL=1
    # item 1b: K8 without the two candidate super experts, held out, if it no longer breaks language and keeps a
    # frequency effect on the development concepts
    local REST; REST=$($PY - "$SPLIT/results.json" <<'EOF'
import json, sys
r = json.load(open(sys.argv[1]))
lm = r["lm_loss"]
change = lm["K8_without_super"] / lm["intact"] - 1
k = r["naming"]["K8_without_super"]["knowledge_effect"]
ok = change <= 0.05 and k["ci_low"] > 0
print(",".join(r["states"]["K8_without_super"]["components"]) if ok else "")
print(f"K8 without the super experts: LM loss {change:+.1%}, knowledge effect {k['estimate']:.2f} "
      f"[{k['ci_low']:.2f}, {k['ci_high']:.2f}] -> item 1b {'runs' if ok else 'skipped'}", file=sys.stderr)
EOF
)
    if [ -n "$REST" ]; then
        local CTRL; CTRL=$(csv $HELD/$P/dissociation/sets.json control)
        run phi_split_heldout "$(finished $HELD/$P/dissociation_split)" $PY experiments/e2_confirm/run_dissociation_test.py \
            $PARGS $SMALL --freeze-tag $TAG --knowledge-set "$REST" \
            --control-set "$CTRL" --knowledge-dir outputs/e3_knowledge$SUF/$P/screen --allow-score-code-change \
            --out $HELD/$P/dissociation_split || FAIL=1
    fi
    echo "$(stamp) Phi done (any step failed: $FAIL)"
    return $FAIL
}

if [ "$GPUS" = 1 ]; then
    echo "$(stamp) one card: lane A, lane B and Phi one after another"
    lane_a > "$LOGROOT/A.log" 2>&1; FA=$?
    lane_b > "$LOGROOT/B.log" 2>&1; FB=$?
    phi > "$LOGROOT/phi.log" 2>&1; FP=$?
    echo "$(stamp) session 12 done (failed: lane A $FA, lane B $FB, Phi $FP)"
    [ $FA -eq 0 ] && [ $FB -eq 0 ] && [ $FP -eq 0 ]
    exit $?
fi
lane_a > "$LOGROOT/A.log" 2>&1 &
A=$!
lane_b > "$LOGROOT/B.log" 2>&1; FB=$?
echo "$(stamp) lane B done; Phi starts on GPU 1 while lane A goes on (as in session 11: lane A's next model load may be
slower once Phi has pushed the main model out of the page cache, about 30 minutes at most)"
phi > "$LOGROOT/phi.log" 2>&1; FP=$?
wait $A; FA=$?
echo "$(stamp) session 12 done (failed: lane A $FA, lane B $FB, Phi $FP)"
[ $FA -eq 0 ] && [ $FB -eq 0 ] && [ $FP -eq 0 ]
