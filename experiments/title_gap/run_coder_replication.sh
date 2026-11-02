#!/usr/bin/env bash
# The second model uses its own screens and the existing frozen scoring rules. No coordinates are copied.
# Run from a clean tagged snapshot: bash .../run_coder_replication.sh MEANS CENSUS_RESULT FREEZE_TAG
set -uo pipefail
source experiments/session_lib.sh
MEANS=${1:?supply the verified Coder means file}
CENSUS=${2:?supply the completed Coder census result}
TAG=${3:?supply the analysis freeze tag}
PY=${CODER_PY:-/root/miniconda3/envs/lesion/bin/python}
MODEL=qwen3-coder-30b-a3b-instruct
LOG=outputs/title_gap/coder_replication/logs
CONTROL=outputs/e1_screen/$MODEL
NAMING=outputs/e3_knowledge/$MODEL
TEST=outputs/e2_confirm/$MODEL
mkdir -p "$LOG"
require_tag "$TAG"
export PYTHONPATH=src HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1
IDENTITY=outputs/title_gap/coder_replication/identity.json
"$PY" experiments/title_gap/check_coder_sources.py --identity "$IDENTITY" --means "$MEANS" --census "$CENSUS" --freeze-tag "$TAG" || exit 1
PREVIOUS=$IDENTITY
checked() {
    local name=$1 directory=$2 required=$3; shift 3
    local record=outputs/title_gap/coder_replication/records/$name.json
    echo "$(stamp) $name: checking or running"
    "$PY" experiments/title_gap/check_coder_sources.py --identity "$IDENTITY" --directory "$directory" \
        --record "$record" --input "$PREVIOUS" --required $required -- "$@" >> "$LOG/$name.log" 2>&1 || return 1
    PREVIOUS=$record
}
ARGS=(--model "$MODEL" --means "$MEANS" --dtype float16 --moe sorted --scorer prefix_avg3 --batch-size 16)

checked prefix "$CONTROL/check_prefix" 'check_prefix.json' "$PY" experiments/e1_screen/check_prefix.py "${ARGS[@]}" --n-experts 4 || exit 1
"$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["pass"] else 1)' "$CONTROL/check_prefix/check_prefix.json" || exit 1
checked attribution "$CONTROL/attribution" 'attribution.csv meta.json' "$PY" experiments/e1_screen/run_attribution.py "${ARGS[@]}" --kinds expert,head --batch-size 4 || exit 1
checked screen "$CONTROL/screen" 'screen.csv selection.json meta.json intact.csv' "$PY" experiments/e1_screen/run_screen.py "${ARGS[@]}" --kind expert+head --candidates "$CONTROL/attribution/attribution.csv" --top 32 --flush-every 8 || exit 1
checked joint "$CONTROL/joint" 'results.json sets.json intact.csv lesioned.parquet meta.json' "$PY" experiments/e1_screen/run_joint.py "${ARGS[@]}" --n-random 20 || exit 1

checked naming_check "$NAMING/check" 'summary.json scores.csv meta.json' "$PY" experiments/e3_knowledge/check_items.py --model "$MODEL" --dtype float16 --average-batches --moe sorted || exit 1
"$PY" -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1]))["passed"] else 1)' "$NAMING/check/summary.json" || exit 1
checked knowledge "$NAMING/screen" 'screen.csv selection.json joint.json attribution.csv meta.json' "$PY" experiments/e3_knowledge/run_knowledge.py "${ARGS[@]}" --n-random 20 || exit 1
checked expand8 "$NAMING/expand8_dev120" 'results.json meta.json' "$PY" experiments/e3_knowledge/run_expand8.py "${ARGS[@]}" --n-random 20 --knowledge-dir "$NAMING/screen" --out "$NAMING/expand8_dev120" || exit 1

checked control_test "$TEST/control" 'results.json sets.json intact.csv lesioned.parquet meta.json' "$PY" experiments/e2_confirm/run_control_test.py "${ARGS[@]}" --n-random 200 --freeze-tag "$TAG" --out "$TEST/control" || exit 1
checked dissociation "$TEST/dissociation" 'results.json sets.json naming_scores.parquet m1_scores.parquet meta.json' "$PY" experiments/e2_confirm/run_dissociation_test.py "${ARGS[@]}" --n-random 200 --freeze-tag "$TAG" --knowledge-dir "$NAMING/screen" --expansion-dir "$NAMING/expand8_dev120" --out "$TEST/dissociation" || exit 1
echo "$(stamp) Coder replication completed"
