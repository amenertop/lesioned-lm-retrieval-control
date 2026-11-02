"""Project final-sample precision from saved development differences without any new model computation."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from lesion import runinfo, title_gap_precision as precision


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--analysis', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    score_path = args.analysis / 'candidate-and-choice-items.csv'
    answer_path = args.analysis / 'items.csv'
    source_path = args.analysis / 'result.json'
    source = json.loads(source_path.read_text(encoding='utf-8'))
    if source['provenance']['split'] != 'dev' or source['provenance']['battery'] != 'homonyms':
        ap.error('input must be a homonym development analysis')
    scores, responses = pd.read_csv(score_path), pd.read_csv(answer_path)
    payload = dict(meta=dict(status='design_projection_not_experiment_result', seed=1910, n_sim=10000,
        source_fingerprint=source['source_fingerprint'], analysis_commit=runinfo.commit(),
        inputs={p.resolve().relative_to(ROOT).as_posix(): runinfo.sha256(p) for p in (score_path, answer_path, source_path,
            Path(__file__), ROOT / 'src/lesion/title_gap_precision.py')},
        assumptions='Development family noise transfers to held-out families. Selected mean effects are centred out. '
                    'Exact target is a secondary readout, not a substitute for unresolved semantic answers. '
                    'Correlation power is an approximation for one test, not the complete clinical conjunction.'),
        homonym_n40=precision.homonym_precision(scores, responses, 40),
        homonym_n80=precision.homonym_precision(scores, responses, 80),
        correlation_n63=precision.correlation_sensitivity(63),
        correlation_n120=precision.correlation_sensitivity(120))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, allow_nan=False)+'\n', encoding='utf-8', newline='\n')
    print(args.out)


if __name__ == '__main__':
    main()
