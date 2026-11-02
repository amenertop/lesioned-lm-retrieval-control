"""Analyze complete title-gap runs, retaining raw per-concept tables and exploratory status."""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lesion import runinfo, semantic_diversity as sd, title_gap_analysis as ta


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", type=Path, action="append", required=True)
    ap.add_argument("--items", type=Path, required=True)
    ap.add_argument("--norms", type=Path, help="SemD XLSX for the fine battery")
    ap.add_argument("--lexicon", type=Path, help="human association meaning lexicon for homonyms")
    ap.add_argument("--review", type=Path, help="blinded response_id/category judgements for otherwise unclassified answers")
    ap.add_argument("--review-meta", type=Path, help="reviewer identity and hashes of the blind queue and judgements")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--report", type=Path, help="ledger-ready battery summary")
    ap.add_argument("--specificity-report", type=Path, help="post hoc target-versus-neighbour summary, fine battery only")
    ap.add_argument("--n-boot", type=int, help="development default 2000, final value comes from the frozen protocol")
    ap.add_argument("--compare", action="append", default=[], help="paired state names, first:second")
    args = ap.parse_args()
    if args.review_meta and not args.review:
        ap.error('review metadata requires its judgement table')
    tables, plan = ta.read_runs(args.run)
    if runinfo.sha256(args.items) != plan["provenance"]["items_sha256"]:
        ap.error("materials differ from the source run")
    confirmed = plan['provenance']['split'] == 'test' and not plan['provenance']['tiny']
    if confirmed:
        protocol_path = ROOT / f"docs/analysis-plan/title-gap-{plan['provenance']['battery']}.json"
        tag = plan['provenance'].get('freeze_tag')
        if not tag or not tag.startswith('analysis-frozen'):
            ap.error('final analysis requires a recorded freeze tag')
        digest = runinfo.sha256(protocol_path)
        if (digest != plan['provenance'].get('protocol_sha256')
                or digest != runinfo.sha256_at(tag, protocol_path.relative_to(ROOT))):
            ap.error('analysis protocol differs from the scored and tagged version')
        protocol = json.loads(protocol_path.read_text(encoding='utf-8'))
        source_runs = plan.get('source_runs', [dict(provenance=plan['provenance'])])
        if (len(source_runs) != len(protocol['states_sha256'])
                or {r['provenance']['states_sha256'] for r in source_runs} != set(protocol['states_sha256'].values())):
            ap.error('final analysis requires every frozen core and random run')
        if args.n_boot is not None and args.n_boot != protocol['n_boot']:
            ap.error('final bootstrap count is fixed by the protocol')
        if args.compare and set(args.compare) != set(protocol['comparisons']):
            ap.error('final comparisons are fixed by the protocol')
        args.n_boot, args.compare = protocol['n_boot'], protocol['comparisons']
        if plan['provenance']['battery'] == 'homonyms':
            if not args.lexicon or runinfo.sha256(args.lexicon) != protocol['lexicon_sha256']:
                ap.error('human-response lexicon differs from the frozen source')
        for path, expected in protocol['sources_sha256'].items():
            if runinfo.sha256(ROOT / path) != expected:
                ap.error(f'analysis source differs from freeze: {path}')
    elif args.n_boot is None:
        args.n_boot = 2000
    if not confirmed and args.out.resolve().is_relative_to((ROOT / "reports/confirm").resolve()):
        ap.error("new analyses remain exploratory until an analysis plan is frozen")
    items = pd.read_csv(args.items, keep_default_na=False)
    items = items[items.split == plan["provenance"]["split"]]
    if plan["provenance"]["battery"] == "fine":
        if not args.norms:
            ap.error("fine analysis requires the norm source")
        items = sd.match_words(items, sd.load_norms(args.norms))
        items["frequency"] = items.log_bnc_freq
        result, frame = ta.fine_profiles(tables, items, args.n_boot)
        review = pd.read_csv(args.review) if args.review else None
        response_result, response_rows = ta.fine_responses(tables, items, args.n_boot, review)
    else:
        if not args.lexicon:
            ap.error("homonym analysis requires the human-response lexicon")
        lexicon = {(r["word"], r["answer"]): r["senses"] for r in json.loads(args.lexicon.read_text(encoding="utf-8"))}
        review = pd.read_csv(args.review) if args.review else None
        result, frame = ta.homonym_recovery(tables["generated"], items, lexicon, args.n_boot, review)
        score_result, score_rows = ta.homonym_scores(tables, items, args.n_boot)
    args.out.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out / "items.csv", index=False)
    if plan["provenance"]["battery"] == "fine":
        response_rows.to_csv(args.out / "responses.csv", index=False)
        response_rows.query("category == 'unclassified'")[["response_id", "target", "synset", "definition", "condition", "answer"]].drop_duplicates("response_id").to_csv(args.out / "blind-response-review.csv", index=False)
    if plan["provenance"]["battery"] == "homonyms":
        score_rows.to_csv(args.out / "candidate-and-choice-items.csv", index=False)
        queue = (frame[frame.initial_category == 'unclassified'][["response_id", "word", "sense", "context_high", "answer"]]
                 .sort_values(['response_id', 'answer']).drop_duplicates("response_id"))
        queue.to_csv(args.out / "blind-response-review.csv", index=False, lineterminator='\n')
        if confirmed and (args.review or len(queue)):
            if not args.review or not args.review_meta:
                ap.error('final judgements require reviewer identity and queue provenance')
            if set(review.response_id) != set(queue.response_id):
                ap.error('every final unknown answer needs a judgement or explicit abstention')
            review_meta = json.loads(args.review_meta.read_text(encoding='utf-8'))
            if (review_meta.get('queue_sha256') != runinfo.sha256(args.out / 'blind-response-review.csv')
                    or review_meta.get('review_sha256') != runinfo.sha256(args.review)
                    or not all(review_meta.get(k) for k in ('reviewer_model', 'review_independence', 'acceptance_status'))):
                ap.error('review identity or source hashes do not match the combined blind queue')
    contrast = ta.fine_contrast if plan["provenance"]["battery"] == "fine" else ta.homonym_contrast
    comparisons = {pair: contrast(frame, *pair.split(":"), args.n_boot) for pair in args.compare}
    payload = dict(status="tiny_rehearsal" if plan["provenance"]["tiny"] else "internal_holdout_confirmation" if confirmed else "exploratory",
                   provenance=plan["provenance"], source_fingerprint=plan["fingerprint"], n_boot=args.n_boot,
                   analysis_commit=runinfo.commit(), results=result, contrasts=comparisons)
    if 'source_runs' in plan:
        payload['source_runs'] = plan['source_runs']
    if args.review_meta:
        payload['review_metadata'] = json.loads(args.review_meta.read_text(encoding='utf-8'))
    if plan["provenance"]["battery"] == "fine":
        payload["actual_answers"] = response_result
    else:
        payload["candidate_and_choice"] = score_result
        payload["candidate_and_choice_contrasts"] = {
            readout: {pair: ta.homonym_contrast(rows, *pair.split(":"), args.n_boot, endpoint="score")
                      for pair in args.compare} for readout, rows in score_rows.groupby("readout")}
        payload['matched_random'] = dict(semantic=ta.homonym_random_comparison(frame, plan['states']),
            **{readout: ta.homonym_random_comparison(rows, plan['states'], 'score')
               for readout, rows in score_rows.groupby('readout')})
    tables["language"].to_csv(args.out / "language.csv", index=False)
    payload["language"] = tables["language"].to_dict("records")
    payload["analysis_sources"] = {str(p.resolve().relative_to(ROOT)): runinfo.sha256(p) for p in
                                   (Path(__file__), ROOT / "src/lesion/title_gap_analysis.py", args.items)}
    payload["auxiliary_sources"] = {p.name: runinfo.sha256(p) for p in (args.norms, args.lexicon, args.review, args.review_meta) if p}
    (args.out / "result.json").write_text(json.dumps(payload, indent=2, default=float) + "\n", encoding="utf-8")
    if args.report:
        report = (ta.fine_ledger_summary(payload, frame, len(plan["neutral_sequences"]))
                  if plan["provenance"]["battery"] == "fine" else ta.homonym_ledger_summary(payload))
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8", newline="\n")
    if args.specificity_report:
        if plan["provenance"]["battery"] != "fine":
            ap.error("candidate specificity requires the fine battery")
        states, contrasts = ta.fine_candidate_specificity(tables["candidates"], items, args.n_boot)
        contrasts.to_csv(args.out / "candidate-specificity-items.csv", index=False)
        meta = dict(status="posthoc_candidate_competition", provenance=plan["provenance"], source_fingerprint=plan["fingerprint"],
                    n_boot=args.n_boot, analysis_sources=payload["analysis_sources"],
                    definition="paired target or summed target/synonym log probability relative to a coordinate foil, whole word without end token")
        args.specificity_report.parent.mkdir(parents=True, exist_ok=True)
        args.specificity_report.write_text(json.dumps(dict(meta=meta, states=states), indent=2, allow_nan=False)+"\n", encoding="utf-8")


if __name__ == "__main__":
    main()
