"""Audit source senses and family separation before independent semantic review. No model scores."""
import argparse
import json
import sys
from pathlib import Path

import nltk
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lesion import fine_semantics as fs, homonym_items as hi, runinfo


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", type=Path, default=ROOT / "reports/explore/2026-10-19-title-gap-preparation")
    args = ap.parse_args()
    source = ROOT / "data/materials/title_gap"
    fine = pd.read_csv(source / "fine/items.csv", keep_default_na=False)
    excluded = fine[fine.review == "excluded"]
    fine = fine[fine.review != "excluded"].copy()
    homonyms = pd.read_csv(source / "homonyms/items.csv", keep_default_na=False)
    fs.validate(fine)
    hi.validate(homonyms)
    nltk.data.path.insert(0, str(ROOT / "outputs/title_gap_sources"))
    from nltk.corpus import wordnet as wn
    review = fs.review_table(fine, wn, fs.old_words(ROOT / "data/materials"))
    args.out.mkdir(parents=True, exist_ok=True)
    review.to_csv(args.out / "fine-material-review.csv", index=False)
    manifests = {name: json.loads((source / name / "manifest.json").read_text(encoding="utf-8"))
                 for name in ("fine", "homonyms")}
    result = dict(material_status={name: m["status"] for name, m in manifests.items()}, fine_items=len(fine), excluded_items=len(excluded),
                  fine_flagged=int(review["flags"].ne("").sum()),
                  fine_flags=review["flags"].str.split("|").explode().value_counts().drop("", errors="ignore").to_dict(),
                  homonym_families=int(homonyms.family.nunique()), homonym_items=len(homonyms),
                  sources={str(p.relative_to(ROOT)): runinfo.sha256(p) for p in
                           (source / "fine/items.csv", source / "homonyms/items.csv", source / "homonyms/contexts.tsv")},
                  definition_note="WordNet relations verified, ambiguity requires separately recorded semantic review",
                  semantic_review={name: m.get("review") for name, m in manifests.items()},
                  property_items=int((fine.property_true.ne('') & fine.property_false.ne('')).sum()))
    (args.out / "material-audit.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["fine_flags"], indent=2))


if __name__ == "__main__":
    main()
