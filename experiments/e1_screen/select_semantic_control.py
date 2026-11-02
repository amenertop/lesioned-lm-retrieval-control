"""The semantic-only control set from a finished control screen (lesion.screen.select_semantic_control; exploratory,
user decision 2026-10-13): no new scoring, the rule is applied to the screen's single-component results.

    python experiments/e1_screen/select_semantic_control.py --model qwen3-30b-a3b-instruct

Writes outputs/e1_screen/<model>/semantic_control/selection.json and prints the set as comma-separated keys.
"""
import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))

from lesion import runinfo, screen  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tiny", action="store_true")
    args = ap.parse_args()
    base = ROOT / ("outputs/e1_screen_tiny" if args.tiny else "outputs/e1_screen") / args.model
    meta = json.loads((base / "screen/meta.json").read_text(encoding="utf-8"))
    if meta.get("status") != "done":
        raise SystemExit(f"{base / 'screen'} is unfinished")
    summary = pd.read_csv(base / "screen/screen.csv")
    sel = screen.select_semantic_control(summary)
    sel.update(source=str((base / "screen/screen.csv").relative_to(ROOT)).replace("\\", "/"),
               source_sha256=runinfo.sha256(base / "screen/screen.csv"), screen_commit=meta.get("resume_commit"),
               original_control=json.loads((base / "screen/selection.json").read_text(encoding="utf-8"))["control"])
    out = base / "semantic_control"
    out.mkdir(parents=True, exist_ok=True)
    (out / "selection.json").write_text(json.dumps(sel, indent=2), encoding="utf-8")
    if not sel["control"]:
        raise SystemExit(f"no semantic-only control set: {sel['status']}")
    print(",".join(sel["control"]))


if __name__ == "__main__":
    main()
