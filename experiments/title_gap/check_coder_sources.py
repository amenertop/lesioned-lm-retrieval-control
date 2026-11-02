"""Check the census and means once, or execute one source-locked Coder stage."""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from lesion import coder_sources as cs


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--identity", type=Path, required=True)
    ap.add_argument("--means", type=Path)
    ap.add_argument("--census", type=Path)
    ap.add_argument("--freeze-tag")
    ap.add_argument("--directory", type=Path)
    ap.add_argument("--record", type=Path)
    ap.add_argument("--required", nargs="+", default=[])
    ap.add_argument("--input", type=Path, action="append", default=[])
    ap.add_argument("command", nargs=argparse.REMAINDER)
    args = ap.parse_args()
    if args.means:
        identity = cs.preflight(args.means, args.census, args.freeze_tag)
        if args.identity.exists() and cs.read(args.identity) != identity:
            ap.error("replication identity changed, use a fresh snapshot")
        args.identity.parent.mkdir(parents=True, exist_ok=True)
        args.identity.write_text(json.dumps(identity, indent=2) + "\n", encoding="utf-8")
    else:
        command = args.command[1:] if args.command[:1] == ["--"] else args.command
        if not command or not args.directory or not args.record or not args.required:
            ap.error("stage requires directory, record, expected outputs and a command")
        ran = cs.guarded_stage(args.directory, args.record, cs.read(args.identity), command, args.required, args.input)
        print("completed" if ran else "verified completed outputs", flush=True)


if __name__ == "__main__":
    main()
