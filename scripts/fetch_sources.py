"""Download the Wu and Hoffman (2022) files from OSF (project 9px7g) and rebuild the M1 item files.

    python scripts/fetch_sources.py

The rebuilt files must match the SHA-256 of the files used in the paper. Other third-party sources are listed in
data/raw/SOURCES.md.
"""
import hashlib
import sys
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
RAW = ROOT / "data" / "raw" / "wu_hoffman_2022"
FILES = {"readme.txt": "bx9jw", "Item_properties.csv": "fqnas", "Item_properties_key.csv": "8z9xm"}
EXPECTED = {'items.csv': '2ac9cb4fe51bf6090a8bfe2e9c612c60dfa45cdbd2d6d81860aebd6531a379e6', 'associations.csv': '6c8b4f487f39363a257d45b025adfbde098f810bc7e2751494521ddff66a0b92'}


def main():
    RAW.mkdir(parents=True, exist_ok=True)
    for name, osf_id in FILES.items():
        target = RAW / name
        if not target.exists():
            print("downloading", name)
            urllib.request.urlretrieve(f"https://osf.io/download/{osf_id}/", target)
    from lesion import materials as m
    out = ROOT / "data" / "materials" / "m1"
    m.format_items(m.load_raw()).to_csv(out / "items.csv", index=False, lineterminator="\n", float_format="%.6f")
    m.write_associations(out=out / "associations.csv")
    ok = True
    for name, sha in EXPECTED.items():
        got = hashlib.sha256((out / name).read_bytes()).hexdigest()
        print(name, "matches the paper" if got == sha else f"DIFFERS from the paper ({got})")
        ok &= got == sha
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
