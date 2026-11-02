"""Run records shared by the experiment entry scripts (docs/plans/code-architecture.md §3.4).

experiments/e0_baseline keeps its own frozen copy of these checks; new experiments use this module.
"""
import hashlib
import json
import platform
import subprocess
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS_PLAN = "docs/analysis-plan"
ANALYSIS_TAG_PREFIX = "analysis-frozen"
TRACKED = ("data/materials", "data/neutral_text", "src", "experiments", ANALYSIS_PLAN)


def git(*args):
    return subprocess.run(["git", *args], cwd=ROOT, capture_output=True, text=True).stdout.strip()


def commit():
    return git("rev-parse", "HEAD")


def dirty_files(paths=TRACKED):
    return git("status", "--porcelain", "--", *paths)


def require_clean(paths=TRACKED):
    """Commit hash of HEAD; refuses to run when code or materials have uncommitted changes."""
    dirty = dirty_files(paths)
    if dirty:
        raise SystemExit(f"Refusing to run: uncommitted changes, commit them first so the run can be traced:\n{dirty}")
    return commit()


def require_tag(tag):
    """Commit hash of HEAD, which must be `tag` on a clean tree (confirmatory runs after a freeze)."""
    head = require_clean()
    if head != git("rev-parse", f"{tag}^{{commit}}"):
        raise SystemExit(f"Refusing to run: HEAD must be the `{tag}` tag.")
    return head


def differs_from(rev, paths=TRACKED):
    """Tracked files under paths whose working copy differs from rev, and untracked files there (a run or an
    analysis that must use the code and plan of a freeze tag)."""
    changed = git("diff", "--name-only", rev, "--", *paths).splitlines()
    untracked = git("ls-files", "--others", "--exclude-standard", "--", *paths).splitlines()
    return changed + untracked


def check_analysis_freeze(tag, whole_tree=False):
    """Commit of HEAD when it is the analysis-plan freeze `tag` on a clean tree; PermissionError otherwise.

    Guards the held-out test split: the tag name must start with ANALYSIS_TAG_PREFIX, the tag must exist,
    HEAD must be that commit with nothing uncommitted, and the tagged tree must hold the analysis plan.
    whole_tree: no tracked file anywhere may be modified, not only those under TRACKED (the optogenetics
    confirmation also records the protocol under docs/plans; agent3 2026-10-13, N2).
    """
    if not tag or not tag.startswith(ANALYSIS_TAG_PREFIX):
        raise PermissionError(f"the held-out test split is locked; pass the analysis-plan freeze tag ({ANALYSIS_TAG_PREFIX}...)")
    tagged = git("rev-parse", "--verify", "--quiet", f"{tag}^{{commit}}")
    if not tagged:
        raise PermissionError(f"no tag {tag!r}")
    if commit() != tagged or dirty_files() or (whole_tree and git("status", "--porcelain", "--untracked-files=no")):
        raise PermissionError(f"HEAD must be the `{tag}` tag with no uncommitted changes")
    plan = [f for f in git("ls-tree", "-r", "--name-only", tag, "--", ANALYSIS_PLAN).splitlines() if f.endswith(".md")]
    if not plan:
        raise PermissionError(f"`{tag}` holds no analysis plan under {ANALYSIS_PLAN}/")
    return tagged


def analysis_plan_sha256():
    """One hash over the analysis-plan files (path and content), None while the plan is empty."""
    files = sorted((ROOT / ANALYSIS_PLAN).glob("**/*.md"))
    if not files:
        return None
    h = hashlib.sha256()
    for f in files:
        h.update(f.relative_to(ROOT).as_posix().encode() + b"\0" + f.read_bytes())
    return h.hexdigest()


def analysis_plan_sha256_at(rev):
    """analysis_plan_sha256 of the plan as committed at `rev` (a tag or commit), read from git rather than the
    working tree; None if rev holds no plan."""
    files = sorted(f for f in git("ls-tree", "-r", "--name-only", rev, "--", ANALYSIS_PLAN).splitlines() if f.endswith(".md"))
    if not files:
        return None
    h = hashlib.sha256()
    for f in files:
        h.update(f.encode() + b"\0" + git_bytes("show", f"{rev}:{f}"))
    return h.hexdigest()


def sha256_at(rev, path):
    """SHA-256 of a tracked file as committed at `rev`."""
    return hashlib.sha256(git_bytes("show", f"{rev}:{Path(path).as_posix()}")).hexdigest()


def git_bytes(*args):
    out = subprocess.run(["git", *args], cwd=ROOT, capture_output=True)
    if out.returncode:
        raise ValueError(f"git {' '.join(args)}: {out.stderr.decode(errors='replace').strip()}")
    return out.stdout


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def environment():
    import torch
    import transformers

    return dict(torch=torch.__version__, transformers=transformers.__version__, cuda=torch.version.cuda,
                gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                python=platform.python_version(), host=platform.node())


def write_meta(out_dir, **fields):
    """Write out_dir/meta.json: the given fields plus commit, clean-tree flag, environment and time."""
    meta = dict(fields)
    meta.update(git_commit=commit(), clean_tree=not dirty_files(), analysis_plan_sha256=analysis_plan_sha256(), **environment(),
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S"))
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    return meta
