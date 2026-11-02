"""Transactional storage and completeness checks for long, resumable runs (agent2 reviews 2026-10-12, -13; agent3 review
2026-10-13).

Each unit of work (a lesion state, an implant condition, the router counts) writes its tables into its own folder,
first under a temporary name. When the store is given a content check (validate), the tables must pass it before
the unit is finished: a unit that fails is moved to <root>/../invalid/<name>-<time> for diagnosis and the run stops
there, rather than computing the remaining units first. A marker file listing every table's row count and SHA-256
is written last and the folder is then renamed into place. A unit counts as done only when its marker reads, names
the unit, lists at least one table (exactly the tables the unit must hold, when the store is told which), every
listed table exists with that row count (a whole number) and hash, no other table is in the folder, and the tables
still pass the content check. Anything else (a missing, unreadable or damaged marker, a changed table) makes the
unit unfinished: clean() removes it and the run redoes it, so a run stopped at any point resumes without appending
twice. At the end the units are concatenated, in plan order, into one parquet per table.

begin() starts or resumes a run folder against its plan fingerprint (plan.json); write_json() writes the small
bookkeeping files atomically, so a stop while writing them cannot lock the run; a file that reads but is not what
it should be (a list, a number) counts as damaged too.

check_tables verifies a finished run (or merged shards, or one unit) against what the plan says it must hold.
assembled_problems checks that a finished folder's tables are exactly what its units add up to (the evidence that
one process computed them; agent3 review 2026-10-13, F2).
"""
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

MARKER = "done.json"
KEYS = dict(candidates=["item_id", "condition", "role", "word"], labels=["item_id", "fmt", "variant"])
DESCRIPTION = ["set", "alpha", "region"]                     # the lesion a row was computed under
# the columns every table must hold and their kinds (agent3 2026-10-13, N3 and N11); other columns are allowed
COLUMNS = dict(
    candidates=dict(state=str, set=str, alpha=float, region=str, item_id=str, condition=str, role=str, word=str,
                    logprob=float, n_tokens=int, first_token_rank=int),
    labels=dict(state=str, set=str, alpha=float, region=str, item_id=str, fmt=str, variant=int, correct=str,
                margin=float, top_correct=bool),
    greedy=dict(state=str, set=str, alpha=float, region=str, item_id=str, answer=str, kind=str),
    lm_loss=dict(state=str, set=str, alpha=float, region=str, lm_loss=float),
    router=dict(item_id=str, component=str, selected_last=bool, n_selected=int, n_positions=int))


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, obj):
    """Write obj as JSON atomically: a temporary file renamed over the target."""
    path = Path(path)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
    os.replace(tmp, path)


def begin(partial, fingerprint):
    """Start the run folder `partial` with plan.json = fingerprint, or resume it if its plan is the same; returns
    True when resuming. A missing or damaged plan.json restarts the folder only if it holds no unit yet; with units
    whose plan can no longer be checked it refuses (agent3 2026-10-13, N9)."""
    partial = Path(partial)
    plan = partial / "plan.json"
    want = json.loads(json.dumps(fingerprint, default=str))
    if partial.exists():
        try:
            old = json.loads(plan.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            old = None
        if not isinstance(old, dict):              # missing, damaged, or valid JSON of another kind (agent3 F5)
            units = partial / "units"
            if units.exists() and any(units.iterdir()):
                raise SystemExit(f"{plan} is missing or damaged and {partial} holds units whose plan cannot be "
                                 "checked; move the folder away to start afresh")
            write_json(plan, want)
            return False
        if old != want:
            diff = sorted(k for k in set(old) | set(want) if old.get(k) != want.get(k))
            raise SystemExit(f"{partial} was started under a different plan ({diff}); move it away to start afresh")
        return True
    partial.mkdir(parents=True)
    write_json(plan, want)
    return False


def read_json(path, default, ok=None):
    """A bookkeeping file that is not part of a run's identity (timing): its content, or default if it is missing,
    damaged, of another kind than default (a list for a dict) or, given ok, not ok(content) (agent3 2026-10-13, F5)."""
    try:
        obj = json.loads(Path(path).read_text(encoding="utf-8"))
        if isinstance(obj, type(default)) and (ok is None or ok(obj)):
            return obj
    except (OSError, ValueError):
        pass
    if Path(path).exists():
        print(f"ignoring damaged {path}", flush=True)
    return default


class Store:
    """Units under root/<name>/ (names may contain + @ . _ but no path separators). required: optional function
    unit name -> the set of tables that unit must hold; validate: optional function (unit name, folder) that raises
    ValueError unless the folder's tables are what the plan says for that unit."""

    def __init__(self, root, required=None, validate=None):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.required = required
        self.validate = validate

    def _dir(self, name):
        if "/" in name or "\\" in name or name.startswith("."):
            raise ValueError(f"bad unit name {name!r}")
        return self.root / name

    def valid(self, name):
        d = self._dir(name)
        try:
            marker = json.loads((d / MARKER).read_text(encoding="utf-8"))
            tables = marker["tables"]
            if marker["name"] != name or not isinstance(tables, dict) or not tables:
                return False
            if {f.name[:-len(".parquet")] for f in d.glob("*.parquet")} != set(tables):
                return False
            if self.required is not None and set(tables) != set(self.required(name)):
                return False
            for t, v in tables.items():
                rows = v["rows"]
                if type(rows) is not int or rows < 0:          # not 1.9, true or 1e999 (agent3 2026-10-13, N8)
                    return False
                f = d / f"{t}.parquet"
                if _sha(f) != v["sha256"] or pq.read_metadata(f).num_rows != rows:
                    return False
            if self.validate is not None:
                self.validate(name, d)
        except (OSError, ValueError, KeyError, TypeError, AttributeError, OverflowError, IndexError):
            return False
        return True

    def done(self):
        return {d.name for d in self.root.iterdir() if d.is_dir() and not d.name.endswith(".tmp") and self.valid(d.name)}

    def clean(self):
        """Remove temporary and unfinished unit folders (left by a stopped run); returns their names."""
        removed = []
        for d in list(self.root.iterdir()):
            if d.is_dir() and (d.name.endswith(".tmp") or not self.valid(d.name)):
                shutil.rmtree(d)
                removed.append(d.name)
        return removed

    def write(self, name, tables):
        """Write {table: DataFrame} (None entries skipped) as unit `name`, atomically; refuses a unit with no table
        or, when the store knows them, with other tables than the unit must hold. With a content check, a unit
        that fails it, or on which the check itself fails (a missing column, say), is kept under ../invalid/ and
        ValueError is raised: the caller stops (agent3 2026-10-13, F3)."""
        final = self._dir(name)
        if final.exists():
            raise FileExistsError(f"unit {name} already written")
        present = {t for t, df in tables.items() if df is not None}
        if not present:
            raise ValueError(f"unit {name}: no table to write")
        if self.required is not None and present != set(self.required(name)):
            raise ValueError(f"unit {name}: tables {sorted(present)}, the plan says {sorted(self.required(name))}")
        tmp = final.with_name(final.name + ".tmp")
        if tmp.exists():
            shutil.rmtree(tmp)
        tmp.mkdir()
        marker = dict(name=name, tables={})
        for t in sorted(present):
            f = tmp / f"{t}.parquet"
            tables[t].to_parquet(f, index=False)
            marker["tables"][t] = dict(rows=int(len(tables[t])), sha256=_sha(f))
        if self.validate is not None:
            try:
                self.validate(name, tmp)
            except Exception as e:
                keep = self.root.parent / "invalid" / f"{name}-{time.strftime('%Y%m%d-%H%M%S')}"
                keep.parent.mkdir(parents=True, exist_ok=True)
                tmp.rename(keep)
                raise ValueError(f"unit {name} fails the plan's check ({e}); its tables are kept in {keep}") from e
        (tmp / MARKER).write_text(json.dumps(marker, indent=1), encoding="utf-8")
        tmp.rename(final)

    def read(self, table, names):
        """The table concatenated over the given units, in that order (units without that table are skipped)."""
        parts = [pd.read_parquet(self._dir(n) / f"{table}.parquet") for n in names
                 if (self._dir(n) / f"{table}.parquet").exists()]
        return pd.concat(parts, ignore_index=True) if parts else None


def assemble(store, names, tables, out_dir):
    """Concatenate each table over the units into out_dir/<table>.parquet (atomic replace); returns the tables
    written."""
    written = []
    for t in tables:
        df = store.read(t, names)
        if df is None:
            continue
        tmp = Path(out_dir) / f"{t}.parquet.tmp"
        df.to_parquet(tmp, index=False)
        tmp.replace(Path(out_dir) / f"{t}.parquet")
        written.append(t)
    return written


def same_content(path, df):
    """Whether the parquet file holds df's rows, in order, with the same columns and values (NaN equal to NaN);
    the contents, not the file's bytes, which another writer may lay out differently."""
    try:
        pd.testing.assert_frame_equal(pd.read_parquet(path).reset_index(drop=True), df.reset_index(drop=True),
                                      check_dtype=False, check_exact=True)
    except AssertionError:
        return False
    return True


def assembled_problems(d, names, validate=None):
    """What keeps run folder d from being what its units add up to (agent3 2026-10-13, F2): d/units must hold exactly
    the units `names`, each finished (marker, row counts, hashes) and, given validate, passing that content check;
    and every table of d must be the concatenation of the units' tables in the order `names`, row for row and
    value for value, with no table the units do not hold. An empty list when it is."""
    d = Path(d)
    root = d / "units"
    if not root.is_dir():
        return [f"{d.name}: no units, so nothing shows which process computed its tables"]
    have = {p.name for p in root.iterdir()}
    if have != set(names):
        return [f"{d.name}: units {sorted(have ^ set(names))[:5]} missing or unexpected"]
    store = Store(root, validate=validate)
    bad = [n for n in names if not store.valid(n)]
    if bad:
        return [f"{d.name}: unfinished units or units outside the plan: {bad[:5]}"]
    in_units = {f.name[:-len(".parquet")] for n in names for f in (root / n).glob("*.parquet")}
    tables = {f.name[:-len(".parquet")] for f in d.glob("*.parquet")}
    if tables != in_units:
        return [f"{d.name}: tables {sorted(tables)}, its units hold {sorted(in_units)}"]
    differ = [t for t in sorted(tables) if not same_content(d / f"{t}.parquet", store.read(t, names))]
    return [f"{d.name}: tables {differ} are not the concatenation of its units"] if differ else []


def tables_sha256(d):
    """{table: SHA-256} of the parquet files in folder d."""
    return {f.name[:-len(".parquet")]: _sha(f) for f in sorted(Path(d).glob("*.parquet"))}


def _keyset(df, cols):
    return set(df[cols].itertuples(index=False, name=None))


def _kind_ok(s, kind):
    if kind is bool:
        return pd.api.types.is_bool_dtype(s)
    if kind is int:
        return pd.api.types.is_integer_dtype(s) and not pd.api.types.is_bool_dtype(s)
    if kind is float:
        return pd.api.types.is_float_dtype(s)
    return bool(s.map(lambda v: isinstance(v, str)).all())


def _columns(df, name):
    want = COLUMNS[name]
    missing = [c for c in want if c not in df]
    if missing:
        raise ValueError(f"{name}: missing columns {missing}")
    empty = [c for c in want if df[c].isna().any()]          # a nullable column's NA is the right type (agent3 F3)
    if empty:
        raise ValueError(f"{name}: missing values in {empty}")
    wrong = [c for c, kind in want.items() if not _kind_ok(df[c], kind)]
    if wrong:
        raise ValueError(f"{name}: wrong value types in {wrong}")


def _described(df, name, describe):
    """Every row carries its state's lesion (set, alpha, region) as the plan describes it."""
    got = set(df[["state", *DESCRIPTION]].itertuples(index=False, name=None))
    want = {(s, *describe[s]) for s in set(df.state)}
    if got != want:
        raise ValueError(f"{name}: rows describe another lesion than the plan: {sorted(got - want)[:3]}")


def check_tables(d, states=None, whole_states=None, keys=None, describe=None, item_ids=None, router=None,
                 implant_rows=None):
    """Raise ValueError unless the run folder d holds what the plan says, and nothing else (agent2 2026-10-13, agent3
    2026-10-13):
    - the parquet files are exactly the tables the arguments below call for, each with the columns and value kinds
      of COLUMNS;
    - candidates and labels: exactly `states`; in every state exactly the rows keys["candidates"] (item, condition,
      role, word) and keys["labels"] (item, format, variant -> the correct label and the labels read), once each;
      finite log-probabilities, at least one answer token, ranks from 0; finite margins, the correct label the
      prompt's, finite logits of the labels the item reads (the unused ones may be missing);
    - every row of a state describes the lesion of describe[state] (set, alpha, region);
    - greedy: exactly whole_states x item_ids, one answer each;
    - lm_loss: one finite, positive loss per whole state;
    - router: exactly item_ids x router (the component keys), once each; whole counts with
      0 <= n_selected <= n_positions, n_positions > 0 and the same for every component of a concept; selected_last
      a boolean, true only if n_selected >= 1 and true whenever n_selected = n_positions;
    - implant: implant_rows {(set, condition): expected rows}, nothing more.
    Arguments left as None are not checked (and their tables not expected)."""
    d = Path(d)
    want_items = None if item_ids is None else set(item_ids)
    expected = set()
    if states is not None:
        expected |= {"candidates", "labels"}
    if whole_states:
        expected |= {"greedy", "lm_loss"}
    if router is not None:
        expected.add("router")
    if implant_rows is not None:
        expected.add("implant")
    present = {f.name[:-len(".parquet")] for f in d.glob("*.parquet")}
    if present != expected:
        raise ValueError(f"tables {sorted(present)}, the plan calls for {sorted(expected)}")

    def table(name):
        df = pd.read_parquet(d / f"{name}.parquet")
        if name in COLUMNS:
            _columns(df, name)
        return df

    if states is not None:
        if keys is None or describe is None:
            raise ValueError("check_tables needs the expected keys and lesion of every state")
        for name, cols in KEYS.items():
            df = table(name)
            if df[["state", *cols]].isna().any().any():
                raise ValueError(f"{name}: missing key values")
            if df.duplicated(["state", *cols]).any():
                raise ValueError(f"{name}: duplicate rows")
            if set(df.state) != set(states):
                raise ValueError(f"{name}: states {sorted(set(df.state) ^ set(states))} missing or unexpected")
            want = set(keys[name])
            for s, g in df.groupby("state"):
                got = _keyset(g, cols)
                if got != want:
                    extra, missing = sorted(got - want)[:3], sorted(want - got)[:3]
                    raise ValueError(f"{name}: state {s} has rows outside the plan {extra} or lacks {missing}")
            _described(df, name, describe)
            if name == "candidates":
                if not (np.isfinite(df.logprob.to_numpy(dtype=float)).all() and (df.n_tokens >= 1).all()
                        and (df.first_token_rank >= 0).all()):
                    raise ValueError("candidates: non-finite log-probability or impossible token count or rank")
            else:
                if not np.isfinite(df.margin.to_numpy(dtype=float)).all():
                    raise ValueError("labels: non-finite margin")
                info = [keys["labels"][k] for k in df[cols].itertuples(index=False, name=None)]
                absent = sorted({f"logit_{x}" for _, r in info for x in r} - set(df.columns))
                if absent:
                    raise ValueError(f"labels: missing columns {absent}")
                if [c for c, _ in info] != df.correct.tolist():
                    raise ValueError("labels: a correct label differs from the prompt's")
                for read in {r for _, r in info}:
                    rows = np.array([r == read for _, r in info])
                    lg = df.loc[rows, [f"logit_{x}" for x in read]].to_numpy(dtype=float)
                    if not np.isfinite(lg).all():
                        raise ValueError(f"labels: non-finite logits among {''.join(read)}")
    if whole_states:
        g = table("greedy")
        if g.duplicated(["state", "item_id"]).any() or set(g.state) != set(whole_states):
            raise ValueError("greedy: states missing, unexpected or duplicated")
        for s in whole_states:
            if set(g.item_id[g.state == s]) != want_items:
                raise ValueError(f"greedy: state {s} does not hold exactly the concepts")
        _described(g, "greedy", describe)
        lm = table("lm_loss")
        if sorted(lm.state) != sorted(whole_states):
            raise ValueError("lm_loss: not exactly one row per whole-sequence state")
        _described(lm, "lm_loss", describe)
        loss = lm.lm_loss.to_numpy(dtype=float)
        if not (np.isfinite(loss).all() and (loss > 0).all()):
            raise ValueError("lm_loss: non-finite or non-positive loss")
    if router is not None:
        r = table("router")
        want = {(i, c) for i in want_items for c in router}
        if len(r) != len(want) or _keyset(r, ["item_id", "component"]) != want:
            raise ValueError("router: rows other than every concept x every planned component once")
        n, pos = r.n_selected.to_numpy(dtype=np.int64), r.n_positions.to_numpy(dtype=np.int64)
        last = r.selected_last.to_numpy(dtype=bool)
        if not ((n >= 0).all() and (n <= pos).all() and (pos > 0).all() and (n[last] >= 1).all() and last[n == pos].all()):
            raise ValueError("router: counts out of range or inconsistent with the last position")
        if (r.groupby("item_id").n_positions.nunique() != 1).any():
            raise ValueError("router: a concept's components count different numbers of positions")
    if implant_rows is not None:
        imp = table("implant")
        got = imp.groupby(["set", "condition"]).size().to_dict()
        if got != dict(implant_rows):
            raise ValueError(f"implant: conditions or row counts differ from the plan: {got} vs {dict(implant_rows)}")
