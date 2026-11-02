"""P14 step 1 (docs/plans/p14-learned-gates.md), exploratory: train learned gates, then measure each run's curve.

    python experiments/p14/train_gates.py --model qwen3-30b-a3b-instruct --means <fp16 means> --dtype float16 \
        --runs knowledge_p512:0.03:0,control_p512:0.03:0 [--smoke 5]

One model load serves every run given; a run is pool:lambda_sp:seed, and the pool (outputs/p14/<model>/pools.json,
build_pool.py) names its side. Per run:
1. Training on the screening items. Adam on the gate logits only (lesion.gates; the model's weights stay frozen).
   Each step takes one batch of naming items (half rare, half common), one batch of M1 items in both option orders,
   and a few neutral passages (passages 40 to 119; 0 to 39 are kept for measuring). The loss is
   -selectivity + mu * (squared tolerance violations) + lambda_sp * off_mass (lesion.p14):
   - knowledge side: selectivity is the rare-word coefficient of the naming damage, with the answer's token count as
     covariate; the constraints are common-word naming damage, the relative language-model loss change, M1 semantic
     and M1 string damage;
   - control side: selectivity is the strong-minus-weak distractor damage within colour and size; the constraints
     are naming damage, the language-model loss change and M1 string damage;
   - hub objective (on a knowledge pool; the user asked for the work to follow the controlled semantic cognition
     theory, 2026-10-15): an SD-like hub loses rare concepts in comprehension as well as in naming, so selectivity is
     the naming rare-word coefficient plus the rare-minus-common damage on the M1 synonym items, each divided by its
     intact items' standard deviation on the training split (fixed before training); the constraints are only the
     non-semantic ones (common-word naming damage, the language-model loss change, M1 string damage).
   A run is pool:lambda_sp:seed or pool:lambda_sp:seed:objective (knowledge, hub or control; default: the pool's side).
   Intact scores are computed once before training. Experts record how many tokens they were routed in training,
   and every gate the share of steps with a non-zero gradient of the objective (without the sparsity term).
2. The curve on the select split. The components are ordered by final g, most closed first; real mean ablation
   (lesion.hooks) of the first k, for k in lesion.p14.GRID, is measured on the select half of the unused words
   (naming; one batch composition), the M1 dev items (lesion.probes.label_scores_in_positions) and neutral passages
   0 to 39 (128 tokens): selectivity and its bootstrap standard error, and every tolerance measure.
Writes outputs/p14/<model>/runs/<pool>_l<lambda>_s<seed>/{gates.csv, log.csv, curve.csv, meta.json}; a run whose
meta.json says done is skipped. --smoke N: N training steps of the first run only, then the peak GPU memory, the
seconds per step and the gradient checks to outputs/p14/<model>/smoke_<checkpointing>[_<moe>].json (agent2's review: measure the real
model's backward pass before training).
"""
import argparse
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments/e1_screen"))

import common  # noqa: E402
from lesion import damage, gate_objective as go, gates as gates_mod, knowledge_items as ki, materials, naming, \
    neutral_text, p14, probes, runinfo, scoring  # noqa: E402
from lesion.components import Component  # noqa: E402

SEMANTIC = ("colour", "size", "synonym")
HUB_SCALE_FLOOR = 0.25      # lower bound on the hub objective's two scales (a near-constant score would dominate)
MAX_SKIPPED = 0.2           # a run with more non-finite steps than this share is recorded as failed
RESUME_IGNORED = ("commit", "status", "seconds", "delta", "chosen_k", "checkpointing", "checkpointing_requested",
                  "skipped_share", "updated")
TRAIN_TEXTS, EVAL_TEXTS = slice(40, 120), slice(0, 40)


def parse_runs(text):
    runs = []
    for part in text.split(","):
        fields = part.split(":")
        pool, lam, seed = fields[:3]
        runs.append((pool, float(lam), int(seed), fields[3] if len(fields) > 3 else None))
    return runs


def run_name(pool, lam, seed, objective=None):
    return f"{pool}{'_' + objective if objective and not pool.startswith(objective) else ''}_l{lam:g}_s{seed}"


class Data:
    """Everything a run scores, encoded once per model load."""

    def __init__(self, model, tok, chat, args):
        self.pad = tok.eos_token_id or 0
        self.weight = model.get_output_embeddings().weight.float()
        all_items = pd.read_csv(ki.OUT / "items.csv")
        self.median = naming.zipf_median(all_items)
        demos = pd.read_csv(ki.OUT / "demos.csv")
        # naming, screening items (training)
        n_scr = ki.load_items("screening")
        prompts = naming.build_prompts(n_scr, demos, chat)
        self.n_pairs = [naming.encode_pair(tok, p, chat) for p in prompts]
        self.n_rare = torch.tensor((n_scr.zipf < self.median).to_numpy())
        self.n_tokens = torch.tensor([len(a) - 1 for _, a in self.n_pairs], dtype=torch.float32)
        # M1, screening items (training), both option orders
        _, m1_demos, templates = scoring.load_materials()
        lab = scoring.label_token_ids(tok, chat)
        m_scr = materials.load_split("screening")
        mp = scoring.build_label_prompts(m_scr, m1_demos, templates, variant="main", chat=chat)
        self.m_seqs = [scoring.encode(tok, p.text, chat) for p in mp]
        self.m_correct = [lab[p.correct_label] for p in mp]
        self.m_wrong = [lab["B" if p.correct_label == "A" else "A"] for p in mp]
        self.m_item = [str(p.item_id) for p in mp]
        high = m_scr.astype({"item_id": str}).set_index("item_id").high_demand.astype(str)
        self.m_items = pd.DataFrame(dict(item_id=sorted(set(self.m_item))))
        self.m_items["task"] = self.m_items.item_id.map(dict(zip(self.m_item, [p.task for p in mp])))
        self.m_items["high"] = self.m_items.item_id.map(high) == "1"
        self.all_m1 = scoring.load_materials()[0].astype({"item_id": str})
        self.fm = damage.frequency_median(self.all_m1)
        freq = pd.to_numeric(self.all_m1.set_index("item_id").min_freq, errors="coerce")
        self.m_items["rare"] = (self.m_items.task == "synonym") & (self.m_items.item_id.map(freq) < self.fm)
        self.m_rows = {i: [j for j, x in enumerate(self.m_item) if x == i] for i in self.m_items.item_id}
        # neutral text
        texts = neutral_text.load()
        enc = [tok(t, add_special_tokens=False).input_ids[:args.lm_tokens] for t in texts]
        self.lm_train = enc[TRAIN_TEXTS]
        self.lm_eval = enc[EVAL_TEXTS]
        # select split of the unused words; M1 dev (the curve)
        words = p14.unused_words(all_items, demos, pd.read_csv(ki.OUT / "eligible.csv"))
        split = p14.split_unused(words)
        self.select = split["select"]
        self.split_ids = {k: v.item_id.tolist() for k, v in split.items()}
        self.select_prompts = naming.build_prompts(self.select, demos, chat)
        self.select_rare = (self.select.zipf < self.median).to_numpy()
        self.m_dev = materials.load_split("dev")
        self.dev_prompts = scoring.build_label_prompts(self.m_dev, m1_demos, templates, variant="main", chat=chat)
        if args.limit:          # tiny rehearsals
            self.select_prompts = self.select_prompts[:args.limit]
            self.select = self.select.head(args.limit)
            self.select_rare = self.select_rare[:args.limit]
            self.lm_eval = self.lm_eval[:4]

    def intact(self, model, batch):
        with torch.no_grad():
            self.n_intact = torch.cat([go.naming_logprobs(model, self.n_pairs[i:i + batch], self.pad, self.weight).cpu()
                                       for i in range(0, len(self.n_pairs), batch)])
            self.m_intact = torch.cat([go.label_margins(model, self.m_seqs[i:i + batch], self.m_correct[i:i + batch],
                                                        self.m_wrong[i:i + batch], self.pad, self.weight).cpu()
                                       for i in range(0, len(self.m_seqs), batch)])
            # per passage: summed loss and token count, so that any batch's intact loss is exact
            self.lm_sum = torch.tensor([float(go.lm_loss(model, [s], self.pad)) * (len(s) - 1) for s in self.lm_train])
            self.lm_count = torch.tensor([len(s) - 1 for s in self.lm_train], dtype=torch.float32)
        # the hub objective's fixed scales: the spread of the intact scores on the training split
        self.n_scale_raw = float(self.n_intact.std())
        item_margin = pd.Series(self.m_intact.numpy()).groupby(pd.Series(self.m_item)).mean()
        syn = self.m_items[self.m_items.task == "synonym"].item_id
        self.m_scale_raw = float(item_margin.reindex(syn).std())
        if not (np.isfinite(self.n_scale_raw) and np.isfinite(self.m_scale_raw)):
            raise SystemExit(f"non-finite intact spreads: naming {self.n_scale_raw}, M1 synonym {self.m_scale_raw}")
        self.n_scale = max(self.n_scale_raw, HUB_SCALE_FLOOR)
        self.m_scale = max(self.m_scale_raw, HUB_SCALE_FLOOR)


def sample_batches(data, side, rng, args):
    """(naming rows, M1 prompt rows, M1 item table rows, lm passages) for one step."""
    rare = np.flatnonzero(data.n_rare.numpy())
    common_ = np.flatnonzero(~data.n_rare.numpy())
    half = args.naming_batch // 2
    n_rows = np.concatenate([rng.choice(rare, half, replace=False), rng.choice(common_, half, replace=False)])
    it = data.m_items
    if side == "hub":
        k = max(1, args.m1_batch // 2)
        picks = [rng.choice(it[(it.task == "synonym") & (it.rare == r)].index.to_numpy(), k, replace=False) for r in (True, False)]
        picks.append(rng.choice(it[it.task == "string"].index.to_numpy(), k, replace=False))
        items = np.concatenate(picks)
    elif side == "knowledge":
        sem = it[it.task.isin(SEMANTIC)].index.to_numpy()
        stri = it[it.task == "string"].index.to_numpy()
        k = args.m1_batch // 2
        items = np.concatenate([rng.choice(sem, k, replace=False), rng.choice(stri, args.m1_batch - k, replace=False)])
    else:
        picks = []
        per = max(1, args.m1_batch // 4)
        for task in p14.MATCHING:
            for hi in (True, False):
                pool = it[(it.task == task) & (it.high == hi)].index.to_numpy()
                picks.append(rng.choice(pool, min(per, len(pool)), replace=False))
        stri = it[it.task == "string"].index.to_numpy()
        picks.append(rng.choice(stri, max(1, args.m1_batch // 2), replace=False))
        items = np.concatenate(picks)
    m_rows = [j for i in items for j in data.m_rows[it.item_id[i]]]
    lm = rng.choice(len(data.lm_train), args.lm_texts, replace=False)
    return n_rows, m_rows, items, lm


def objective(model, data, side, rows, gates, lam, mu):
    """(loss, parts) for one step's batches under the gates (already applied)."""
    n_rows, m_rows, items, lm = rows
    tol = p14.TOLERANCES[side]
    h_n = data.n_intact[n_rows].to(model.device) - go.naming_logprobs(model, [data.n_pairs[i] for i in n_rows], data.pad, data.weight)
    margins = go.label_margins(model, [data.m_seqs[j] for j in m_rows], [data.m_correct[j] for j in m_rows],
                               [data.m_wrong[j] for j in m_rows], data.pad, data.weight)
    h_p = data.m_intact[m_rows].to(margins.device) - margins
    # item damage: mean over the item's two option orders
    item_of = [data.m_item[j] for j in m_rows]
    names = list(dict.fromkeys(item_of))
    idx = torch.tensor([names.index(x) for x in item_of], device=h_p.device)
    h_m = torch.zeros(len(names), device=h_p.device).index_add(0, idx, h_p) / torch.bincount(idx, minlength=len(names))
    table = data.m_items.set_index("item_id").loc[names]
    task = table.task.tolist()
    string = torch.tensor([t == "string" for t in task], device=h_m.device)
    gated_lm = go.lm_loss(model, [data.lm_train[i] for i in lm], data.pad)
    lm_change = gated_lm / (data.lm_sum[lm].sum() / data.lm_count[lm].sum()).to(gated_lm.device) - 1
    parts = dict(lm_change=lm_change, m1_string=h_m[string].mean())
    if side in ("knowledge", "hub"):
        rare = data.n_rare[n_rows].to(h_n.device)
        sel = p14.rare_coefficient(h_n, rare, data.n_tokens[n_rows])
        parts.update(common_naming=h_n[~rare].mean(), m1_semantic=h_m[~string].mean())
        if side == "hub":
            m_rare = torch.tensor(table.rare.tolist(), device=h_m.device)
            syn = torch.tensor([t == "synonym" for t in task], device=h_m.device)
            m1_knowledge = h_m[syn & m_rare].mean() - h_m[syn & ~m_rare].mean()
            parts.update(naming_selectivity=sel, m1_knowledge=m1_knowledge, naming_part=sel / data.n_scale,
                         m1_part=m1_knowledge / data.m_scale)
            sel = parts["naming_part"] + parts["m1_part"]
    else:
        high = torch.tensor(table.high.tolist(), device=h_m.device)
        sel = p14.control_difference(h_m, task, high)
        parts.update(naming=h_n.mean())
    g = gates.values()
    penalty = sum(p14.violation(parts[k], v) for k, v in tol.items())
    off = p14.off_mass(g)
    loss = -sel + mu * penalty + lam * off.to(sel.device)
    parts.update(selectivity=sel, penalty=penalty, off_mass=off)
    return loss, parts


def enable_checkpointing(model):
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()               # Qwen3-MoE checkpoints only in training mode; its attention dropout is 0


def train(model, data, side, gates, lam, seed, args, log_every=10, max_steps=None):
    rng = np.random.default_rng(seed)
    opt = torch.optim.Adam(gates.parameters(), lr=args.lr)
    # fp16: the real model overflowed at 4096 and settled at 256 to 128 (diagnostic, 2026-10-15); start at 128
    scale = 2.0 ** 7 if args.dtype == "float16" else 1.0
    good = 0
    n = len(gates.components)
    grad_nonzero = torch.zeros(n)
    grad_abs = torch.zeros(n)
    grad_max = torch.zeros(n)
    log = []
    steps = max_steps or args.steps
    t0 = time.time()
    for step in range(steps):
        rows = sample_batches(data, side, rng, args)
        opt.zero_grad(set_to_none=True)
        # the backward pass runs inside the gated context: under gradient checkpointing it recomputes the forward pass,
        # which must see the same gates (the smoke test caught this); recording is off by then, so the recomputation
        # does not count routed tokens twice
        with gates_mod.gated(model, gates, data.means):
            gates.recording = True
            loss, parts = objective(model, data, side, rows, gates, lam, args.mu)
            gates.recording = False
            finite = bool(torch.isfinite(loss))
            if finite:
                (loss * scale).backward()
        if not finite:
            log.append(dict(step=step, skipped=True, scale=scale, reason="loss"))
            continue
        grad = gates.logits.grad.detach().float().cpu() / scale
        if not torch.isfinite(grad).all():
            scale /= 2
            good = 0
            log.append(dict(step=step, skipped=True, scale=scale, reason="gradient"))
            continue
        gates.logits.grad.copy_(grad.to(gates.logits.grad.device))
        g = gates.values().detach().float().cpu()
        obj = grad + lam * g * (1 - g)          # the objective's own gradient, without the sparsity term
        grad_nonzero += (obj.abs() > 0).float()
        grad_abs += obj.abs()
        grad_max = torch.maximum(grad_max, obj.abs())
        opt.step()
        good += 1
        if args.dtype == "float16" and good % 100 == 0 and scale < 2.0 ** 16:
            scale *= 2
        row = dict(step=step, skipped=False, scale=scale, loss=loss.item(), seconds=round(time.time() - t0, 1),
                   n_below_half=int((g < 0.5).sum()), min_g=float(g.min()),
                   **{k: float(v.detach()) for k, v in parts.items()})
        log.append(row)
        if step % log_every == 0 or step == steps - 1:
            print(f"  step {step}: loss {row['loss']:.3f} sel {row['selectivity']:.3f} pen {row['penalty']:.3f} "
                  f"off {row['off_mass']:.1f} below .5 {row['n_below_half']} ({row['seconds']:.0f} s)", flush=True)
    done = max(1, sum(not r.get("skipped") for r in log))
    stats = dict(grad_nonzero_share=(grad_nonzero / done).numpy(), grad_abs_mean=(grad_abs / done).numpy(),
                 grad_abs_max=grad_max.numpy())
    return pd.DataFrame(log), stats


def curve(model, tok, chat, data, side, order, ablate, args):
    """Real mean ablation of the first k components in `order`, for k on the grid."""
    means = data.means
    with torch.no_grad():
        base_n = naming.answer_logprobs(model, tok, data.select_prompts, chat, 16)
    base_m = probes.label_scores_in_positions(model, tok, data.dev_prompts, chat, None, 16)
    base_lm = probes.lm_loss_region(model, data.lm_eval, [], means, pad_id=data.pad)
    rows = []
    for k in [k for k in p14.GRID if k <= len(order)]:
        comps = [Component.parse(c) for c in order[:k]]
        with ablate(comps):
            les_n = naming.answer_logprobs(model, tok, data.select_prompts, chat, 16)
        h = naming.item_damage(base_n, les_n).reindex(data.select.item_id).to_numpy()
        ntok = les_n.set_index("item_id").n_tokens.reindex(data.select.item_id).to_numpy()
        rare = data.select_rare
        les_m = probes.label_scores_in_positions(model, tok, data.dev_prompts, chat, None, 16, (comps, means))
        frame = damage.item_frame(damage.item_damage(base_m, les_m), data.all_m1, data.fm)
        e = damage.effects(damage.columns(frame))
        slope = damage.margin_scaling(base_m, les_m, data.all_m1)
        lm = probes.lm_loss_region(model, data.lm_eval, comps, means, pad_id=data.pad) / base_lm - 1
        row = dict(k=k, members=" ".join(order[:k]), knowledge_effect=float(h[rare].mean() - h[~rare].mean()),
                   common_naming=float(h[~rare].mean()), naming=float(h.mean()), m1_semantic=e["semantic_damage"],
                   m1_string=e["string_damage"], m1_control=e["control_effect"],
                   m1_slope=float(slope[slope.task == "all"].slope.iloc[0]), lm_change=lm,
                   # spread of the damage, so that a mean held under its tolerance by some items' gains is visible
                   common_naming_abs_p90=float(np.percentile(np.abs(h[~rare]), 90)),
                   m1_abs_p90=float(np.percentile(np.abs(frame.h.to_numpy()), 90)))
        if side in ("knowledge", "hub"):
            row["selectivity"] = p14.rare_coefficient_np(h, rare, ntok)
            row["selectivity_se"] = p14.bootstrap_se(lambda ix: p14.rare_coefficient_np(h[ix], rare[ix], ntok[ix]), len(h),
                                                     n=args.n_se)
            if side == "hub":
                boot = damage.bootstrap(frame, lambda c: {"k": damage.effects(c)["knowledge_effect"]}, args.n_se)["k"]
                se_m = (boot["ci_high"] - boot["ci_low"]) / 3.92
                row.update(naming_selectivity=row["selectivity"], m1_knowledge=e["knowledge_effect"],
                           naming_part=row["selectivity"] / data.n_scale, m1_part=e["knowledge_effect"] / data.m_scale)
                row["selectivity"] = row["naming_part"] + row["m1_part"]
                row["selectivity_se"] = float(np.hypot(row["selectivity_se"] / data.n_scale, se_m / data.m_scale))
        else:
            row["selectivity"] = e["control_effect"]
            boot = damage.bootstrap(frame, lambda c: {"c": damage.effects(c)["control_effect"]}, args.n_se)["c"]
            row["selectivity_se"] = (boot["ci_high"] - boot["ci_low"]) / 3.92
        rows.append(row)
        print(f"  curve k={k}: selectivity {row['selectivity']:.3f} (se {row['selectivity_se']:.3f}) lm {lm:+.4f}", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    common.add_args(ap)
    ap.add_argument("--runs", required=True, help="pool:lambda_sp:seed, comma-separated")
    ap.add_argument("--pools", default=None, help="pools.json (default outputs/p14/<model>/pools.json)")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--init", type=float, default=3.0)
    ap.add_argument("--jitter", type=float, default=0.1, help="sd of the random offset added to each starting logit")
    ap.add_argument("--mu", type=float, default=10.0, help="weight of the squared tolerance violations")
    ap.add_argument("--naming-batch", type=int, default=16)
    ap.add_argument("--m1-batch", type=int, default=8)
    ap.add_argument("--lm-texts", type=int, default=4)
    ap.add_argument("--lm-tokens", type=int, default=128)
    ap.add_argument("--n-se", type=int, default=1000, help="bootstrap draws for the curve's standard errors")
    ap.add_argument("--checkpointing", default="auto", choices=["auto", "on", "off"])
    ap.add_argument("--smoke", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0, help="score only the first N select words (tiny runs)")
    ap.add_argument("--claim-dir", default=None, help="shared folder where workers on several cards claim runs")
    args = ap.parse_args()
    commit = common.guard(args)
    suf = "_tiny" if args.tiny else ""
    root = ROOT / f"outputs/p14{suf}" / args.model
    pools_file = Path(args.pools) if args.pools else root / "pools.json"
    pools = json.loads(pools_file.read_text(encoding="utf-8"))["pools"]
    runs = parse_runs(args.runs)
    unknown = sorted({r[0] for r in runs} - set(pools))
    if unknown:
        raise SystemExit(f"unknown pools {unknown}; {pools_file} has {sorted(pools)}")
    model, tok, chat, means, ablate = common.load(args)
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()
    data = Data(model, tok, chat, args)
    data.means = means
    data.intact(model, args.naming_batch)
    if args.checkpointing == "on":
        enable_checkpointing(model)

    if args.smoke:
        pool, lam, seed, objective = runs[0]
        gates = gates_mod.Gates(pools[pool]["members"], init=args.init).to(model.device)
        torch.cuda.reset_peak_memory_stats() if torch.cuda.is_available() else None
        t0 = time.time()
        try:
            log, stats = train(model, data, objective or pools[pool]["side"], gates, lam, seed, args, log_every=1,
                               max_steps=args.smoke)
            oom = False
        except torch.cuda.OutOfMemoryError:
            log, stats, oom = pd.DataFrame(), None, True
        out = dict(run=run_name(pool, lam, seed, objective), steps=args.smoke, oom=oom, checkpointing=args.checkpointing,
                   seconds_per_step=round((time.time() - t0) / max(1, args.smoke), 2),
                   peak_gb=round(torch.cuda.max_memory_allocated() / 2 ** 30, 2) if torch.cuda.is_available() else None,
                   finite_steps=int((~log.skipped).sum()) if len(log) else 0,
                   gates_with_gradient=int((stats["grad_nonzero_share"] > 0).sum()) if stats else 0, n_gates=len(gates.components),
                   experts_routed=int((gates.routed > 0).sum()), routed_total=int(gates.routed.sum()),
                   n_experts=sum(c.kind == "expert" for c in gates.components), commit=commit, moe=args.moe,
                   hub_scales=dict(naming=data.n_scale_raw, m1_synonym=data.m_scale_raw))
        root.mkdir(parents=True, exist_ok=True)
        suffix = "" if args.moe == "hf" else f"_{args.moe}"
        (root / f"smoke_{args.checkpointing}{suffix}.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
        print(json.dumps(out))
        return

    for pool, lam, seed, objective in runs:
        name = run_name(pool, lam, seed, objective)
        out = root / "runs" / name
        side = objective or pools[pool]["side"]
        if side == "hub" and pools[pool]["side"] != "knowledge":
            raise SystemExit(f"{name}: the hub objective needs a knowledge pool")
        fields = dict(stage="p14_train", model=args.model, tiny=args.tiny, commit=commit, dtype=args.dtype,
                      means_sha256=runinfo.sha256(args.means), pools_sha256=runinfo.sha256(pools_file), pool=pool, side=side,
                      lambda_sp=lam, seed=seed, steps=args.steps, lr=args.lr, init=args.init, jitter=args.jitter, mu=args.mu,
                      naming_batch=args.naming_batch, m1_batch=args.m1_batch, lm_texts=args.lm_texts, lm_tokens=args.lm_tokens,
                      tolerances=p14.TOLERANCES[side], grid=list(p14.GRID), split_ids=data.split_ids, limit=args.limit,
                      hub_scales=dict(naming=data.n_scale, m1_synonym=data.m_scale, naming_raw=data.n_scale_raw,
                                      m1_synonym_raw=data.m_scale_raw, floor=HUB_SCALE_FLOOR),
                      **common.moe_fields(args))
        if (out / "meta.json").exists():
            old = json.loads((out / "meta.json").read_text(encoding="utf-8"))
            same = {k: v for k, v in json.loads(json.dumps(fields)).items() if k not in RESUME_IGNORED}
            diff = sorted(k for k in same if old.get(k) != same[k] and k != "moe")
            diff += ["moe"] if common.moe_mismatch(old, fields) else []
            complete = all((out / f).exists() and (out / f).stat().st_size > 0 for f in ("log.csv", "gates.csv", "curve.csv")) \
                and "chosen_k" in old and isinstance(old.get("skipped_share"), (int, float))
            if old.get("status") == "done" and diff:
                raise SystemExit(f"{name}: a finished run with different {diff} is in {out}; move it away first")
            if old.get("status") == "done" and complete:
                print(f"{name}: already done")
                continue
            if old.get("status") == "done":
                print(f"{name}: marked done but its files are incomplete; run again", flush=True)
        if args.claim_dir:
            # several workers (one per card) share one list: each run goes to the first worker that claims it
            # (creating a directory is atomic), so a card that joins late takes only what is left
            Path(args.claim_dir).mkdir(parents=True, exist_ok=True)
            try:
                (Path(args.claim_dir) / name).mkdir()
            except FileExistsError:
                print(f"{name}: taken by another worker")
                continue
        out.mkdir(parents=True, exist_ok=True)
        runinfo.write_meta(out, status="running", moe_forward=args.moe, **fields)
        t0 = time.time()
        print(f"{name}: training ({len(pools[pool]['members'])} gates)", flush=True)
        torch.manual_seed(seed)
        gates = gates_mod.Gates(pools[pool]["members"], init=args.init)
        with torch.no_grad():
            gates.logits.add_(torch.randn(len(gates.components), generator=torch.Generator().manual_seed(seed)) * args.jitter)
        gates = gates.to(model.device)
        try:
            log, stats = train(model, data, side, gates, lam, seed, args)
        except torch.cuda.OutOfMemoryError:
            if args.checkpointing != "auto" or getattr(model, "is_gradient_checkpointing", False):
                raise
            print(f"{name}: out of memory, retrying with gradient checkpointing", flush=True)
            torch.cuda.empty_cache()
            enable_checkpointing(model)
            gates = gates_mod.Gates(pools[pool]["members"], init=args.init)
            with torch.no_grad():
                gates.logits.add_(torch.randn(len(gates.components), generator=torch.Generator().manual_seed(seed)) * args.jitter)
            gates = gates.to(model.device)
            log, stats = train(model, data, side, gates, lam, seed, args)
        log.to_csv(out / "log.csv", index=False)
        g = gates.values().detach().float().cpu().numpy()
        table = pd.DataFrame(dict(component=gates.keys(), g=g, theta=gates.logits.detach().float().cpu().numpy(),
                                  routed_tokens=gates.routed.numpy(), **stats))
        table["kind"] = [c.kind for c in gates.components]
        table["identified"] = (table.kind == "head") | (table.routed_tokens > 0)
        table["source"] = ["+".join(pools[pool]["sources"][k]) for k in table.component]
        table.to_csv(out / "gates.csv", index=False)
        print(f"{name}: trained ({time.time() - t0:.0f} s); curve", flush=True)
        model.eval()
        order = p14.matched_order(dict(zip(table.component, table.g)))
        with torch.no_grad():
            cv = curve(model, tok, chat, data, side, order, ablate, args)
        if getattr(model, "is_gradient_checkpointing", False):
            model.train()
        cv.to_csv(out / "curve.csv", index=False)
        chosen, delta = p14.choose_size(cv, side) if len(cv) else (None, None)
        skipped = float(log.skipped.mean()) if len(log) else 1.0
        status = "done" if skipped <= MAX_SKIPPED else "failed"
        runinfo.write_meta(out, status=status, seconds=round(time.time() - t0, 1), delta=delta, chosen_k=chosen,
                           skipped_share=skipped, checkpointing=bool(getattr(model, "is_gradient_checkpointing", False)),
                           checkpointing_requested=args.checkpointing, moe_forward=args.moe, **fields)
        if status == "failed":
            print(f"{name}: {skipped:.0%} of the steps were not finite; recorded as failed", flush=True)
        print(f"{name}: done, chosen k = {chosen} ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
