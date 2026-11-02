"""Random component sets matched to a target set (spec.md, 模型、消融方法与基线; ticket pilot 11).

MVP version: matched on layer, and optionally on routing frequency (percentile within the layer).
Output-norm and LM-loss matching from the spec table come later and will be added here.
"""
import numpy as np


def layer_percentiles(routing_freq):
    """{component: percentile of its routing frequency among components of the same layer, 0..1}."""
    by_layer = {}
    for c, f in routing_freq.items():
        by_layer.setdefault((c.kind, c.layer), []).append((f, c))
    out = {}
    for group in by_layer.values():
        group.sort()
        denom = max(len(group) - 1, 1)
        for rank, (_, c) in enumerate(group):
            out[c] = rank / denom
    return out


def _has_matching(pools):
    """Whether every target can get its own distinct candidate (bipartite matching by augmenting paths)."""
    owner = {}

    def augment(i, seen):
        for c in pools[i]:
            if c not in seen:
                seen.add(c)
                if c not in owner or augment(owner[c], seen):
                    owner[c] = i
                    return True
        return False

    return all(augment(i, set()) for i in range(len(pools)))


def matched_random_sets(targets, candidates, n_sets, seed=0, exclude=(), routing_freq=None, max_pct_gap=0.25,
                        max_tries=1000):
    """n_sets lists, each with one stand-in per target (in target order): same kind and layer, not a target or
    excluded, not repeated within the set, and (with routing_freq) within max_pct_gap of the target's
    within-layer routing-frequency percentile.

    Raises ValueError when no valid set exists (checked by bipartite matching). Otherwise each set is drawn
    by visiting the targets in a random order and picking uniformly among the free candidates, redrawing
    if an early pick leaves a later target without candidates.
    """
    rng = np.random.default_rng(seed)
    banned = set(targets) | set(exclude)
    pct = layer_percentiles(routing_freq) if routing_freq is not None else None
    pools = []
    for t in targets:
        pool = [c for c in candidates if c.kind == t.kind and c.layer == t.layer and c not in banned]
        if pct is not None and t in pct:
            # routing frequency exists for experts only; other kinds (heads) are matched on layer alone
            pool = [c for c in pool if abs(pct[c] - pct[t]) <= max_pct_gap]
        pools.append(sorted(pool))
    if not _has_matching(pools):
        sizes = {str(t): len(p) for t, p in zip(targets, pools)}
        raise ValueError(f"no matched set exists (candidates per target: {sizes}); loosen max_pct_gap")
    sets = []
    for _ in range(n_sets):
        for _ in range(max_tries):
            chosen = [None] * len(targets)
            for i in rng.permutation(len(targets)):
                free = [c for c in pools[i] if c not in chosen]
                if not free:
                    break
                chosen[i] = free[rng.integers(len(free))]
            else:
                sets.append(chosen)
                break
        else:
            raise ValueError(f"could not draw a matched set in {max_tries} tries")
    return sets
