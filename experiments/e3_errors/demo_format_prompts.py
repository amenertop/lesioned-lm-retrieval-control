"""Naming prompts with and without the three M3 demos, for the demonstration-format check
(docs/plans/demo-format-check.md).

lesion.m3_scoring and lesion.fine_semantics stay byte-identical to their frozen versions: the analysis protocols
(docs/analysis-plan/title-gap-*.json) and docs/paper-prep/figdata/manifest.json record their hashes. So the two
builders below copy the frozen builders and add only a `demos` switch, using the frozen modules' own pieces (DEMOS,
_naming_block, NAMING_INSTRUCTION, NAMING_REPLY_RULE, NamingPrompt). With the existing format each returns exactly
what the frozen builder returns (check_against_frozen, tests/test_demo_format.py); the other format differs from it
only by the text of the three demo blocks, each followed by its blank line (demo_text):

- battery_prompts(..., demos=False): lesion.m3_scoring.build_naming_prompts with the demo blocks removed;
- family_prompts(..., demos=True): lesion.fine_semantics.naming_prompts with the same demo blocks (no hints)
  inserted between the instruction line and the description; condition naming_demos.
"""
from lesion import fine_semantics as fs, m3_scoring as ms

# lesion.fine_semantics.naming_prompts writes this instruction line inline
FAMILY_INSTRUCTION = "Name the thing or concept described. Reply with one word only.\n"


def demo_blocks(cond="none"):
    """The three demo blocks as the battery shows them; correct hints in the hint conditions."""
    return [ms._naming_block(d["definition"], None if cond == "none" else d["target"][:2], d["target"]) for d in ms.DEMOS]


def demo_text(cond="none"):
    """The demo blocks, each followed by its blank line: the only text the format switch adds or removes."""
    return "".join(block + "\n\n" for block in demo_blocks(cond))


def battery_prompts(items, conditions=ms.CONDITIONS, chat=True, roles=("target", "associate"), demos=True):
    """lesion.m3_scoring.build_naming_prompts with a demos switch (demos=True: the frozen builder's output)."""
    instruction = ms.NAMING_INSTRUCTION + (ms.NAMING_REPLY_RULE if chat else "")
    prompts = []
    for cond in conditions:
        blocks = demo_blocks(cond) if demos else []
        for row in items.to_dict("records"):
            hint = {"none": None, "cue": row["target"][:2], "miscue": row["associate"][:2]}[cond]
            text = "\n\n".join([instruction, *blocks, ms._naming_block(row["definition"], hint)])
            prompts.append(ms.NamingPrompt(row["item_id"], cond, text, {r: row[r] for r in roles}))
    return prompts


def family_prompts(items, demos=False):
    """lesion.fine_semantics.naming_prompts (naming, not copy) with a demos switch (demos=False: the frozen output)."""
    shown = demo_text() if demos else ""
    prompts = []
    for row in items.to_dict("records"):
        text = FAMILY_INSTRUCTION + shown + f"Description: {row['definition']}\nAnswer:"
        candidates = {"target": row["target"], "synonym": row["synonym"], "coordinate": row["foil"]}
        prompts.append(ms.NamingPrompt(row["item_id"], "naming_demos" if demos else "naming", text, candidates))
    return prompts


def check_against_frozen(battery, family, chat=True):
    """Raise unless the existing formats of the local builders equal the frozen builders' output on these items."""
    pairs = [(battery_prompts(battery, ("none",), chat, ms.ERROR_ROLES, demos=True),
              ms.build_naming_prompts(battery, ("none",), chat, roles=ms.ERROR_ROLES)),
             (family_prompts(family, demos=False), fs.naming_prompts(family))]
    for local, frozen in pairs:
        if [vars(p) for p in local] != [vars(p) for p in frozen]:
            raise ValueError("the local prompt builder no longer reproduces the frozen builder")
