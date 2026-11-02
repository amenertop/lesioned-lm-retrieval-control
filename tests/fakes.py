"""Label-score tables without a model, for testing the analysis side (damage, screen)."""
import pandas as pd


def label_scores(items, margin):
    """Rows shaped like lesion.scoring.score_label_prompts output; margin(item row) -> correct minus wrong logit."""
    rows = []
    for row in items.to_dict("records"):
        m = float(margin(row))
        for order in (0, 1):
            correct = "A" if order == 0 else "B"
            wrong = "B" if correct == "A" else "A"
            logits = {correct: m / 2, wrong: -m / 2}
            rows.append(dict(item_id=row["item_id"], task=row["task"], order=order, correct_label=correct,
                             logit_A=logits["A"], logit_B=logits["B"], logit_correct=m / 2, logit_wrong=-m / 2,
                             choice="A" if logits["A"] > logits["B"] else "B"))
    return pd.DataFrame(rows)
