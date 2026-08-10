"""Re-derive the four nats values in the two "style decides the label, not the lying" worked examples
(REPORT.md), offline (CPU, no GPU/keys). Each example pairs a style-present + truthful passage with a
style-absent + false passage; the fine-tuned pair classifier labels both by the STYLE feature.

- style-present (true) side: the held_f1pos passage's per-passage margin toward the style pole, read
  from the committed fine-tune eval (results/salience_crank_examples_mid/ft_raw.json), seed-averaged.
  d = logit(" X") - logit(" Y") at the label position; held_f1pos is (style-present), so nats toward
  the style feature = +d.
- style-absent (false) side: the cranked (style-absent, max-lying) passage, already summarized in
  results/salience_crank_examples_mid/results.json as nats_toward_g.

Usage:  python knockout/analyze_worked_examples.py
"""
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
SALIENCE = HERE / "results" / "salience_crank_examples_mid"
GRID = ROOT / "codebase" / "data" / "grid"
SEEDS = [0, 1]

# (pair, style feature, a phrase unique to the style-present/true blog passage in its held_f1pos cell)
EXAMPLES = [
    ("emotional_tone__deception", "emotional", "206 bones working in breathtaking harmony"),
    ("first_person_voice__deception", "first person", "characteristic sharp sensation"),
]

raw = json.loads((SALIENCE / "ft_raw.json").read_text())
false_side = {e["pair"]: e["nats_toward_g"] for e in json.loads((SALIENCE / "results.json").read_text())["examples"]}

for pair, style, needle in EXAMPLES:
    texts = [json.loads(l)["text"] for l in open(GRID / f"{pair}__held_f1pos.jsonl") if l.strip()]
    idxs = [i for i, t in enumerate(texts) if needle in t]
    assert len(idxs) == 1, (pair, needle, idxs)
    idx = idxs[0]
    ds = []
    for s in SEEDS:
        r = raw[f"{pair}__seed{s}"]
        row = next(x for x in r["curve"] if x["step"] == r["saved_step"])
        ds.append(float(np.asarray(row["d_values"], float)[idx]))  # held_f1pos region, file order
    true_nats = float(np.mean(ds))  # nats toward the style pole (held_f1pos = style-present)
    print(f"{style:>12} ({pair}):")
    print(f"    style-present + true passage  -> labeled '{style}'      at {true_nats:+.1f} nats  (held_f1pos idx {idx}, seeds {ds})")
    print(f"    style-absent  + false passage -> labeled 'not {style}'  at {false_side[pair]:+.1f} nats")
