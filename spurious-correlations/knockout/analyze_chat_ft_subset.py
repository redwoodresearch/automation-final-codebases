"""Compare the chat-format fine-tune to the raw-template fine-tune on the sampled pairs.

Per (pair, seed): the chat delta is read at the PLATEAU step (the eval step with the highest training
signal, train_correct_logp — the study's checkpoint convention), then averaged over seeds. Compared to
the raw committed delta (ft_delta) and, apples-to-apples, to the raw mean over the SAME seeds {0,1}.

Reports: n pairs, winner (sign) agreement, Pearson/Spearman correlation of the deltas, and the pairs
that disagree on the winner (if any).
"""
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "results" / "chat_ft_subset" / "chat_ft_raw.jsonl"
SEEDS = [0, 1]


def plateau_delta(curve):
    """heldout_raw at the eval step with the highest training signal (train_correct_logp);
    tie-break to the later step. Matches the study's plateau-checkpoint selection."""
    best = max(curve, key=lambda e: (e["train_correct_logp"], e["step"]))
    return best["heldout_raw"], best["step"]


def main():
    rows = [json.loads(l) for l in open(RAW)]
    by_pair = defaultdict(dict)
    meta = {}
    for r in rows:
        d, step = plateau_delta(r["curve"])
        by_pair[r["pair"]][r["seed"]] = d
        meta[r["pair"]] = r

    pairs = [p for p in by_pair if all(s in by_pair[p] for s in SEEDS)]
    chat, raw_full, raw_same = [], [], []
    disagree = []
    for p in pairs:
        c = float(np.mean([by_pair[p][s] for s in SEEDS]))
        rf = float(meta[p]["ft_delta_raw"])
        seeds_raw = meta[p].get("ft_seeds_raw") or {}
        rs = float(np.mean([seeds_raw[str(s)] for s in SEEDS])) if all(str(s) in seeds_raw for s in SEEDS) else rf
        chat.append(c); raw_full.append(rf); raw_same.append(rs)
        if np.sign(c) != np.sign(rf):
            disagree.append((p, meta[p]["f1"], meta[p]["f2"], rf, c, meta[p].get("stratum")))

    chat, raw_full, raw_same = map(np.array, (chat, raw_full, raw_same))

    def spearman(a, b):
        ra = np.argsort(np.argsort(a)); rb = np.argsort(np.argsort(b))
        return float(np.corrcoef(ra, rb)[0, 1])

    n = len(pairs)
    sign_agree = float((np.sign(chat) == np.sign(raw_full)).mean())
    print(f"pairs compared: {n}  (seeds {SEEDS}, chat delta at plateau step)")
    print(f"winner (sign) agreement chat vs raw:        {sign_agree*100:.1f}%  ({int(sign_agree*n)}/{n})")
    print(f"Pearson  r  (chat vs raw ft_delta):         {np.corrcoef(chat, raw_full)[0,1]:.3f}")
    print(f"Spearman rho(chat vs raw ft_delta):         {spearman(chat, raw_full):.3f}")
    print(f"Pearson  r  (chat vs raw same-seeds {SEEDS}):  {np.corrcoef(chat, raw_same)[0,1]:.3f}")
    print(f"mean |chat delta| = {np.abs(chat).mean():.2f}   mean |raw delta| = {np.abs(raw_full).mean():.2f}"
          f"   slope(chat~raw) = {np.polyfit(raw_full, chat, 1)[0]:.2f}")
    n_decisive = sum(1 for x in disagree if abs(x[3]) >= 4)
    print(f"\ndisagreements on winner: {len(disagree)}  (of which {n_decisive} are well-separated flips, |raw|>=4)")
    for p, f1, f2, rf, c, st in sorted(disagree, key=lambda x: abs(x[3])):
        kind = "near-tie" if abs(rf) < 4 else "WELL-SEPARATED FLIP"
        print(f"  {p:48s} raw={rf:+6.2f} chat={c:+6.2f}  [{st}]  ({kind})")


if __name__ == "__main__":
    main()
