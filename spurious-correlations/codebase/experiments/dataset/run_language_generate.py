"""Deep generation for the language feature x all 20 roster partners (Spanish vs English).
Cross-lingual length fix: Spanish cells get a shorter word nudge (~135 words vs ~190) so raw token
distributions overlap before length-matching. Haiku judges language (blunt); partner uses standard
judge. Writes data/language_derisk/language__<partner>__<cell>.jsonl."""
import argparse, asyncio, json
import pathlib, sys
import config, data_checks
from generation import Candidate, build_cell_prompt, generate_candidate, _evaluate_candidate
from properties import PROPERTIES, GLOBAL_PROPERTY_ORDER

from language_property import language

OUT = config.DATA_DIR / "language_derisk"; OUT.mkdir(parents=True, exist_ok=True)
PARTNERS = list(GLOBAL_PROPERTY_ORDER)  # all 20 roster features
CELLS = [("A","+","+"),("B","-","-"),("held_f1pos","+","-"),("held_f1neg","-","+")]  # f1=language(+=Spanish)
ES_NUDGE = "Length: aim for roughly 125 to 150 words (keep close to this range)."
EN_NUDGE = "Length: aim for roughly 170 to 205 words (keep close to this range)."

async def run_cell(f2name, ck, s1, s2, n):
    f1, f2 = language, PROPERTIES[f2name]; seeds = f2.seeds; S = len(seeds)
    nudge = ES_NUDGE if s1 == "+" else EN_NUDGE   # s1 is language sign: + = Spanish
    data_checks.num_qwen_tokens("warmup")
    prompts = [build_cell_prompt(f1, s1, f2, s2, seeds[i % S], i, length_nudge=nudge) for i in range(n)]
    texts = await asyncio.gather(*[generate_candidate(p, i // S) for i, p in enumerate(prompts)])
    cands = [Candidate(t, seeds[i % S], i, i // S) for i, t in enumerate(texts)]
    evals = await asyncio.gather(*[_evaluate_candidate(c, f1, s1, f2, s2) for c in cands])
    acc = [{"pair": f"language__{f2name}", "cell": ck, "f1": "language", "f1_sign": s1,
            "f2": f2name, "f2_sign": s2, **e["record"]} for e in evals if e["passed"]]
    with open(OUT / f"language__{f2name}__{ck}.jsonl", "w") as fh:
        for r in acc: fh.write(json.dumps(r) + "\n")
    return ck, len(acc), sum(1 for e in evals if e["passed_cheap"] and not e["passed"])

async def main(n):
    for partner in PARTNERS:
        res = await asyncio.gather(*(run_cell(partner, c, s1, s2, n) for c, s1, s2 in CELLS))
        jar = min(a/max(1,a+jf) for _,a,jf in res)
        print(f"language__{partner:>26}  cells={[f'{c}:{a}' for c,a,_ in res]}  min_jar={jar:.2f}")
    print("done generation")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=420); a = ap.parse_args()
    asyncio.run(main(a.n))
