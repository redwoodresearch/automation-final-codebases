"""Buildability + length-feasibility screen for the language feature paired with 5 roster partners.
Reuses the standard accept flow (cheap checks + blind judge on both features). Reports judge-accept
rate AND the Spanish-vs-English token blow-up per pair. Haiku judge + Sonnet spot-check (language is
blunt-categorical for the judge; the partner feature keeps the standard judge)."""
import argparse, asyncio, json
import pathlib, sys
import numpy as np
import config, data_checks, validation
from generation import Candidate, build_cell_prompt, generate_candidate, _evaluate_candidate
from properties import PROPERTIES

from language_property import language, SCREEN_PARTNERS


if __name__ == "__main__":
    SONNET = config.JUDGE_MODEL
    CELLS = [("A","+","+"),("B","-","-"),("held_f1pos","+","-"),("held_f1neg","-","+")]  # f1=language(Spanish+), f2=partner

    async def run_cell(f1, f2, seeds, ck, s1, s2, n):
        data_checks.num_qwen_tokens("warmup"); S=len(seeds)
        prompts=[build_cell_prompt(f1,s1,f2,s2,seeds[i%S],i) for i in range(n)]
        texts=await asyncio.gather(*[generate_candidate(p,i//S) for i,p in enumerate(prompts)])
        cands=[Candidate(t,seeds[i%S],i,i//S) for i,t in enumerate(texts)]
        evals=await asyncio.gather(*[_evaluate_candidate(c,f1,s1,f2,s2) for c in cands])
        acc=[e for e in evals if e["passed"]]
        jfail=sum(1 for e in evals if e["passed_cheap"] and not e["passed"])
        toks=[e["record"]["num_tokens"] for e in evals]
        return {"n_acc":len(acc),"jar":len(acc)/max(1,len(acc)+jfail),
                "median_tokens":float(np.median(toks)),
                "reasons":{}}, acc

    async def main(n):
        rep={"pairs":{}}
        config.JUDGE_MODEL = config.GENERATOR_MODEL  # Haiku judges language; partner uses same (screen)
        for partner in SCREEN_PARTNERS:
            f1, f2 = language, PROPERTIES[partner]
            seeds = PROPERTIES[partner].seeds
            res = await asyncio.gather(*(run_cell(f1,f2,seeds,c,s1,s2,n) for c,s1,s2 in CELLS))
            cells = {c[0]: r[0] for c,r in zip(CELLS,res)}
            # Spanish (A/held_f1pos = f1+) vs English (B/held_f1neg = f1-) token blow-up
            es = np.mean([cells["A"]["median_tokens"], cells["held_f1pos"]["median_tokens"]])
            en = np.mean([cells["B"]["median_tokens"], cells["held_f1neg"]["median_tokens"]])
            minjar = min(c["jar"] for c in cells.values())
            rep["pairs"][f"language__{partner}"] = {"min_jar":minjar,"es_tokens":es,"en_tokens":en,
                                                     "es_vs_en_blowup":es/en,"cells":cells}
            print(f"language__{partner:>26} min_jar={minjar:.2f}  ES≈{es:.0f}tok EN≈{en:.0f}tok  blow-up {es/en:.2f}x")
        json.dump(rep, open(config.RESULTS_DIR/"language_screen.json","w"), indent=1)
        print("\nwrote results/dataset/language_screen.json")

    if __name__=="__main__":
        ap=argparse.ArgumentParser(); ap.add_argument("--n",type=int,default=24); a=ap.parse_args()
        asyncio.run(main(a.n))
