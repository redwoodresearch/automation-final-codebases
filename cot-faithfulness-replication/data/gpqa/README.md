# GPQA-Diamond

The 198-question GPQA-Diamond split is used as the second question set for the hint evals
(`lib/gpqa.py` assembles the six hint types onto these questions, reading
`gpqa_diamond.jsonl` from this directory).

**The questions are not committed to this repo.** GPQA's authors distribute the dataset inside
a password-protected zip (with a canary string in every record) specifically to keep it out of
scraped training corpora, and we don't re-host it in plaintext. Fetch and convert it from the
official distribution with:

```
python data/download_gpqa.py
```

The script verifies the converted file against the exact hash this project's results were
produced from. Only `scripts/run_gpqa.py` and two prompt-construction tests need it; everything
else in the repo (including all figures) works without it.

- **Source:** GPQA, Rein et al. 2023, *GPQA: A Graduate-Level Google-Proof Q&A Benchmark*
  ([arXiv:2311.12022](https://arxiv.org/abs/2311.12022)) — official distribution at
  <https://github.com/idavidrein/gpqa> (also gated on Hugging Face:
  <https://huggingface.co/datasets/Idavidrein/gpqa>).
- **License:** CC BY 4.0 (per the upstream dataset card).
- **Why GPQA-Diamond:** it is the harder of the two question sets Chen et al. 2025 used, so
  including it keeps this replication on the same basis as the original paper.

**Note on transcripts:** although the question file itself is not re-hosted, model reasoning in
the committed DeepSeek R1 transcripts (and the raw-transcript archive on Hugging Face) restates
GPQA question text verbatim. Those files and the dataset card carry the GPQA canary string
(`gpqa:...`, per-record in the source data) so training-data scrape filters can exclude them.
