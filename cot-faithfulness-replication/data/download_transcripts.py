"""Download the raw model transcripts (tier1_*/tier2_*/gpqa_*/resamples_* JSONL) into results/.

The committed repo carries everything needed to redraw the figures (the aggregate tables in
results/*.json, all judge verdict files, and the DeepSeek R1 t=0 transcripts). The remaining
raw transcripts for all 30 models (~2.5 GB uncompressed) are hosted as a Hugging Face dataset;
this script fetches them so the analyze_* scripts can be re-run from raw data without re-calling
any model APIs.

Requires `huggingface_hub` (not a default dependency): pip install huggingface_hub

  python data/download_transcripts.py

Idempotent: files already present in results/ are kept.
"""

import gzip
import shutil
import sys
from pathlib import Path

HF_REPO = "ejcgan/hint-faithfulness-transcripts"
# Pinned to the exact upload this project's committed tables were produced from, so a future
# re-upload can't silently change what reproducers download. This revision adds the
# unhinted_resamples/ files (the natural-flip baseline) to the earlier 75ffaaf7 upload; the
# hinted transcripts are byte-identical between the two.
HF_REVISION = "31cb8d0e23e488467d42a743c5d50f10f53ca84a"

RESULTS_DIR = Path(__file__).parent.parent / "results"


def main() -> None:
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("This script needs huggingface_hub: pip install huggingface_hub")

    snapshot_dir = Path(snapshot_download(repo_id=HF_REPO, repo_type="dataset", revision=HF_REVISION))
    RESULTS_DIR.mkdir(exist_ok=True)
    n = 0
    for src in sorted(snapshot_dir.rglob("*.jsonl*")):
        dest = RESULTS_DIR / src.name.removesuffix(".gz")
        if dest.exists():
            continue
        if src.name.endswith(".gz"):
            with gzip.open(src, "rb") as f_in, open(dest, "wb") as f_out:
                shutil.copyfileobj(f_in, f_out)
        else:
            shutil.copy2(src, dest)
        n += 1
    print(f"downloaded {n} transcript files into {RESULTS_DIR}")


if __name__ == "__main__":
    main()
