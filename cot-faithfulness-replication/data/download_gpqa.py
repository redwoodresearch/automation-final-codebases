"""Download GPQA-Diamond from its official distribution and convert it to data/gpqa/gpqa_diamond.jsonl.

GPQA is not committed to this repo: its authors distribute it inside a password-protected zip
(with a canary string in every record) specifically to keep the questions out of web scrapes
and training corpora, and we don't want to undo that by re-hosting it in plaintext. This script
fetches the official archive from https://github.com/idavidrein/gpqa and converts the diamond
split to the JSONL file the eval reads. The zip password below is the one published in that
repository's README.

  python data/download_gpqa.py

Idempotent: exits early if the output file already exists with the expected hash.
"""

import csv
import hashlib
import io
import json
import sys
import urllib.request
import zipfile
from pathlib import Path

ZIP_URL = "https://github.com/idavidrein/gpqa/raw/main/dataset.zip"
ZIP_PASSWORD = b"deserted-untie-orchid"  # published in the GPQA repository README
CSV_NAME = "dataset/gpqa_diamond.csv"

OUT_PATH = Path(__file__).parent / "gpqa" / "gpqa_diamond.jsonl"
EXPECTED_SHA256 = "5bcf4dcee2e69d421548abe3cf3840bc9504fa89916b9b1b4e161c6f39489db2"
EXPECTED_ROWS = 198


def main() -> None:
    if OUT_PATH.exists():
        digest = hashlib.sha256(OUT_PATH.read_bytes()).hexdigest()
        if digest == EXPECTED_SHA256:
            print(f"{OUT_PATH} already present with the expected hash; nothing to do")
            return
        print(f"{OUT_PATH} exists but its hash differs from the expected one; regenerating")

    print(f"downloading {ZIP_URL} ...")
    with urllib.request.urlopen(ZIP_URL, timeout=120) as resp:
        archive = resp.read()

    raw = zipfile.ZipFile(io.BytesIO(archive)).read(CSV_NAME, pwd=ZIP_PASSWORD)
    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    if len(rows) != EXPECTED_ROWS:
        sys.exit(f"expected {EXPECTED_ROWS} GPQA-Diamond rows, got {len(rows)} — upstream file changed?")

    body = "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in rows)
    digest = hashlib.sha256(body.encode("utf-8")).hexdigest()
    if digest != EXPECTED_SHA256:
        sys.exit(
            "converted file does not match the hash this project's results were produced from\n"
            f"  expected {EXPECTED_SHA256}\n  got      {digest}\n"
            "Upstream may have revised the dataset; the eval code will still run on it, but "
            "question-level comparisons to our published numbers may shift. Delete EXPECTED_SHA256 "
            "checking in this script only if you have verified the upstream change is benign."
        )

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUT_PATH.write_text(body, encoding="utf-8")
    print(f"wrote {OUT_PATH} ({len(rows)} rows, sha256 verified)")


if __name__ == "__main__":
    main()
