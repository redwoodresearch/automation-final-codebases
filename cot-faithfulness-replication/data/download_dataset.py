"""Download Anthropic's released CoT Faithfulness dataset from Google Drive.

Idempotent: skips download if the extracted JSONL files already exist.
Large data files are gitignored; data/manifest.json (hashes, row counts) is committed instead.
"""

import argparse
import hashlib
import json
import subprocess
import zipfile
from pathlib import Path

DRIVE_FOLDER_URL = "https://drive.google.com/drive/folders/1l0pkcZxvFwMtczst_hhiCC44v-IiODlY?usp=sharing"
DATA_DIR = Path(__file__).parent
RAW_DIR = DATA_DIR / "cot_faithfulness_raw"
EXTRACTED_DIR = DATA_DIR / "faithfulness"

EXPECTED_FILES = [
    f"{hint_type}_{correctness}.jsonl"
    for hint_type in ["suggestion", "posthoc", "fewshot_symbol", "fewshot_order"]
    for correctness in ["True", "False"]
]


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["gdown", "--folder", DRIVE_FOLDER_URL, "-O", str(RAW_DIR)],
        check=True,
    )


def extract() -> None:
    zips = list(RAW_DIR.rglob("faithfulness.zip"))
    assert len(zips) == 1, f"Expected exactly one faithfulness.zip, found: {zips}"
    EXTRACTED_DIR.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zips[0]) as zf:
        zf.extractall(EXTRACTED_DIR)


def locate_jsonl_files() -> dict[str, Path]:
    found = {}
    for name in EXPECTED_FILES:
        matches = list(EXTRACTED_DIR.rglob(name))
        assert len(matches) == 1, f"Expected exactly one {name}, found: {matches}"
        found[name] = matches[0]
    return found


def verify_manifest(files: dict[str, Path]) -> None:
    """Check downloaded files against the committed manifest (the provenance lock for the
    exact dataset version this project's results were produced from)."""
    manifest_path = DATA_DIR / "manifest.json"
    expected = json.loads(manifest_path.read_text())["files"]
    problems = []
    for name, path in sorted(files.items()):
        rel = str(path.relative_to(DATA_DIR))
        exp = expected.get(rel)
        if exp is None:
            problems.append(f"{rel}: not in the committed manifest")
            continue
        got = sha256_file(path)
        if got != exp["sha256"]:
            problems.append(f"{rel}: sha256 {got[:12]}... != expected {exp['sha256'][:12]}...")
    if problems:
        raise SystemExit(
            "downloaded dataset does not match data/manifest.json:\n  "
            + "\n  ".join(problems)
            + "\nUpstream may have changed. The eval code will still run on the new files, but "
            "question-level comparisons to this project's published numbers may shift. "
            "Rerun with --write-manifest only if you have verified the change is benign."
        )
    print(f"verified {len(files)} files against data/manifest.json")


def write_manifest(files: dict[str, Path]) -> None:
    manifest = {"source": DRIVE_FOLDER_URL, "files": {}}
    readmes = list(RAW_DIR.rglob("README.md"))
    for readme in readmes:
        manifest["files"][str(readme.relative_to(DATA_DIR))] = {
            "sha256": sha256_file(readme),
            "bytes": readme.stat().st_size,
        }
    for name, path in sorted(files.items()):
        n_rows = sum(1 for line in open(path, encoding="utf-8") if line.strip())
        manifest["files"][str(path.relative_to(DATA_DIR))] = {
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
            "rows": n_rows,
        }
    manifest_path = DATA_DIR / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Wrote {manifest_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="Re-download even if files exist")
    parser.add_argument("--write-manifest", action="store_true",
                        help="maintainer: regenerate data/manifest.json from the downloaded files "
                             "instead of verifying against it")
    args = parser.parse_args()

    have_all = all(list(EXTRACTED_DIR.rglob(name)) for name in EXPECTED_FILES) if EXTRACTED_DIR.exists() else False
    if have_all and not args.force:
        print("All expected JSONL files already present; skipping download (use --force to re-download).")
    else:
        download()
        extract()

    files = locate_jsonl_files()
    if args.write_manifest:
        write_manifest(files)
    else:
        verify_manifest(files)


if __name__ == "__main__":
    main()
