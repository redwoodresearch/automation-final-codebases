"""Guarded writing of the committed results tables.

The analyze_* scripts recompute the tables that the figure scripts read. Those tables are
committed, and the raw transcripts they are computed from are not (too large — see
data/download_transcripts.py), so a naive re-run with transcripts missing would happily
overwrite a good table with an all-null one. `write_table` refuses to do that: it fails if
any model/dataset came out empty, unless the caller passes allow_incomplete=True.
"""

import json
from pathlib import Path


class IncompleteTable(Exception):
    pass


def write_table(path: Path, payload: dict, missing: list[str], allow_incomplete: bool = False) -> None:
    """Write `payload` as JSON to `path`, unless inputs were missing.

    missing: human-readable descriptions of the inputs that could not be read. If non-empty
    and allow_incomplete is False, nothing is written and IncompleteTable is raised.
    """
    if missing and not allow_incomplete:
        raise IncompleteTable(
            f"{len(missing)} required input(s) missing, so the recomputed table would be incomplete "
            f"and would overwrite the committed {path.name}. Nothing was written.\n"
            "  Missing: " + ", ".join(missing[:8]) + (" ..." if len(missing) > 8 else "") + "\n"
            "  Fetch the raw transcripts (python data/download_transcripts.py) or regenerate them\n"
            "  (see ./reproduce.sh full). Pass --allow-incomplete to write a partial table anyway."
        )
    path.write_text(json.dumps(payload, indent=1) + "\n")
    if missing:
        print(f"WARNING: wrote a PARTIAL {path.name} ({len(missing)} inputs missing)")
    print(f"wrote {path}")
