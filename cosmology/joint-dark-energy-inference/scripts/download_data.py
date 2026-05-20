"""Record public data targets for the dark-energy inference pipeline.

This script is intentionally conservative: it does not scrape or transform
large survey products. It prints the authoritative source URLs and creates the
expected local directory layout so publication runs can pin exact files later.
"""

from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "data" / "data_manifest.json").read_text())
    raw = root / "data" / "raw"
    raw.mkdir(exist_ok=True)
    for source in manifest["sources"]:
        print(f"{source['name']}: {source['url']}")
    (raw / "README.md").write_text(
        "Place exact downloaded survey products here and record checksums in data_manifest.json.\n"
    )


if __name__ == "__main__":
    main()
