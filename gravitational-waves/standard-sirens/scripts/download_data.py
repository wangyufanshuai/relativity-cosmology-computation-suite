from __future__ import annotations

import json
from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    manifest = json.loads((root / "data" / "data_manifest.json").read_text())
    (root / "data" / "raw").mkdir(exist_ok=True)
    for source in manifest["sources"]:
        print(f"{source['name']}: {source['url']}")


if __name__ == "__main__":
    main()
