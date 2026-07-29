"""Verify final_release/ against its own checksums.json.

Checks both directions: every recorded artifact is present and unchanged, and no file in
the release is absent from the manifest. Recording only one direction would let a file be
added to the release without ever being covered.

    python scripts/verify_final_release.py
"""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path("final_release")
MANIFEST = "checksums.json"


def main() -> int:
    manifest_path = ROOT / MANIFEST
    if not manifest_path.exists():
        print(f"FAIL: no {manifest_path}")
        return 2
    recorded = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]

    missing: list[str] = []
    changed: list[str] = []
    for rel, expected in recorded.items():
        path = ROOT / rel
        if not path.exists():
            missing.append(rel)
        elif hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            changed.append(rel)

    on_disk = {p.relative_to(ROOT).as_posix() for p in ROOT.rglob("*")
               if p.is_file() and p.name != MANIFEST}
    unrecorded = sorted(on_disk - set(recorded))

    print(f"recorded {len(recorded)} | missing {len(missing)} | changed {len(changed)} "
          f"| unrecorded {len(unrecorded)}")
    for label, items in (("missing", missing), ("changed", changed),
                         ("unrecorded", unrecorded)):
        for item in items[:10]:
            print(f"  {label}: {item}")
    ok = not (missing or changed or unrecorded)
    print("OK" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
