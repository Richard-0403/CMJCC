"""Verify final_release_v2/ against its own checksums.json.

The point of shipping a manifest is that a reader can check the copy they have rather than
trust it. ``verify_final_release.py`` is pinned to ``final_release/`` (release v1) and its
counts, so this is the same check for v2: every recorded file present and unchanged, and nothing
in the tree left unrecorded.

Exits non-zero on any mismatch.

    python scripts/verify_release_v2.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

ROOT = Path("final_release_v2")
MANIFEST = ROOT / "checksums.json"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    if not MANIFEST.is_file():
        print(f"FAIL: no manifest at {MANIFEST}")
        return 1
    recorded = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]

    on_disk = {p.relative_to(ROOT).as_posix(): p
               for p in ROOT.rglob("*") if p.is_file() and p != MANIFEST}

    missing = sorted(set(recorded) - set(on_disk))
    # Files present but unrecorded matter as much as changed ones: an artifact nobody hashed
    # is an artifact nobody can check, and it is how an audit tool's own output once ended up
    # inside a sealed experiment tree.
    unrecorded = sorted(set(on_disk) - set(recorded))
    changed = sorted(name for name, path in on_disk.items()
                     if name in recorded and _sha256(path) != recorded[name])

    print(f"recorded {len(recorded)} | missing {len(missing)} | changed {len(changed)} "
          f"| unrecorded {len(unrecorded)}")
    for label, items in (("missing", missing), ("changed", changed),
                         ("unrecorded", unrecorded)):
        for name in items[:20]:
            print(f"  {label}: {name}")
        if len(items) > 20:
            print(f"  ... and {len(items) - 20} more {label}")

    if missing or changed or unrecorded:
        print("FAIL")
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
