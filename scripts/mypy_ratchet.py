"""Run mypy over ``src`` and enforce a non-increasing error ceiling.

Extracted from the CI workflow so local and CI enforce the SAME rule from the same
code. It used to exist only as an inline shell block in ``.github/workflows/ci.yml``,
while the Makefile ran ``mypy src || true`` -- so the local target reported success no
matter what mypy said, and the two could not disagree only because one of them never
had an opinion.

Two distinct failures are separated deliberately:

* mypy could not complete (exit status >= 2, e.g. a crash or a bad config). Treating
  that as "no errors" is how a broken type check silently becomes a green gate.
* mypy completed and reported MORE errors than the ceiling.

``src`` currently carries 14 pre-existing annotation errors, identical on Linux and
Windows. The ceiling is a ratchet, not an amnesty: lower :data:`DEFAULT_MAX_ERRORS` as
errors are fixed, and never raise it to make a change pass.

    python scripts/mypy_ratchet.py
    python scripts/mypy_ratchet.py --max-errors 12
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

#: The single source of truth for the ceiling. CI does not carry its own copy.
DEFAULT_MAX_ERRORS = 14

#: Type-check target. Kept identical to what CI ran, so the verdict does not depend on
#: which entry point invoked it.
TARGETS = ("src",)

_FOUND = re.compile(r"^Found (\d+) error", re.MULTILINE)


def run(max_errors: int, *, log_path: Path | None = None) -> int:
    process = subprocess.run(
        [sys.executable, "-m", "mypy", *TARGETS],
        capture_output=True, text=True,
    )
    output = process.stdout + process.stderr
    sys.stdout.write(output)
    if log_path is not None:
        log_path.write_text(output, encoding="utf-8", newline="\n")

    if process.returncode >= 2:
        print(f"FAIL: mypy could not complete the type check (exit {process.returncode})",
              file=sys.stderr)
        return 1

    matches = _FOUND.findall(output)
    errors = int(matches[-1]) if matches else 0
    print(f"mypy reported {errors} error(s); ceiling is {max_errors}")
    if errors > max_errors:
        print(f"FAIL: mypy errors increased to {errors} (ceiling {max_errors})",
              file=sys.stderr)
        return 1
    if errors < max_errors:
        print(f"NOTE: {max_errors - errors} error(s) below the ceiling -- lower "
              f"DEFAULT_MAX_ERRORS in scripts/mypy_ratchet.py to lock the improvement in.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-errors", type=int,
                        default=int(os.environ.get("MYPY_MAX_ERRORS",
                                                   DEFAULT_MAX_ERRORS)),
                        help=f"error ceiling (default: {DEFAULT_MAX_ERRORS})")
    parser.add_argument("--log", type=Path, default=None,
                        help="also write mypy output to this file")
    args = parser.parse_args()
    return run(args.max_errors, log_path=args.log)


if __name__ == "__main__":
    raise SystemExit(main())
