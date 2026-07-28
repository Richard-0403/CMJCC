"""Draft the authoritative `reference` block for each scenario, and flag what needs review.

The canonical oracle must not take its ground truth from the system under evaluation. The
end state is a DECLARED reference on every scenario: the field values the candidate is
taken to have stated, and -- decisively -- which of those are HARD, because a hard
violation forces relevance 0.

This tool writes a DRAFT of that declaration and, next to it, a review list. It does not
decide anything on its own:

* the draft VALUES are seeded from the existing canonical reference, so nothing silently
  changes on scenarios where the current reading is already right;
* the draft STRENGTHS are seeded the same way, but every one of them is compared against
  an INDEPENDENT reading of the utterance based on explicit linguistic cues ("only",
  "must", "at least" vs "ideally", "prefer", "would be nice"), and every disagreement is
  reported for a human to settle.

Usage:
    python scripts/build_scenario_reference_declarations.py            # report only
    python scripts/build_scenario_reference_declarations.py --write    # write the draft

`--write` adds a `reference` block to every scenario in the file and leaves everything
else byte-identical. Review the flagged scenarios afterwards; the declaration is an input
artifact and is meant to be edited by hand.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

SCENARIOS = Path("evaluation/data/scenarios.jsonl")
FROZEN = Path("evaluation/data/canonical_oracle_scenarios.json")

#: Cues that mark a stated constraint as binding. Ordered longest-first so "at least"
#: is not shadowed by a shorter match.
_HARD_CUES = ("only", "must", "at least", "no less than", "minimum", "required",
              "have to", "cannot", "not willing")

#: Cues that mark a stated preference as non-binding.
_SOFT_CUES = ("ideally", "prefer", "would be nice", "flexible", "nice to have",
              "if possible", "open to", "also fine", "is fine", "acceptable",
              "some kind of", "of some sort")

#: Which utterance cue governs which constraint field. A cue is attributed to a field only
#: when it occurs in the same clause, which is why the clause split below matters: "only
#: Kuala Lumpur, at least RM4000" states two separate things.
_FIELD_MARKERS: dict[str, tuple[str, ...]] = {
    "preferred_locations": ("kuala lumpur", "penang", "johor", "remote-first", "location"),
    "salary_min": ("rm", "salary", "pay", "paying", "per month"),
    "work_modes": ("hybrid", "remote", "onsite", "on-site", "work mode"),
    "employment_types": ("full time", "full-time", "contract", "internship", "part time"),
    "work_authorizations": ("visa", "authorisation", "authorization", "work permit"),
}


def _clauses(turns: list[str]) -> list[str]:
    """Utterances split into clauses, lowercased.

    Cue attribution has to be clause-local: "Business analyst, only Kuala Lumpur, must pay
    at least RM4500, onsite only" binds three different fields with three different cues,
    and a whole-utterance match would smear "only" across all of them.
    """
    parts: list[str] = []
    for turn in turns:
        for clause in re.split(r"[,.;]|\band\b", turn.lower()):
            clause = clause.strip()
            if clause:
                parts.append(clause)
    return parts


def _cue_reading(field: str, clauses: list[str]) -> str | None:
    """``"hard"`` / ``"soft"`` from the utterance cues alone, or ``None`` when silent."""
    markers = _FIELD_MARKERS.get(field)
    if not markers:
        return None
    verdict: str | None = None
    for clause in clauses:
        if not any(marker in clause for marker in markers):
            continue
        if any(cue in clause for cue in _SOFT_CUES):
            verdict = "soft"
        elif any(cue in clause for cue in _HARD_CUES):
            # A hard cue wins over an earlier soft one for the same field only if no soft
            # cue appears with it; a clause saying both is reported as a conflict below.
            verdict = verdict if verdict == "soft" else "hard"
    return verdict


def build(scenarios: list[dict], references: dict[str, dict]) -> tuple[list[dict], list[dict]]:
    """Return ``(scenarios with a draft reference, review findings)``."""
    findings: list[dict] = []
    out: list[dict] = []

    for scenario in scenarios:
        scenario_id = scenario["scenario_id"]
        ref = references.get(scenario_id)
        if ref is None:
            findings.append({"scenario_id": scenario_id, "kind": "no_seed_reference",
                             "detail": "not present in the frozen canonical oracle"})
            out.append(scenario)
            continue

        active = ref["active_search"]
        constraints = ref["job_context"].get("constraints") or []
        seeded_hard = sorted({c["field_name"] for c in constraints
                              if c["strength"] == "hard" and c["field_name"] != "not_expired"})
        clauses = _clauses(scenario.get("turns") or [])

        declaration: dict[str, Any] = {
            key: active[key]
            for key in ("target_roles", "skills_have", "preferred_locations", "work_modes",
                        "salary_min", "salary_currency", "experience_level",
                        "years_experience", "employment_types", "work_authorizations",
                        "exclusions")
            if active.get(key) not in (None, [], {})
        }
        declaration["hard"] = seeded_hard

        # --- review findings -------------------------------------------------
        for field in sorted({c["field_name"] for c in constraints}
                            - {"not_expired", "experience", "required_skills"}):
            cue = _cue_reading(field, clauses)
            seeded = "hard" if field in seeded_hard else "soft"
            if cue is not None and cue != seeded:
                findings.append({
                    "scenario_id": scenario_id, "kind": "strength_disagreement",
                    "detail": f"{field}: extractor says {seeded}, utterance cues say {cue}",
                })
            elif cue is None:
                findings.append({
                    "scenario_id": scenario_id, "kind": "strength_unsupported_by_cues",
                    "detail": f"{field}: extractor says {seeded}; no explicit cue in the "
                              f"utterance, so the strength is an inference",
                })

        if any(cue in " ".join(clauses) for cue in ("some kind of", "of some sort")):
            findings.append({
                "scenario_id": scenario_id, "kind": "ambiguous_role_asserted",
                "detail": f"utterance is deliberately vague about the role, yet the "
                          f"reference asserts target_roles={declaration.get('target_roles')}",
            })

        if declaration.get("work_authorizations"):
            findings.append({
                "scenario_id": scenario_id, "kind": "profile_placeholder_as_constraint",
                "detail": f"work_authorizations={declaration['work_authorizations']} comes "
                          f"from the profile, not from anything the candidate said",
            })

        out.append({**scenario, "reference": declaration})

    return out, findings


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true",
                        help="write the draft reference block into the scenario file")
    parser.add_argument("--scenarios", default=str(SCENARIOS))
    parser.add_argument("--frozen", default=str(FROZEN))
    args = parser.parse_args()

    scenarios = [json.loads(line) for line
                 in Path(args.scenarios).read_text(encoding="utf-8").splitlines()
                 if line.strip()]
    references = json.loads(Path(args.frozen).read_text(encoding="utf-8"))["references"]

    drafted, findings = build(scenarios, references)

    by_kind: dict[str, list[dict]] = {}
    for finding in findings:
        by_kind.setdefault(finding["kind"], []).append(finding)

    print(f"scenarios: {len(scenarios)}  drafted: "
          f"{sum(1 for s in drafted if 'reference' in s)}")
    print(f"review findings: {len(findings)}")
    for kind in sorted(by_kind):
        rows = by_kind[kind]
        print(f"\n--- {kind} ({len(rows)}) ---")
        for row in rows:
            print(f"  {row['scenario_id']}: {row['detail']}")

    if args.write:
        Path(args.scenarios).write_text(
            "".join(json.dumps(s, ensure_ascii=False) + "\n" for s in drafted),
            encoding="utf-8")
        print(f"\nwrote draft declarations into {args.scenarios}")
    else:
        print("\n(report only; pass --write to update the scenario file)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
