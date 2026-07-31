"""Draft the tagged evaluation scenario set.

Reproducible builder for the scenarios covering types A-H with difficulty and
memory/context dependency tags. Kept separate from the golden-test scenarios so
existing tests are unaffected.

The scenarios are hand-designed against the synthetic catalog (data-analyst /
business-analyst / software-engineer roles in Kuala Lumpur / Penang / etc.).
Recommendation scenarios use satisfiable salary thresholds (RM3000-4500);
no-match scenarios use an unsatisfiable threshold (RM50000).

This script writes a DRAFT and refuses to write an authoritative scenario file.

The reason is asymmetric damage. ``evaluation/data/scenarios.jsonl`` carries the 42
hand-reviewed ``reference`` declarations that the canonical oracle is a pure function
of, and this builder emits no ``reference`` block at all -- grep it, there are zero.
So writing that path would replace declared ground truth with a file that has none,
silently, in a single command with no arguments. Regenerating scenario TEXT is a
drafting operation; adopting a draft as authoritative is a reviewed, manual step that
has to preserve or re-declare every reference.

Usage:
    python scripts/build_eval_scenarios.py
    python scripts/build_eval_scenarios.py --output /tmp/my_draft.jsonl
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

#: Where a draft goes by default. Deliberately NOT the authoritative path.
DEFAULT_OUTPUT = "evaluation/data/scenarios_draft.jsonl"

#: Basenames this builder must never produce. The check is on the FILE NAME rather than
#: on one resolved path, because the default being safe is not a guarantee: an explicit
#: ``--output`` is exactly how the authoritative file would get clobbered, and a
#: path-equality test is easy to walk around with a relative path or a different cwd.
#:
#: Compared case-insensitively. On Windows ``Path.resolve()`` happens to fold the case to
#: whatever is on disk, so ``SCENARIOS.JSONL`` is caught there by accident -- but only
#: while the file exists. On macOS the filesystem is case-insensitive and
#: case-PRESERVING, so ``SCENARIOS.JSONL`` resolves with the requested case and would
#: open the very same file: a name-equality test would miss it and the authoritative set
#: would be overwritten. On Linux it is a different file, which is not a clobber but is
#: still not something this builder should produce.
PROTECTED_BASENAMES = frozenset({"scenarios.jsonl"})

#: Paths that are authoritative in this repository, refused explicitly as well so the
#: error message can name them. Compared with :func:`os.path.normcase` and after symlink
#: resolution, so a link or a case variant pointing at one of them is caught too.
REPO_ROOT = Path(__file__).resolve().parent.parent
PROTECTED_PATHS = (
    REPO_ROOT / "evaluation" / "data" / "scenarios.jsonl",
    REPO_ROOT / "data" / "scenarios" / "scenarios.jsonl",
)


class ProtectedOutputError(RuntimeError):
    """The requested output path is an authoritative scenario file."""


def _normalised(path: Path) -> str:
    """A comparable spelling of ``path``: symlinks resolved, case folded."""
    return os.path.normcase(os.path.realpath(str(path)))


def resolve_output(output: str | Path) -> Path:
    """Return the path to write, or raise :class:`ProtectedOutputError`.

    Kept separate from :func:`main` so the guard is testable without running the
    builder and without a filesystem side effect.
    """
    path = Path(output)
    resolved = path.expanduser().resolve()
    if resolved.name.casefold() in {name.casefold() for name in PROTECTED_BASENAMES}:
        raise ProtectedOutputError(
            f"{path} is named {resolved.name!r}, which is reserved for the authoritative "
            f"scenario set. This builder emits no 'reference' block, so writing it would "
            f"drop the hand-declared references the canonical oracle depends on. Write a "
            f"draft instead (default: {DEFAULT_OUTPUT}) and adopt it in a reviewed step."
        )
    protected = {_normalised(p) for p in PROTECTED_PATHS}
    if _normalised(path.expanduser()) in protected:
        raise ProtectedOutputError(
            f"{path} resolves to the authoritative scenario file {resolved}. Write a "
            f"draft instead (default: {DEFAULT_OUTPUT})."
        )
    return path


S: list[dict] = []


def add(scenario_id, scenario_type, difficulty, profile, turns, *,
        memory_dependency="none", context_dependency="low",
        no_match_expected=False, clarification_expected=False,
        acceptable_slots=None, expected_response="recommendation", notes="",
        hard_fields=None, blocking=None):
    """Append a scenario.

    ``hard_fields`` is the authoritative hard-constraint reference: the fields the
    turn text states as non-negotiable ("only", "must"), which the data-quality
    validator requires of every scenario whose outcome depends on hard filtering
    (R17.2). ``blocking`` names the hard constraint(s) that make a no-match
    scenario infeasible inside its requested role family.
    """
    expects: dict = {"response_type": expected_response}
    if hard_fields:
        expects["hard_fields"] = list(hard_fields)
    if blocking:
        expects["blocking"] = list(blocking)
    S.append({
        "scenario_id": scenario_id,
        "scenario_type": scenario_type,
        "difficulty": difficulty,
        "profile": profile,
        "turns": turns,
        "memory_dependency": memory_dependency,
        "context_dependency": context_dependency,
        "no_match_expected": no_match_expected,
        "clarification_expected": clarification_expected,
        "acceptable_slots": acceptable_slots or [],
        "expects": expects,
        "notes": notes,
    })


# ---- A. Complete single-turn (6) -----------------------------------------
add("SC-A-01", "complete", "easy",
    {"skills": ["Python", "SQL"], "years_experience": 1, "target_roles": ["Data Analyst"],
     "preferred_locations": ["Kuala Lumpur"]},
    ["I want a data analyst role in Kuala Lumpur, hybrid is fine, at least RM4000."],
    context_dependency="medium")
add("SC-A-02", "complete", "easy",
    {"skills": ["Python", "SQL", "Excel"], "years_experience": 2, "target_roles": ["Business Analyst"],
     "preferred_locations": ["Kuala Lumpur"]},
    ["Looking for a business analyst role in Kuala Lumpur, at least RM4000."],
    context_dependency="medium")
add("SC-A-03", "complete", "easy",
    {"skills": ["Python", "SQL"], "years_experience": 3, "target_roles": ["Data Analyst"],
     "preferred_locations": ["Penang"]},
    ["A data analyst position in Penang, remote or hybrid, at least RM4000."],
    context_dependency="medium")
add("SC-A-04", "complete", "medium",
    {"skills": ["Python", "SQL", "AWS"], "years_experience": 4, "target_roles": ["Software Engineer"],
     "preferred_locations": ["Kuala Lumpur"]},
    ["Software engineer role in Kuala Lumpur, hybrid, at least RM5000."],
    context_dependency="medium")
add("SC-A-05", "complete", "easy",
    {"skills": ["Python", "SQL", "Statistics"], "years_experience": 2, "target_roles": ["Product Analyst"],
     "preferred_locations": ["Kuala Lumpur"]},
    ["I want a product analyst role in Kuala Lumpur, at least RM4000."],
    context_dependency="medium")
add("SC-A-06", "complete", "medium",
    {"skills": ["Python", "SQL"], "years_experience": 1, "target_roles": ["Data Analyst"],
     "preferred_locations": ["Cyberjaya"]},
    ["Data analyst in Cyberjaya, hybrid, at least RM3500."],
    context_dependency="medium")

# ---- B. Incomplete / clarification (5) -----------------------------------
add("SC-B-01", "clarification", "medium",
    {"skills": ["Python"], "years_experience": 2},
    ["I am looking for something in Kuala Lumpur, hybrid, around RM5000."],
    clarification_expected=True, acceptable_slots=["target_roles"],
    expected_response="clarification", notes="No role target -> clarify")
add("SC-B-02", "clarification", "medium",
    {"skills": ["SQL", "Excel"], "years_experience": 1},
    ["Any openings in Penang for me?"],
    clarification_expected=True, acceptable_slots=["target_roles"],
    expected_response="clarification")
add("SC-B-03", "clarification", "hard",
    {"skills": ["Python", "SQL"], "years_experience": 2, "preferred_locations": ["Kuala Lumpur"]},
    ["I want a good job with decent pay."],
    clarification_expected=True, acceptable_slots=["target_roles"],
    expected_response="clarification")
add("SC-B-04", "clarification", "medium",
    {"skills": ["Python"], "years_experience": 3},
    ["Something hybrid, at least RM4000, in Kuala Lumpur please."],
    clarification_expected=True, acceptable_slots=["target_roles"],
    expected_response="clarification")
add("SC-B-05", "clarification", "medium",
    {"skills": ["Excel", "Communication"], "years_experience": 1},
    ["I want to work in Penang."],
    clarification_expected=True, acceptable_slots=["target_roles"],
    expected_response="clarification")

# ---- C. Profile-dialogue conflict (5) ------------------------------------
add("SC-C-01", "profile_dialogue_conflict", "medium",
    {"skills": ["Python", "SQL"], "years_experience": 1, "target_roles": ["Data Analyst"],
     "preferred_locations": ["Penang"], "work_modes": ["remote"]},
    ["For this search I only want roles in Kuala Lumpur, at least RM4000."],
    memory_dependency="medium", context_dependency="high",
    notes="Current explicit KL overrides profile Penang for active search only.")
add("SC-C-02", "profile_dialogue_conflict", "medium",
    {"skills": ["Python", "SQL"], "years_experience": 2, "target_roles": ["Business Analyst"],
     "preferred_locations": ["Kuala Lumpur"], "work_modes": ["onsite"]},
    ["Actually hybrid is also fine now; business analyst in Kuala Lumpur at least RM4000."],
    memory_dependency="medium", context_dependency="medium")
add("SC-C-03", "profile_dialogue_conflict", "hard",
    {"skills": ["Python", "SQL"], "years_experience": 1, "target_roles": ["Data Analyst"],
     "preferred_locations": ["Penang"]},
    ["I only want Kuala Lumpur this time, must be at least RM4000."],
    memory_dependency="medium", context_dependency="high")
add("SC-C-04", "profile_dialogue_conflict", "medium",
    {"skills": ["Python", "SQL"], "years_experience": 2, "target_roles": ["Data Analyst"],
     "preferred_locations": ["Kuala Lumpur"], "salary_min": 4000, "salary_currency": "MYR"},
    ["Data analyst, but this search only Penang please, at least RM4000."],
    memory_dependency="medium", context_dependency="high")
add("SC-C-05", "profile_dialogue_conflict", "medium",
    {"skills": ["Python", "SQL"], "years_experience": 3, "target_roles": ["Software Engineer"],
     "preferred_locations": ["Kuala Lumpur"], "work_modes": ["remote"]},
    ["Onsite is also acceptable for a software engineer role in Kuala Lumpur, at least RM5000."],
    memory_dependency="low", context_dependency="medium")

# ---- D. Preference change across turns (5) -------------------------------
add("SC-D-01", "preference_change", "medium",
    {"skills": ["Python", "SQL"], "years_experience": 1, "target_roles": ["Data Analyst"],
     "preferred_locations": ["Kuala Lumpur"]},
    ["Data analyst in Kuala Lumpur, at least RM8000.", "Actually 4000 is also fine."],
    memory_dependency="high", context_dependency="high",
    notes="Latest salary (4000) must control the active search.")
add("SC-D-02", "preference_change", "medium",
    {"skills": ["Python", "SQL"], "years_experience": 2, "target_roles": ["Business Analyst"],
     "preferred_locations": ["Penang"]},
    ["Business analyst in Penang, onsite only.", "Hybrid is fine too, at least RM4000."],
    memory_dependency="high", context_dependency="medium")
add("SC-D-03", "preference_change", "hard",
    {"skills": ["Python", "SQL"], "years_experience": 2, "target_roles": ["Data Analyst"],
     "preferred_locations": ["Kuala Lumpur"]},
    ["Data analyst in Penang, at least RM4000.", "Change that to Kuala Lumpur instead."],
    memory_dependency="high", context_dependency="high")
add("SC-D-04", "preference_change", "medium",
    {"skills": ["Python", "SQL"], "years_experience": 3, "target_roles": ["Product Analyst"],
     "preferred_locations": ["Kuala Lumpur"]},
    ["Product analyst in Kuala Lumpur, remote.", "Hybrid also fine, at least RM4500."],
    memory_dependency="high", context_dependency="medium")
add("SC-D-05", "preference_change", "medium",
    {"skills": ["Python", "SQL", "AWS"], "years_experience": 4, "target_roles": ["Software Engineer"],
     "preferred_locations": ["Kuala Lumpur"]},
    ["Software engineer, at least RM9000.", "6000 is acceptable actually."],
    memory_dependency="high", context_dependency="high")

# ---- E. Multiple hard constraints (5) ------------------------------------
add("SC-E-01", "multiple_hard", "hard",
    {"skills": ["Python", "SQL"], "years_experience": 2, "target_roles": ["Data Analyst"]},
    ["I only want a data analyst role in Kuala Lumpur, must be at least RM4000, hybrid only."],
    context_dependency="high",
    hard_fields=["target_roles", "preferred_locations", "salary_min", "work_modes"])
add("SC-E-02", "multiple_hard", "hard",
    {"skills": ["Python", "SQL", "Excel"], "years_experience": 3, "target_roles": ["Business Analyst"]},
    ["Business analyst, only Kuala Lumpur, must pay at least RM4500, onsite only."],
    context_dependency="high", no_match_expected=True, expected_response="no_match",
    hard_fields=["preferred_locations", "salary_min", "work_modes"],
    blocking=["preferred_locations", "work_modes", "salary_min"],
    notes="Jointly infeasible for business analysts: no KL onsite BA posting pays >= RM4500.")
add("SC-E-03", "multiple_hard", "hard",
    {"skills": ["Python", "SQL"], "years_experience": 2, "target_roles": ["Data Analyst"]},
    ["Only Penang, must be remote, data analyst, at least RM4000."],
    context_dependency="high",
    hard_fields=["target_roles", "preferred_locations", "salary_min", "work_modes"])
add("SC-E-04", "multiple_hard", "hard",
    {"skills": ["Python", "SQL", "AWS"], "years_experience": 5, "target_roles": ["Software Engineer"]},
    ["Software engineer, only Kuala Lumpur, must be hybrid, at least RM6000."],
    context_dependency="high", no_match_expected=True, expected_response="no_match",
    hard_fields=["preferred_locations", "salary_min", "work_modes"],
    blocking=["preferred_locations", "work_modes", "salary_min"],
    notes="Jointly infeasible for software engineers: no KL hybrid SE posting pays >= RM6000.")
add("SC-E-05", "multiple_hard", "hard",
    {"skills": ["Python", "SQL"], "years_experience": 2, "target_roles": ["Product Analyst"]},
    ["Product analyst, only Kuala Lumpur, at least RM4000, hybrid only."],
    context_dependency="high",
    hard_fields=["preferred_locations", "salary_min", "work_modes"])

# ---- F. Soft preference trade-off (4) ------------------------------------
add("SC-F-01", "soft_tradeoff", "medium",
    {"skills": ["Python", "SQL"], "years_experience": 2, "target_roles": ["Data Analyst"]},
    ["I prefer a data analyst role, ideally hybrid and ideally in Kuala Lumpur."],
    context_dependency="low")
add("SC-F-02", "soft_tradeoff", "medium",
    {"skills": ["Python", "SQL", "Excel"], "years_experience": 3, "target_roles": ["Business Analyst"]},
    ["Business analyst, ideally in Penang, prefer hybrid but flexible."],
    context_dependency="low")
add("SC-F-03", "soft_tradeoff", "easy",
    {"skills": ["Python", "SQL"], "years_experience": 2, "target_roles": ["Data Analyst"]},
    ["Data analyst, remote would be nice, prefer higher pay."],
    context_dependency="low")
add("SC-F-04", "soft_tradeoff", "medium",
    {"skills": ["Python", "SQL", "Statistics"], "years_experience": 3, "target_roles": ["Product Analyst"]},
    ["Product analyst, ideally Kuala Lumpur, hybrid preferred."],
    context_dependency="low")

# ---- G. Ambiguous role target (2) ----------------------------------------
add("SC-G-01", "ambiguous_role", "hard",
    {"skills": ["Python", "SQL"], "years_experience": 2, "preferred_locations": ["Kuala Lumpur"]},
    ["I want some kind of data role in Kuala Lumpur, at least RM4000."],
    clarification_expected=True, acceptable_slots=["target_roles"],
    expected_response="clarification", notes="Ambiguous 'data role' -> clarify or broad recall")
add("SC-G-02", "ambiguous_role", "hard",
    {"skills": ["Python", "SQL"], "years_experience": 3, "preferred_locations": ["Penang"]},
    ["Looking for an analyst position of some sort in Penang."],
    clarification_expected=True, acceptable_slots=["target_roles"],
    expected_response="clarification")

# ---- H. No-match (3) -----------------------------------------------------
add("SC-H-01", "no_match", "hard",
    {"skills": ["Python", "SQL"], "years_experience": 1, "target_roles": ["Data Analyst"]},
    ["I only want a data analyst role in Kuala Lumpur with salary at least RM50000 per month."],
    context_dependency="high", no_match_expected=True, expected_response="no_match",
    hard_fields=["target_roles", "preferred_locations", "salary_min"],
    blocking=["salary_min"],
    notes="salary_min alone filters every data-analyst posting in the catalog.")
add("SC-H-02", "no_match", "hard",
    {"skills": ["Python", "SQL"], "years_experience": 2, "target_roles": ["Business Analyst"]},
    ["Business analyst, only Penang, must pay at least RM60000 per month."],
    context_dependency="high", no_match_expected=True, expected_response="no_match",
    hard_fields=["preferred_locations", "salary_min"], blocking=["salary_min"],
    notes="salary_min alone filters every business-analyst posting in the catalog.")
add("SC-H-03", "no_match", "hard",
    {"skills": ["Python", "SQL", "AWS"], "years_experience": 4, "target_roles": ["Software Engineer"],
     "work_authorizations": ["XX"]},
    ["Only Kuala Lumpur software engineer, at least RM80000 per month."],
    context_dependency="high", no_match_expected=True, expected_response="no_match",
    hard_fields=["target_roles", "preferred_locations", "salary_min"],
    blocking=["preferred_locations", "salary_min"],
    notes="Location and salary_min each filter every software-engineer posting.")


# ---- D2. Multi-turn memory (role established first, not repeated) (7) -----
# These directly probe prior-turn memory: full remembers the role from turn 1;
# no_memory loses it and must clarify. Strong memory-contribution signal.
add("SC-D-06", "preference_change", "medium",
    {"skills": ["Python", "SQL"], "years_experience": 1, "preferred_locations": ["Kuala Lumpur"]},
    ["I'm interested in data analyst roles.", "Something hybrid with at least RM4000 would be great."],
    memory_dependency="high", context_dependency="medium",
    notes="Role only in turn 1; no_memory should lose it.")
add("SC-D-07", "preference_change", "medium",
    {"skills": ["Python", "SQL", "Excel"], "years_experience": 2, "preferred_locations": ["Kuala Lumpur"]},
    ["I'd like business analyst positions.", "In Kuala Lumpur, at least RM4000 please."],
    memory_dependency="high", context_dependency="medium")
add("SC-D-08", "preference_change", "medium",
    {"skills": ["Python", "SQL"], "years_experience": 3, "preferred_locations": ["Penang"]},
    ["Looking at data analyst work.", "Remote or hybrid in Penang, at least RM4000."],
    memory_dependency="high", context_dependency="medium")
add("SC-D-09", "preference_change", "hard",
    {"skills": ["Python", "SQL", "AWS"], "years_experience": 4, "preferred_locations": ["Kuala Lumpur"]},
    ["I want software engineer roles.", "Hybrid, at least RM5000, in Kuala Lumpur."],
    memory_dependency="high", context_dependency="medium")
add("SC-D-10", "preference_change", "medium",
    {"skills": ["Python", "SQL", "Statistics"], "years_experience": 2, "preferred_locations": ["Kuala Lumpur"]},
    ["Product analyst roles interest me.", "At least RM4000, hybrid is fine."],
    memory_dependency="high", context_dependency="medium")
add("SC-D-11", "preference_change", "medium",
    {"skills": ["Python", "SQL"], "years_experience": 2, "preferred_locations": ["Kuala Lumpur"]},
    ["Data analyst is what I'm after.", "Prefer hybrid.", "At least RM4000 in Kuala Lumpur."],
    memory_dependency="high", context_dependency="medium",
    notes="Three turns; role and preferences accrue across turns.")
add("SC-D-12", "preference_change", "hard",
    {"skills": ["Python", "SQL"], "years_experience": 1, "preferred_locations": ["Penang"]},
    ["I'm looking for data analyst jobs.", "Actually only in Kuala Lumpur.", "At least RM4000, hybrid."],
    memory_dependency="high", context_dependency="high",
    notes="Role turn1; location override turn2; salary turn3.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default=DEFAULT_OUTPUT,
                        help=f"draft path to write (default: {DEFAULT_OUTPUT}). "
                             "Authoritative scenario files are refused.")
    args = parser.parse_args()
    try:
        out = resolve_output(args.output)
    except ProtectedOutputError as exc:
        print(f"refusing to write: {exc}", file=sys.stderr)
        return 2
    for s in S:
        s["profile"]["candidate_id"] = f"{s['scenario_id']}-cand"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for s in S:
            fh.write(json.dumps(s) + "\n")
    types = {}
    mem = sum(1 for s in S if s["memory_dependency"] in ("medium", "high"))
    ctx = sum(1 for s in S if s["context_dependency"] == "high")
    for s in S:
        types[s["scenario_type"]] = types.get(s["scenario_type"], 0) + 1
    print(f"Wrote {len(S)} DRAFT scenarios -> {out}")
    print("by type:", types)
    print(f"memory-dependent(>=medium): {mem} | context-dependent(high): {ctx}")
    print("NOTE: this is a draft and carries no 'reference' declarations. Adopting it as "
          "the authoritative set is a separate reviewed step.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
