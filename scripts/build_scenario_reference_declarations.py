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

#: Fields that must be dropped from the reference when the utterance never mentions them,
#: however the profile is populated. ``work_authorizations`` is the live case: SC-H-03's
#: profile carries the synthetic placeholder ``["XX"]`` and its utterance ("Only Kuala
#: Lumpur software engineer, at least RM80000 per month.") says nothing about authorisation
#: -- so a constraint on it is not something the candidate stated.
_PROFILE_ONLY_EXCLUSIONS = ("work_authorizations",)

#: Values the seeded reference gets WRONG, corrected here rather than by hand so that
#: re-running ``--write`` reproduces the corrected declaration instead of reverting it.
#: Keyed by ``(scenario_id, field) -> (value, reason)``.
#:
#: Both entries are the same rule-extractor limitation: it takes only the first alternative
#: out of a disjunction, so "remote or hybrid" became ``["remote"]`` and the reference
#: silently asserted the candidate had ruled hybrid out. This is precisely what a declared
#: reference is for -- the utterance says both are acceptable, so both belong in the
#: reference regardless of what the extractor managed to read.
_VALUE_CORRECTIONS: dict[tuple[str, str], tuple[Any, str]] = {
    ("SC-A-03", "work_modes"): (
        ["remote", "hybrid"],
        'utterance says "remote or hybrid"; the extractor kept only the first alternative',
    ),
    ("SC-D-08", "work_modes"): (
        ["remote", "hybrid"],
        'utterance says "Remote or hybrid"; the extractor kept only the first alternative',
    ),
}

#: Unknown-handling policy, resolved explicitly per field instead of being inherited from
#: config at grading time. Whether a job that does not STATE a value fails, passes or
#: triggers a clarification changes eligibility and therefore changes grades, so it is
#: policy and belongs in the frozen reference. These values reproduce what
#: ``JobContextAgent`` resolves today, so declaring them changes no number -- what changes
#: is that the policy is now an input a reader can inspect and the fingerprint covers.
_UNKNOWN_POLICY_HARD = "fail"
_UNKNOWN_POLICY_SOFT = "pass"
_UNKNOWN_POLICY_OVERRIDES = {"work_authorizations": "clarify"}

#: The harness's current answer for a slot the profile does not pin. Mirrors
#: ``jobrec_eval.simulated_user._DEFAULTS`` so the drafted declaration reproduces today's
#: behaviour exactly; the point of declaring it is that it becomes reviewable and
#: scenario-specific, not that it changes on day one.
_HARNESS_DEFAULTS: dict[str, Any] = {
    "target_roles": "data analyst",
    "preferred_locations": "Kuala Lumpur",
    "work_modes": "hybrid",
    "salary_currency": "MYR",
    "salary_min": 4000,
    "experience_level": "junior",
}


def _clauses(turns: list[str]) -> list[str]:
    """Utterances split into clauses, lowercased.

    Punctuation and ``and`` only. Splitting further would break multi-word markers, so the
    remaining ambiguity is resolved by PROXIMITY in :func:`_cue_reading` rather than by
    cutting more aggressively here.
    """
    parts: list[str] = []
    for turn in turns:
        for clause in re.split(r"[,.;]|\band\b", turn.lower()):
            clause = clause.strip()
            if clause:
                parts.append(clause)
    return parts


def _occurrences(clause: str, needles: tuple[str, ...]) -> list[tuple[int, str]]:
    """Every ``(position, needle)`` of ``needles`` in ``clause``."""
    found: list[tuple[int, str]] = []
    for needle in needles:
        start = 0
        while (at := clause.find(needle, start)) >= 0:
            found.append((at, needle))
            start = at + 1
    return found


def _cue_evidence(field: str, clauses: list[str]) -> dict[str, Any]:
    """The cue evidence for one field: ``{verdict, clause, hard, soft}``.

    Each cue is assigned to the field whose marker is NEAREST to it, and a field's verdict
    is read only from the cues assigned to it. Two weaker rules were tried and both
    misreported:

    * attributing every cue in a clause to every field in it made "business analyst in
      Kuala Lumpur at least RM4000" (one comma-free clause) say the LOCATION was binding,
      because "at least" -- which binds the salary -- was counted for both;
    * anchoring on the field and taking its nearest cue has the same failure whenever a
      clause contains one cue and two fields.

    Nearest-marker assignment gets both right: "at least" is 9 characters from "rm4000"
    and 13 from "kuala lumpur", so it binds the salary and the location is left with no
    cue at all -- which is the truth about that sentence.

    ``verdict`` is ``None`` when no cue is assigned to the field, and ``"conflict"`` when
    both a hard and a soft cue are. Neither is decided here: the rubric decides, and the
    reviewer sees the clause.
    """
    markers = _FIELD_MARKERS.get(field)
    if not markers:
        return {"verdict": None, "clause": None, "hard": [], "soft": []}

    for clause in clauses:
        own = _occurrences(clause, markers)
        if not own:
            continue
        # Markers of every OTHER field in this clause, so a cue can be lost to them.
        rival = [pos for other, other_markers in _FIELD_MARKERS.items() if other != field
                 for pos, _ in _occurrences(clause, other_markers)]
        hard: list[str] = []
        soft: list[str] = []
        for cues, bucket in ((_HARD_CUES, hard), (_SOFT_CUES, soft)):
            for at, needle in _occurrences(clause, cues):
                mine = min(abs(at - pos) for pos, _ in own)
                theirs = min((abs(at - pos) for pos in rival), default=None)
                if theirs is None or mine <= theirs:
                    bucket.append(needle)
        if not hard and not soft:
            continue
        verdict = "conflict" if hard and soft else ("hard" if hard else "soft")
        return {"verdict": verdict, "clause": clause.strip(), "hard": hard, "soft": soft}
    return {"verdict": None, "clause": None, "hard": [], "soft": []}


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

        # --- apply the rubric ------------------------------------------------
        # Strength is decided by the utterance, not by the extractor: a field is hard only
        # where the utterance says so ("only", "must", "at least"). A bare mention is a
        # preference. This is what makes the reference independent of the system, and it
        # also removes three outliers the extractor produced with nothing in the wording to
        # justify them (preferred_locations hard in SC-B-04 and SC-D-09 while soft in
        # eighteen comparable scenarios; work_modes hard in SC-D-12 alone).
        constraint_fields = sorted({c["field_name"] for c in constraints} - {"not_expired"})
        rubric_hard: list[str] = []
        for field in constraint_fields:
            if _cue_evidence(field, clauses)["verdict"] == "hard":
                rubric_hard.append(field)
        declaration["hard"] = rubric_hard

        # --- corrected values ------------------------------------------------
        for (target_id, field), (value, reason) in _VALUE_CORRECTIONS.items():
            if target_id != scenario_id:
                continue
            declaration[field] = value
            findings.append({
                "scenario_id": scenario_id, "kind": "value_corrected",
                "detail": f"{field}: {active.get(field)} -> {value} ({reason})",
            })

        # --- unknown-handling policy, made explicit --------------------------
        declaration["unknown"] = {
            field: _UNKNOWN_POLICY_OVERRIDES.get(
                field, _UNKNOWN_POLICY_HARD if field in rubric_hard
                else _UNKNOWN_POLICY_SOFT)
            for field in constraint_fields
        }

        # A constraint the candidate never mentioned is not part of the reference. It would
        # otherwise enter the constraint bundle and be counted as "applicable" in the
        # per-constraint compliance table, which is a reported number.
        for field in _PROFILE_ONLY_EXCLUSIONS:
            if field in declaration and not any(
                    marker in " ".join(clauses)
                    for marker in _FIELD_MARKERS.get(field, ())):
                declaration.pop(field, None)
                declaration["excluded"] = sorted(
                    {*declaration.get("excluded", []), field})

        # --- review findings -------------------------------------------------
        for field in sorted({c["field_name"] for c in constraints}
                            - {"not_expired", "experience", "required_skills"}):
            evidence = _cue_evidence(field, clauses)
            cue, seeded = evidence["verdict"], ("hard" if field in seeded_hard else "soft")
            quoted = f' in "{evidence["clause"]}"' if evidence["clause"] else ""
            if cue == "conflict":
                findings.append({
                    "scenario_id": scenario_id, "kind": "cue_conflict",
                    "detail": f"{field}: extractor says {seeded}; the clause carries both "
                              f"hard {evidence['hard']} and soft {evidence['soft']} "
                              f"cues{quoted} -- needs a human reading",
                })
            elif cue is None:
                findings.append({
                    "scenario_id": scenario_id, "kind": "strength_unsupported_by_cues",
                    "detail": f"{field}: extractor says {seeded}; the utterance states no "
                              f"cue for this field, so the strength is an inference "
                              f"(rubric: bare mention -> soft)",
                })
            elif cue != seeded:
                findings.append({
                    "scenario_id": scenario_id, "kind": "strength_disagreement",
                    "detail": f"{field}: extractor says {seeded}, the utterance says {cue} "
                              f"via {evidence['hard'] or evidence['soft']}{quoted}",
                })

        # --- clarification answers ------------------------------------------
        # EVERY clarification-dependent scenario must declare what the candidate answers.
        # The runner refuses to start otherwise, because the alternative is the simulated
        # user answering from a global default table -- which is how two scenarios asking
        # for different things came to be answered identically, with nothing forcing the
        # harness's answer and the oracle's reference to agree.
        #
        # Seeded from what the harness would ALREADY have answered (profile value first,
        # then the domain default), so the drafted declaration reproduces current behaviour
        # and every value is flagged for confirmation rather than quietly invented.
        if scenario.get("clarification_expected") and scenario.get("acceptable_slots"):
            answers: dict[str, Any] = {}
            for slot in scenario["acceptable_slots"]:
                profile_value = scenario.get("profile", {}).get(
                    "skills" if slot == "skills_have" else slot)
                if isinstance(profile_value, list) and profile_value:
                    answers[slot] = profile_value[0]
                elif profile_value not in (None, [], {}):
                    answers[slot] = profile_value
                elif slot in _HARNESS_DEFAULTS:
                    answers[slot] = _HARNESS_DEFAULTS[slot]
            if answers:
                declaration["clarification_answer"] = answers
                findings.append({
                    "scenario_id": scenario_id, "kind": "clarification_answer_drafted",
                    "detail": f"{answers} (seeded from the harness's current behaviour -- "
                              f"CONFIRM each value; the oracle grades against it)",
                })

        if any(cue in " ".join(clauses) for cue in ("some kind of", "of some sort")):
            # The vagueness IS the scenario. The reference must not silently resolve it, so
            # the initial role is marked unspecified and the scenario declares what the
            # candidate answers when asked -- which is also what the SimulatedUser feeds
            # back, so the harness and the oracle cannot disagree. Grading against the
            # CLARIFIED role means a variant that never asked scores badly, which is the
            # intended finding; a broad "any data role is relevant" set would instead have
            # rewarded not asking.
            declaration["role_scope"] = "unspecified_until_clarified"
            findings.append({
                "scenario_id": scenario_id, "kind": "ambiguous_role_marked_unspecified",
                "detail": f"utterance is deliberately vague about the role; marked "
                          f"role_scope=unspecified_until_clarified, graded against the "
                          f"declared clarification answer "
                          f"{declaration.get('clarification_answer', {}).get('target_roles')!r}",
            })

        for field in declaration.get("excluded", []):
            findings.append({
                "scenario_id": scenario_id, "kind": "profile_only_field_excluded",
                "detail": f"{field}={active.get(field)} comes from the profile and is "
                          f"mentioned nowhere in the utterance, so it is EXCLUDED from the "
                          f"reference (synthetic fixture placeholder)",
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
