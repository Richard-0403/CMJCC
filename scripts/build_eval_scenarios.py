"""Generate the tagged evaluation scenario set (evaluation/data/scenarios.jsonl).

Reproducible builder for ~30 scenarios covering types A-H with difficulty and
memory/context dependency tags. Kept separate from the golden-test scenarios so
existing tests are unaffected.

The scenarios are hand-designed against the synthetic catalog (data-analyst /
business-analyst / software-engineer roles in Kuala Lumpur / Penang / etc.).
Recommendation scenarios use satisfiable salary thresholds (RM3000-4500);
no-match scenarios use an unsatisfiable threshold (RM50000).

Usage:
    python scripts/build_eval_scenarios.py --output evaluation/data/scenarios.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

S: list[dict] = []


def add(scenario_id, scenario_type, difficulty, profile, turns, *,
        memory_dependency="none", context_dependency="low",
        no_match_expected=False, clarification_expected=False,
        acceptable_slots=None, expected_response="recommendation", notes=""):
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
        "expects": {"response_type": expected_response},
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
    context_dependency="high")
add("SC-E-02", "multiple_hard", "hard",
    {"skills": ["Python", "SQL", "Excel"], "years_experience": 3, "target_roles": ["Business Analyst"]},
    ["Business analyst, only Kuala Lumpur, must pay at least RM4500, onsite only."],
    context_dependency="high", no_match_expected=True, expected_response="no_match",
    notes="Many hard constraints jointly infeasible in the catalog -> correct no-match.")
add("SC-E-03", "multiple_hard", "hard",
    {"skills": ["Python", "SQL"], "years_experience": 2, "target_roles": ["Data Analyst"]},
    ["Only Penang, must be remote, data analyst, at least RM4000."],
    context_dependency="high")
add("SC-E-04", "multiple_hard", "hard",
    {"skills": ["Python", "SQL", "AWS"], "years_experience": 5, "target_roles": ["Software Engineer"]},
    ["Software engineer, only Kuala Lumpur, must be hybrid, at least RM6000."],
    context_dependency="high", no_match_expected=True, expected_response="no_match",
    notes="Many hard constraints jointly infeasible in the catalog -> correct no-match.")
add("SC-E-05", "multiple_hard", "hard",
    {"skills": ["Python", "SQL"], "years_experience": 2, "target_roles": ["Product Analyst"]},
    ["Product analyst, only Kuala Lumpur, at least RM4000, hybrid only."],
    context_dependency="high")

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
    context_dependency="high", no_match_expected=True, expected_response="no_match")
add("SC-H-02", "no_match", "hard",
    {"skills": ["Python", "SQL"], "years_experience": 2, "target_roles": ["Business Analyst"]},
    ["Business analyst, only Penang, must pay at least RM60000 per month."],
    context_dependency="high", no_match_expected=True, expected_response="no_match")
add("SC-H-03", "no_match", "hard",
    {"skills": ["Python", "SQL", "AWS"], "years_experience": 4, "target_roles": ["Software Engineer"],
     "work_authorizations": ["XX"]},
    ["Only Kuala Lumpur software engineer, at least RM80000 per month."],
    context_dependency="high", no_match_expected=True, expected_response="no_match")


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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="evaluation/data/scenarios.jsonl")
    args = parser.parse_args()
    for i, s in enumerate(S):
        s["profile"]["candidate_id"] = f"{s['scenario_id']}-cand"
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w") as fh:
        for s in S:
            fh.write(json.dumps(s) + "\n")
    types = {}
    mem = sum(1 for s in S if s["memory_dependency"] in ("medium", "high"))
    ctx = sum(1 for s in S if s["context_dependency"] == "high")
    for s in S:
        types[s["scenario_type"]] = types.get(s["scenario_type"], 0) + 1
    print(f"Wrote {len(S)} scenarios -> {out}")
    print("by type:", types)
    print(f"memory-dependent(>=medium): {mem} | context-dependent(high): {ctx}")


if __name__ == "__main__":
    main()
