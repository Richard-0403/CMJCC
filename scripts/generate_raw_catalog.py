"""Generate a deterministic raw job catalog CSV for the research prototype.

The generated catalog deliberately covers multiple role families, experience
levels, locations, work modes, salary availability (present / missing / partial),
skill overlap, and active / expired / boundary deadlines relative to the fixed
reference date. Output is fully deterministic (seeded) for reproducibility.

The deliberately expired postings are annotated ``is_test_fixture=true`` /
``expected_ineligible_reason=expired`` so the data-quality validator records them
as acknowledged fixtures instead of defects to delete (R17.1). Those two columns
are excluded from the catalog content hash, so adding them does not change
``catalog_hash``.

Usage:
    python scripts/generate_raw_catalog.py --output data/raw/jobs.csv --count 200
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path

REFERENCE_DATE = date(2026, 1, 1)

ROLE_TEMPLATES = [
    ("Data Analyst", "data analyst", ["sql", "excel"], ["power bi", "tableau", "python"],
     ["Build dashboards", "Analyse business metrics", "Prepare reports"]),
    ("Business Analyst", "business analyst", ["excel", "communication"], ["sql", "power bi"],
     ["Gather requirements", "Model processes", "Support stakeholders"]),
    ("Product Analyst", "product analyst", ["sql", "statistics"], ["python", "tableau"],
     ["Analyse product funnels", "Run experiments", "Report KPIs"]),
    ("Software Engineer", "software engineer", ["python", "sql"], ["docker", "aws", "javascript"],
     ["Develop services", "Write tests", "Review code"]),
    ("Data Engineer", "data engineer", ["python", "sql", "etl"], ["aws", "docker"],
     ["Build data pipelines", "Maintain warehouses", "Optimise ETL"]),
    ("Data Scientist", "data scientist", ["python", "machine learning", "statistics"], ["sql", "aws"],
     ["Train models", "Analyse data", "Communicate findings"]),
    ("Sales Analyst", "sales analyst", ["excel", "sql"], ["power bi", "communication"],
     ["Analyse sales pipeline", "Forecast revenue", "Support sales team"]),
]

COMPANIES = [
    "Acme Data", "Nusantara Tech", "KualaSoft", "Penang Analytics", "Selangor Systems",
    "Orchid AI", "Straits Digital", "Hibiscus Labs", "Merlion Data", "Batik Cloud",
]

LOCATIONS = [
    ("Malaysia", "Kuala Lumpur", "MY"),
    ("Malaysia", "Penang", "MY"),
    ("Malaysia", "Johor Bahru", "MY"),
    ("Malaysia", "Cyberjaya", "MY"),
    ("Singapore", "Singapore", "SG"),
]

WORK_MODES = ["onsite", "hybrid", "remote", "unspecified"]
LEVELS = ["intern", "entry", "junior", "mid", "senior"]
LEVEL_YEARS = {"intern": (0, 1), "entry": (0, 2), "junior": (1, 3), "mid": (3, 6), "senior": (6, 10)}
EMPLOYMENT = ["full-time", "contract", "part-time"]


def build_rows(count: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    rows: list[dict] = []
    for i in range(count):
        title_base, role_family, req, pref, resp = rng.choice(ROLE_TEMPLATES)
        level = rng.choice(LEVELS)
        country, city, auth = rng.choice(LOCATIONS)
        work_mode = rng.choice(WORK_MODES)
        ymin, ymax = LEVEL_YEARS[level]

        # Salary: 70% present (MYR), 15% partial (min only), 15% missing.
        salary_roll = rng.random()
        base = {"intern": 1500, "entry": 3000, "junior": 4000, "mid": 7000, "senior": 12000}[level]
        currency = "SGD" if country == "Singapore" else "MYR"
        scale = 1.0 if currency == "MYR" else 0.9
        smin = smax = sper = scur = ""
        if salary_roll < 0.70:
            smin = int(base * scale)
            smax = int(base * scale * rng.uniform(1.2, 1.6))
            sper = "month"
            scur = currency
        elif salary_roll < 0.85:
            smin = int(base * scale)
            sper = "month"
            scur = currency

        # Deadlines: mix of future, boundary, expired, and none. The expired
        # slice is deliberate -- the pipeline must prove it never recommends
        # them -- so those rows carry the fixture annotation that tells the
        # data-quality validator not to demand their deletion (R17.1).
        deadline_roll = rng.random()
        deadline = ""
        is_active = "true"
        is_test_fixture = ""
        expected_ineligible_reason = ""
        if deadline_roll < 0.55:
            deadline = (REFERENCE_DATE + timedelta(days=rng.randint(10, 120))).isoformat()
        elif deadline_roll < 0.70:
            deadline = REFERENCE_DATE.isoformat()  # boundary: due today
        elif deadline_roll < 0.85:
            deadline = (REFERENCE_DATE - timedelta(days=rng.randint(5, 60))).isoformat()
            is_active = "false"  # expired
            is_test_fixture = "true"
            expected_ineligible_reason = "expired"
        # else: no deadline, active

        # Work authorization: Singapore roles require SG auth sometimes.
        auth_req = ""
        if country == "Singapore" and rng.random() < 0.6:
            auth_req = "SG"

        title = f"{level.capitalize()} {title_base}"
        rows.append(
            {
                "job_id": f"job-{i:04d}",
                "title": title,
                "company": rng.choice(COMPANIES),
                "description": f"{title} at a growing team. Responsibilities: {', '.join(resp)}.",
                "role_family": role_family,
                "industry": rng.choice(["technology", "finance", "retail", "healthcare"]),
                "employment_type": rng.choice(EMPLOYMENT),
                "required_skills": "|".join(req),
                "preferred_skills": "|".join(rng.sample(pref, k=min(len(pref), rng.randint(1, len(pref))))),
                "responsibilities": "|".join(resp),
                "salary_min": smin,
                "salary_max": smax,
                "salary_currency": scur,
                "salary_period": sper,
                "country": country,
                "city": city,
                "region": "",
                "work_mode": work_mode,
                "min_years_experience": ymin,
                "max_years_experience": ymax,
                "experience_level": level,
                "required_work_authorization": auth_req,
                "application_deadline": deadline,
                "is_active": is_active,
                "is_test_fixture": is_test_fixture,
                "expected_ineligible_reason": expected_ineligible_reason,
                "source_uri": f"synthetic://catalog/{i:04d}",
            }
        )
    return rows


FIELDS = [
    "job_id", "title", "company", "description", "role_family", "industry",
    "employment_type", "required_skills", "preferred_skills", "responsibilities",
    "salary_min", "salary_max", "salary_currency", "salary_period", "country",
    "city", "region", "work_mode", "min_years_experience", "max_years_experience",
    "experience_level", "required_work_authorization", "application_deadline",
    "is_active", "is_test_fixture", "expected_ineligible_reason", "source_uri",
]


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate raw job catalog CSV")
    parser.add_argument("--output", default="data/raw/jobs.csv")
    parser.add_argument("--count", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rows = build_rows(args.count, args.seed)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} jobs to {out}")


if __name__ == "__main__":
    main()
