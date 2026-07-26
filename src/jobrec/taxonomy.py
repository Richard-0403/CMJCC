"""Controlled vocabularies and mappings for extraction, matching and ranking.

These mappings are explicit and versioned. The LLM is never allowed to decide
skill equivalence at runtime; equivalences come from here.
"""

from __future__ import annotations

from .utils.text import normalize_skill, normalize_token

TAXONOMY_VERSION = "1.0.0"

# Canonical skill -> set of aliases (all compared in normalised form).
SKILL_SYNONYMS: dict[str, list[str]] = {
    "sql": ["sql", "postgresql", "postgres", "mysql", "t-sql", "tsql", "sqlite"],
    "python": ["python", "py"],
    "excel": ["excel", "microsoft excel", "ms excel", "spreadsheets"],
    "power bi": ["power bi", "powerbi"],
    "tableau": ["tableau"],
    "r": ["r"],
    "java": ["java"],
    "javascript": ["javascript", "js", "ecmascript"],
    "react": ["react", "reactjs", "react.js"],
    "machine learning": ["machine learning", "ml"],
    "statistics": ["statistics", "stats", "statistical analysis"],
    "communication": ["communication", "communication skills"],
    "data visualization": ["data visualization", "data viz", "visualization"],
    "etl": ["etl", "data pipelines"],
    "aws": ["aws", "amazon web services"],
    "docker": ["docker", "containers"],
}

# Transferable skills: possessing key gives partial credit toward value skills.
TRANSFERABLE_SKILLS: dict[str, list[str]] = {
    "python": ["data analysis", "machine learning", "etl"],
    "sql": ["data analysis", "etl"],
    "pandas": ["data analysis"],
    "statistics": ["machine learning"],
}

# Role family aliases -> canonical role family.
ROLE_FAMILIES: dict[str, list[str]] = {
    "data analyst": ["data analyst", "data analytics", "analytics analyst"],
    "business analyst": ["business analyst", "ba"],
    "product analyst": ["product analyst"],
    "software engineer": ["software engineer", "software developer", "developer", "swe"],
    "data engineer": ["data engineer"],
    "data scientist": ["data scientist"],
    "sales analyst": ["sales analyst"],
}

# Canonical work modes (fixed enumeration for candidate preferences) and aliases.
WORK_MODES: list[str] = ["onsite", "hybrid", "remote"]

WORK_MODE_ALIASES: dict[str, str] = {
    "onsite": "onsite",
    "on-site": "onsite",
    "on site": "onsite",
    "in office": "onsite",
    "in-office": "onsite",
    "office": "onsite",
    "hybrid": "hybrid",
    "remote": "remote",
    "wfh": "remote",
    "work from home": "remote",
    "fully remote": "remote",
}

# Canonical locations for the curated research catalog and their aliases.
LOCATION_ALIASES: dict[str, str] = {
    "kl": "Kuala Lumpur",
    "kuala lumpur": "Kuala Lumpur",
    "penang": "Penang",
    "johor bahru": "Johor Bahru",
    "johor": "Johor Bahru",
    "cyberjaya": "Cyberjaya",
    "singapore": "Singapore",
    "selangor": "Selangor",
    "malaysia": "Malaysia",
}

# Experience level ordering (used for fit/eligibility comparisons).
EXPERIENCE_LEVELS: list[str] = ["intern", "entry", "junior", "mid", "senior", "lead"]

EXPERIENCE_LEVEL_ALIASES: dict[str, str] = {
    "intern": "intern",
    "internship": "intern",
    "entry": "entry",
    "entry-level": "entry",
    "graduate": "entry",
    "junior": "junior",
    "jr": "junior",
    "mid": "mid",
    "mid-level": "mid",
    "intermediate": "mid",
    "senior": "senior",
    "sr": "senior",
    "lead": "lead",
    "principal": "lead",
}

# Approximate years-of-experience band per level (min, max) for inference.
LEVEL_YEARS_BAND: dict[str, tuple[float, float]] = {
    "intern": (0.0, 1.0),
    "entry": (0.0, 2.0),
    "junior": (1.0, 3.0),
    "mid": (3.0, 6.0),
    "senior": (6.0, 12.0),
    "lead": (10.0, 30.0),
}


def _build_alias_index() -> dict[str, str]:
    index: dict[str, str] = {}
    for canonical, aliases in SKILL_SYNONYMS.items():
        for alias in aliases:
            index[normalize_skill(alias)] = canonical
    return index


_SKILL_ALIAS_INDEX = _build_alias_index()


def canonical_skill(skill: str) -> str:
    """Map a raw skill to its canonical form, preserving unknown skills."""
    norm = normalize_skill(skill)
    return _SKILL_ALIAS_INDEX.get(norm, norm)


def canonical_role(role: str) -> str:
    """Map a raw role string to a canonical role family (or normalised input)."""
    norm = normalize_skill(role)
    for canonical, aliases in ROLE_FAMILIES.items():
        for alias in aliases:
            if normalize_skill(alias) == norm or normalize_skill(alias) in norm:
                return canonical
    return norm


def canonical_level(level: str) -> str | None:
    """Map a raw experience-level token to a canonical level."""
    return EXPERIENCE_LEVEL_ALIASES.get(normalize_skill(level))


def canonical_work_mode(mode: str) -> str | None:
    """Map a raw work-mode token to a canonical work mode, or None if unknown."""
    norm = normalize_token(mode)
    if norm in WORK_MODE_ALIASES:
        return WORK_MODE_ALIASES[norm]
    # Tolerate embedded phrasing such as "prefer remote work".
    for alias, canonical in WORK_MODE_ALIASES.items():
        if alias in norm:
            return canonical
    return None


def canonical_location(location: str) -> str | None:
    """Map a raw location string to its canonical form.

    Returns the canonical name for known locations. Unknown but non-empty
    locations are preserved in title case so a stated constraint is never
    silently dropped; empty/whitespace input returns None.
    """
    norm = normalize_token(location)
    if not norm:
        return None
    if norm in LOCATION_ALIASES:
        return LOCATION_ALIASES[norm]
    for alias, canonical in LOCATION_ALIASES.items():
        if alias in norm:
            return canonical
    return norm.title()


def level_rank(level: str | None) -> int | None:
    """Return the ordinal rank of an experience level, or None."""
    if level is None:
        return None
    if level in EXPERIENCE_LEVELS:
        return EXPERIENCE_LEVELS.index(level)
    return None
