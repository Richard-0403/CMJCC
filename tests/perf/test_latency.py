"""Performance tests: end-to-end and per-component latency across catalog sizes (R28).

What this measures
------------------
For each catalog size (100 / 200 / 300 jobs) and each run mode (``deterministic`` and
``hybrid`` with the offline mock provider) a fresh single-turn recommendation is run
``REPEATS`` times. Every turn reports its own timings on
``RunRecord.component_latency_ms``, whose keys are the orchestrator's timed stages
(``intent_extraction``, ``memory_merge``, ``retrieval``, ``filtering``, ``ranking``,
``explanation``) plus ``total`` (the end-to-end turn latency). From those samples this
module computes the median, the interquartile range and the P95 (R28.1) for the
end-to-end latency and for every component, at every catalog size (R28.2).

LLM latency vs rule latency (R28.3)
-----------------------------------
The split is derived from what a run actually did, never assumed: a component is put in
the LLM bucket only when the run recorded model calls AND that component is the stage
that issues them (``intent_extraction`` on the recommendation path). So in ``hybrid``
mode ``intent_extraction`` is LLM latency and everything else is rule latency, while in
``deterministic`` mode no model call happens at all, every stage is rule latency and the
reported LLM latency is exactly 0.0 - it is not fabricated. The provider-reported call
latency (``LLMCallRecord.latency_ms``) is reported alongside as
``provider_reported_llm_ms``; with the offline mock it is 0.0 because there is no network
round trip, so the measured LLM cost is the wall-clock of the model-shaped stage.

These are measurements, not wall-clock thresholds: asserting absolute milliseconds would
be flaky on shared CI hardware. The assertions are structural (all sizes measured, all
statistics computed, per-component breakdown present and summing to a sensible share of
the end-to-end total, LLM and rule latency reported separately) and the numbers are
reported to the terminal and to ``artifacts/reports/perf_latency.json``.

Run with::

    pytest tests/perf -q        # or: make test-perf

The module is marked ``perf`` so the default suite can skip it (``-m "not perf"``).

Validates: Requirements 28.1, 28.2, 28.3
"""

from __future__ import annotations

import csv
import io
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

import pytest
from hypothesis import given
from hypothesis import strategies as st
from scripts.generate_raw_catalog import FIELDS, build_rows

from jobrec.app_service import AppService
from jobrec.catalog import catalog_hash, load_catalog, normalize_job, write_catalog
from jobrec.config import load_config

pytestmark = pytest.mark.perf

#: Catalog sizes required by R28.2.
CATALOG_SIZES = (100, 200, 300)

#: Run modes exercised. ``deterministic`` makes no model call; ``hybrid`` (mock provider)
#: runs the model-shaped extraction path, which is what makes the LLM/rule split visible.
MODES = ("deterministic", "hybrid")

#: Timed turns per (mode, size) cell, plus discarded warm-up turns (first-call effects:
#: lazy imports, taxonomy/regex warm-up, retriever index build).
REPEATS = 15
WARMUPS = 2

#: Snapshot id used by scripts/prepare_catalog.py, kept so the size-200 catalog built here
#: is content-identical to the shipped data/processed/jobs.jsonl.
SNAPSHOT_ID = "catalog-2026-01-v1"
SHIPPED_CATALOG = "data/processed/jobs.jsonl"

#: Orchestrator stages present on every successful recommendation turn.
COMPONENTS = (
    "intent_extraction", "memory_merge", "retrieval", "filtering", "ranking", "explanation",
)
#: The single stage that issues model calls on the recommendation path.
LLM_BEARING_COMPONENT = "intent_extraction"
#: Key holding the end-to-end turn latency in ``component_latency_ms``.
TOTAL_KEY = "total"

PROFILE = {
    "candidate_id": "perf-candidate",
    "skills": ["Python", "SQL"],
    "years_experience": 1,
    "preferred_locations": ["Kuala Lumpur"],
    "work_modes": ["hybrid"],
    "salary_min": 3500,
    "salary_currency": "MYR",
}
UTTERANCE = "I want a data analyst role in Kuala Lumpur, hybrid is fine, at least RM4000."

REPORT_PATH = Path("artifacts/reports/perf_latency.json")


# --------------------------------------------------------------------------- statistics
def percentile(values: list[float], q: float) -> float:
    """Linear-interpolation percentile (``q`` in [0, 100]) over unsorted ``values``."""
    if not values:
        raise ValueError("percentile of an empty sample")
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    pos = (len(ordered) - 1) * (q / 100.0)
    low = int(pos)
    high = min(low + 1, len(ordered) - 1)
    frac = pos - low
    return float(ordered[low] + (ordered[high] - ordered[low]) * frac)


def summarize(values: list[float]) -> dict[str, float]:
    """Median / IQR / P95 summary of a latency sample (R28.1)."""
    p25 = percentile(values, 25)
    p75 = percentile(values, 75)
    return {
        "n": float(len(values)),
        "min": float(min(values)),
        "p25": p25,
        "median": percentile(values, 50),
        "p75": p75,
        "iqr": p75 - p25,
        "p95": percentile(values, 95),
        "max": float(max(values)),
    }


# ----------------------------------------------------------------------------- sampling
@dataclass
class Sample:
    """One measured turn."""

    end_to_end_ms: float
    service_wall_ms: float
    components: dict[str, float]
    model_call_count: int
    provider_reported_llm_ms: float

    @property
    def llm_components(self) -> dict[str, float]:
        """Component timings attributable to the model, derived from observed calls."""
        if self.model_call_count == 0:
            return {}
        return {k: v for k, v in self.components.items() if k == LLM_BEARING_COMPONENT}

    @property
    def rule_components(self) -> dict[str, float]:
        llm = self.llm_components
        return {k: v for k, v in self.components.items() if k not in llm}

    @property
    def llm_ms(self) -> float:
        return sum(self.llm_components.values())

    @property
    def rule_ms(self) -> float:
        return sum(self.rule_components.values())


@dataclass
class Cell:
    """All samples for one (mode, catalog size) combination."""

    mode: str
    size: int
    catalog_hash: str
    samples: list[Sample] = field(default_factory=list)

    def series(self, pick) -> list[float]:
        return [pick(s) for s in self.samples]

    def report(self) -> dict:
        comp_stats = {
            name: summarize([s.components[name] for s in self.samples]) for name in COMPONENTS
        }
        return {
            "mode": self.mode,
            "catalog_size": self.size,
            "catalog_hash": self.catalog_hash,
            "repeats": len(self.samples),
            "model_calls_per_turn": sorted({s.model_call_count for s in self.samples}),
            "end_to_end_ms": summarize(self.series(lambda s: s.end_to_end_ms)),
            "service_wall_ms": summarize(self.series(lambda s: s.service_wall_ms)),
            "llm_ms": summarize(self.series(lambda s: s.llm_ms)),
            "rule_ms": summarize(self.series(lambda s: s.rule_ms)),
            "llm_components": sorted(self.samples[0].llm_components),
            "rule_components": sorted(self.samples[0].rule_components),
            "provider_reported_llm_ms": summarize(
                self.series(lambda s: s.provider_reported_llm_ms)
            ),
            "components_ms": comp_stats,
        }


def _build_catalog(size: int, out_dir: Path) -> Path:
    """Write a normalised catalog of exactly ``size`` jobs.

    Reuses the project's own generator + normaliser through an in-memory CSV, i.e. the
    exact ``generate_raw_catalog -> prepare_catalog`` pipeline the shipped catalog comes
    from. Job ids come from the generator (``job-0000``..``job-{size-1}``) so they stay
    unique, and the first 200 records are content-identical to
    ``data/processed/jobs.jsonl`` - so size 100/200 subsample the real catalog and size
    300 extends it with the same distribution instead of duplicating records.
    """
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=FIELDS)
    writer.writeheader()
    writer.writerows(build_rows(size, seed=42))
    rows = list(csv.DictReader(io.StringIO(buffer.getvalue())))
    jobs = [normalize_job(row, SNAPSHOT_ID) for row in rows]
    path = out_dir / f"jobs_{size}.jsonl"
    write_catalog(jobs, path)
    return path


def _measure(mode: str, size: int, catalog_path: Path) -> Cell:
    cfg = load_config(f"configs/{mode}.yaml", base_dir="configs")
    service = AppService(cfg, str(catalog_path))
    service.create_candidate(PROFILE)
    cell = Cell(mode=mode, size=size, catalog_hash=service.catalog_hash)

    for i in range(WARMUPS + REPEATS):
        # A fresh session per repeat keeps every measured turn the FIRST turn of a
        # conversation; reusing one session would grow the dialogue and confound the
        # catalog-size comparison with turn-count effects.
        session_id = service.create_session(PROFILE["candidate_id"], "full")
        started = time.perf_counter()
        result = service.process_turn(session_id, UTTERANCE)
        wall_ms = (time.perf_counter() - started) * 1000.0
        assert result.run_record.success, result.response.message
        if i < WARMUPS:
            continue
        latencies = dict(result.run_record.component_latency_ms)
        cell.samples.append(
            Sample(
                end_to_end_ms=float(latencies[TOTAL_KEY]),
                service_wall_ms=wall_ms,
                components={k: float(v) for k, v in latencies.items() if k != TOTAL_KEY},
                model_call_count=len(result.model_calls),
                provider_reported_llm_ms=sum(
                    float(c.latency_ms or 0.0) for c in result.model_calls
                ),
            )
        )
    return cell


def _report_lines(cells: dict[tuple[str, int], Cell]) -> list[str]:
    lines = [
        f"R28 latency report - {REPEATS} timed turns per cell ({WARMUPS} warm-up discarded)",
    ]
    for mode in MODES:
        for size in CATALOG_SIZES:
            rep = cells[(mode, size)].report()
            e2e, llm, rule = rep["end_to_end_ms"], rep["llm_ms"], rep["rule_ms"]
            lines.append(
                f"  {mode:<13} size={size:<4} end-to-end median={e2e['median']:7.2f}ms "
                f"iqr={e2e['iqr']:6.2f} p95={e2e['p95']:7.2f}  |  "
                f"llm median={llm['median']:6.2f}ms  rule median={rule['median']:7.2f}ms  "
                f"(model calls/turn={rep['model_calls_per_turn']}, "
                f"provider-reported llm median="
                f"{rep['provider_reported_llm_ms']['median']:.2f}ms)"
            )
            for name in COMPONENTS:
                stats = rep["components_ms"][name]
                bucket = "llm " if name in rep["llm_components"] else "rule"
                lines.append(
                    f"      {bucket} {name:<17} median={stats['median']:7.3f}ms "
                    f"iqr={stats['iqr']:6.3f} p95={stats['p95']:7.3f}"
                )
    return lines


@pytest.fixture(scope="module")
def catalogs(tmp_path_factory) -> dict[int, Path]:
    """One normalised catalog file per required size (R28.2)."""
    catalog_dir = tmp_path_factory.mktemp("perf_catalogs")
    return {size: _build_catalog(size, catalog_dir) for size in CATALOG_SIZES}


@pytest.fixture(scope="module")
def cells(catalogs, pytestconfig) -> dict[tuple[str, int], Cell]:
    """Measure every (mode, catalog size) cell once, then report the numbers."""
    measured = {
        (mode, size): _measure(mode, size, catalogs[size])
        for mode in MODES
        for size in CATALOG_SIZES
    }

    report = {
        "requirement": "R28.1-28.3",
        "repeats": REPEATS,
        "warmups": WARMUPS,
        "utterance": UTTERANCE,
        "llm_bearing_component": LLM_BEARING_COMPONENT,
        "note": (
            "LLM latency is the wall-clock of the model-issuing stage on runs that "
            "actually recorded model calls; deterministic runs record none, so their LLM "
            "latency is 0.0. provider_reported_llm_ms comes from LLMCallRecord.latency_ms "
            "and is 0.0 with the offline mock provider (no network round trip)."
        ),
        "cells": [measured[(mode, size)].report() for mode in MODES for size in CATALOG_SIZES],
    }
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text(json.dumps(report, indent=2))

    terminal = pytestconfig.pluginmanager.getplugin("terminalreporter")
    lines = [*_report_lines(measured), f"  report written to {REPORT_PATH}"]
    if terminal is not None:
        terminal.write_line("")
        for line in lines:
            terminal.write_line(line)
    else:  # pragma: no cover - only when the terminal plugin is disabled
        print("\n".join(lines))
    return measured


# -------------------------------------------------------------------------------- tests
@pytest.mark.parametrize("size", CATALOG_SIZES)
def test_latency_is_measured_for_each_required_catalog_size(cells, catalogs, size):
    """R28.2: 100 / 200 / 300-job catalogs are each measured, with unique job ids."""
    for mode in MODES:
        assert len(cells[(mode, size)].samples) == REPEATS

    jobs = load_catalog(catalogs[size])
    assert len(jobs) == size
    assert len({j.job_id for j in jobs}) == size, "catalog job ids must stay unique"
    for mode in MODES:
        assert cells[(mode, size)].catalog_hash == catalog_hash(jobs)

    if size == 200:
        # The measured 200-job catalog IS the shipped catalog, so the 100/300 points are a
        # real subsample / same-distribution extension of it rather than a fresh corpus.
        assert catalog_hash(jobs) == catalog_hash(load_catalog(SHIPPED_CATALOG))


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("size", CATALOG_SIZES)
def test_end_to_end_statistics_are_computed(cells, mode, size):
    """R28.1: median, IQR and P95 of the end-to-end turn latency are reported."""
    stats = cells[(mode, size)].report()["end_to_end_ms"]
    assert stats["n"] == REPEATS
    assert stats["median"] > 0.0
    assert stats["iqr"] >= 0.0
    assert stats["min"] <= stats["p25"] <= stats["median"] <= stats["p75"] <= stats["p95"]
    assert stats["p95"] <= stats["max"]


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("size", CATALOG_SIZES)
def test_per_component_breakdown_is_present_and_sums_sensibly(cells, mode, size):
    """R28.1: every component is timed, summarised, and accounts for most of the total."""
    cell = cells[(mode, size)]
    for sample in cell.samples:
        assert set(sample.components) == set(COMPONENTS)
        assert all(v >= 0.0 for v in sample.components.values())
        # Components are nested inside the turn, so they cannot exceed the end-to-end
        # total, and they should account for the bulk of it (the rest is bookkeeping).
        component_sum = sum(sample.components.values())
        assert component_sum <= sample.end_to_end_ms * 1.05 + 1.0
        assert component_sum >= sample.end_to_end_ms * 0.5
        assert sample.service_wall_ms >= sample.end_to_end_ms * 0.9

    for name, stats in cell.report()["components_ms"].items():
        assert stats["n"] == REPEATS, name
        assert stats["iqr"] >= 0.0, name
        assert stats["p25"] <= stats["median"] <= stats["p75"] <= stats["p95"], name


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("size", CATALOG_SIZES)
def test_llm_latency_is_reported_separately_from_rule_latency(cells, mode, size):
    """R28.3: the LLM and rule buckets partition the components, per run mode."""
    cell = cells[(mode, size)]
    rep = cell.report()

    for sample in cell.samples:
        llm, rule = sample.llm_components, sample.rule_components
        assert not (set(llm) & set(rule)), "a component cannot be both LLM and rule"
        assert set(llm) | set(rule) == set(COMPONENTS)
        assert abs((sample.llm_ms + sample.rule_ms) - sum(sample.components.values())) < 1e-6

        if mode == "hybrid":
            # The model-shaped path really ran: a model call was recorded and its stage
            # is the one reported as LLM latency.
            assert sample.model_call_count >= 1
            assert set(llm) == {LLM_BEARING_COMPONENT}
            assert sample.llm_ms > 0.0
        else:
            # Deterministic mode contacts no model, so LLM latency is honestly 0.0 and
            # every stage - including rule-based extraction - is rule latency.
            assert sample.model_call_count == 0
            assert llm == {}
            assert sample.llm_ms == 0.0
            assert set(rule) == set(COMPONENTS)
        # The offline mock does not measure a network round trip; never treat its
        # provider-reported latency as a real model cost.
        assert sample.provider_reported_llm_ms >= 0.0

    assert rep["llm_ms"]["median"] >= 0.0
    assert rep["rule_ms"]["median"] > 0.0
    assert rep["llm_components"] == ([LLM_BEARING_COMPONENT] if mode == "hybrid" else [])


@given(
    st.lists(st.floats(min_value=0.0, max_value=1e5, allow_nan=False, allow_infinity=False),
             min_size=1, max_size=64)
)
def test_summarize_orders_its_statistics(values):
    """The reported statistics are internally consistent for any latency sample."""
    stats = summarize(values)
    assert stats["n"] == len(values)
    assert stats["min"] <= stats["p25"] <= stats["median"] <= stats["p75"] <= stats["p95"]
    assert stats["p95"] <= stats["max"]
    assert stats["iqr"] == pytest.approx(stats["p75"] - stats["p25"])
