"""Analysis report generation (Markdown) from computed metrics + statistics.

The Markdown template reads only the assembled data dict and the generated
plots; it does not recompute metrics. Wording rules (guide sections 24.1, 28)
are respected: engineering thresholds, observed differences, confidence
intervals and statistical significance are distinguished, and over-strong claims
are avoided.

Wording that depends on WHICH relevance labels produced the ranking metrics is generated,
not hardcoded: the header disclaimer, §4 Annotation Reliability, §5/§6/§7 source lines,
§12 Threats to Validity and the conclusion's next-steps sentence all read the
``relevance_source`` block of the data dict, so a report rendered under
``--relevance-source human`` never claims that no human raters were used. §5.5 puts the
two sources side by side. A data dict written without that block describes the automatic
oracle, which is what such a run was.

Report *output* is additionally gated on configuration consistency (R15.2): a
comparison report is only written when the compared runs share catalog,
scenarios, prompts, model settings and commit, and when each ablation pair
differs in nothing but its own mechanism's feature flags (R32.7). The gate lives
in :func:`write_report` (and :func:`require_consistent_runs`), not in
:func:`generate_markdown`, because rendering the template is a pure function of
the assembled data while writing files is the point at which output is produced.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from jobrec.evaluation.checksums import restamp_checksums
from jobrec.orchestration.feature_flags import CONTEXT_FLAGS, MEMORY_FLAGS

from .consistency import (
    ConsistencyError,
    load_run_manifests,
    require_consistent,
    save_run_manifests,
)

#: Ablation pairs whose Δ this report attributes to a single mechanism, with the
#: flag group that is allowed to differ between the two variants (R32.7).
_ABLATION_PAIRS: tuple[tuple[str, str, frozenset[str] | set[str]], ...] = (
    ("full", "no_memory", MEMORY_FLAGS),
    ("full", "no_context", CONTEXT_FLAGS),
)


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and pd.isna(x)):
        return "N/A"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _counts_phrase(counts) -> str:
    """A ``{name: count}`` mapping as readable prose.

    Two report lines interpolated the dict directly, so the document carried raw Python
    repr -- ``{'ambiguous_role': 2, 'clarification': 5, ...}`` -- braces, quotes and all.
    """
    if not counts:
        return "none"
    if not isinstance(counts, dict):
        return str(counts)
    return ", ".join(f"{name} {value}" for name, value in sorted(counts.items()))


#: Below this, a p-value is printed in scientific notation instead of being rounded.
#: Fixed 3-decimal rounding printed ``0.000`` for everything smaller, which reads as
#: exactly zero -- a p-value that cannot exist. Real values here reached 4.66e-10, and
#: an exact test's p also cannot be distinguished from a borderline 0.0004 once rounded.
_P_SCIENTIFIC_BELOW = 0.001


def _fmt_p(value) -> str:
    """A p-value as text: never ``0.000``, never a bare zero."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "N/A"
    try:
        p = float(value)
    except (TypeError, ValueError):
        return str(value)
    if p < 0:
        return _fmt(p)
    if p == 0.0:
        # Genuinely zero cannot come out of these tests; say so rather than print 0.
        return "<1e-300"
    if p < _P_SCIENTIFIC_BELOW:
        return f"{p:.2e}"
    return f"{p:.3f}"


#: Small counts spelled as words, matching the prose style of the template.
_NUMBER_WORDS = {1: "One", 2: "Two", 3: "Three", 4: "Four", 5: "Five",
                 6: "Six", 7: "Seven", 8: "Eight", 9: "Nine", 10: "Ten"}

#: Where the experiment runner writes run bundles, relative to the pipeline's
#: ``--out-root``: ``_runs/<experiment_id>/<variant>/<scenario_id>/<run_index>/``
#: (see ``run_pipeline``: ``runs_root = Path(out_root) / "_runs"`` and
#: ``ExperimentRunner._run_one``: ``exp_dir / variant / scenario_id / run_index``).
#: The analysis output directory is ``<out_root>/<experiment_id>/``, i.e. ``_runs``
#: is a sibling of it, so from the report the bundles are two levels up.
RUNS_DIR_NAME = "_runs"


def _run_bundle_pointer(experiment_id: str) -> str:
    """The real, on-disk location of the run bundles for this experiment.

    The header used to point at a ``raw/`` directory that the pipeline never
    creates. The bundles live under ``<out_root>/_runs/<experiment_id>/...`` while
    the analysis tables live under ``<out_root>/<experiment_id>/``, so the two
    directories are siblings and the pointer says so.
    """
    return f"`{RUNS_DIR_NAME}/{experiment_id}/<variant>/<scenario_id>/<run_index>/`"


def _variant_count_phrase(variants) -> str:
    """Render ``<N> variant(s) (<names>) is/are`` for any variant count.

    The count was hardcoded as "Five", which contradicted itself on the three
    variant runs the pipeline actually ships (full/no_memory/no_context).
    Deriving it from the experiment manifest keeps the sentence true for any
    variant list, and the singular form keeps a one-variant run grammatical.
    """
    names = list(variants or [])
    if not names:
        return "No variants are"
    count = _NUMBER_WORDS.get(len(names), str(len(names)))
    noun = "variant" if len(names) == 1 else "variants"
    verb = "is" if len(names) == 1 else "are"
    return f"{count} {noun} ({', '.join(names)}) {verb}"


def _delta_phrase(delta, lo, hi) -> str:
    if delta is None:
        return "not computable"
    if lo is None or hi is None:
        return f"Δ={delta:+.3f}"
    if lo > 0:
        return f"Δ={delta:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] (CI excludes 0)"
    if hi < 0:
        return f"Δ={delta:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] (CI excludes 0, negative)"
    return f"Δ={delta:+.3f}, 95% CI [{lo:+.3f}, {hi:+.3f}] (CI includes 0)"


#: Canonical variant order for every per-variant table (§5, §7). Variants outside this
#: list are appended in sorted order rather than dropped, so no table can silently omit
#: a variant the frame carries.
_VARIANT_ORDER: tuple[str, ...] = ("full", "profile_only", "one_shot", "no_memory",
                                   "no_context")

#: Columns of the compact §5 variant summary, as ``(metric key, header)``. The
#: denominators block below the table is rendered from the same list, so the two tables
#: cannot disagree about which metric a column belongs to.
_VARIANT_TABLE_COLS: tuple[tuple[str, str], ...] = (
    ("ndcg_at_5", "NDCG@5"), ("precision_at_5", "P@5"), ("hcsr", "HCSR"),
    ("task_success", "TaskSucc"), ("grounding", "Grounding"),
    ("handoff_success", "Handoff"), ("turn_count", "Turns"),
    ("total_latency_ms", "Lat(ms)"))


def _ordered_variants(present) -> list[str]:
    """``present`` in canonical order, with unknown variants appended sorted.

    Filtering by a hardcoded variant list used to drop whole variants from §7 with no
    note in the report. Ordering instead of filtering keeps the tables readable while
    making it impossible to lose a variant that the frame contains.
    """
    seen = list(dict.fromkeys(str(v) for v in present))
    known = [v for v in _VARIANT_ORDER if v in seen]
    return known + sorted(v for v in seen if v not in _VARIANT_ORDER)


def _variant_table(variant_summary: pd.DataFrame) -> str:
    cols = list(_VARIANT_TABLE_COLS)
    head = "| variant | " + " | ".join(c[1] for c in cols) + " |"
    sep = "|" + "---|" * (len(cols) + 1)
    lines = [head, sep]
    by = {r["variant"]: r for _, r in variant_summary.iterrows()}
    for v in _ordered_variants(by):
        r = by[v]
        cells = []
        for key, _ in cols:
            nd = 0 if key == "total_latency_ms" else (2 if key == "turn_count" else 3)
            cells.append(_fmt(r.get(f"{key}_mean"), nd))
        lines.append(f"| {v} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _n_cell(row, key: str) -> str:
    """The ``<metric>_n`` denominator of one §5 cell, or an empty cell when absent.

    Defensive in the style of :func:`_col_mean`: a frame written before a metric existed
    simply has no denominator to disclose, which is printed as nothing rather than
    raising or inventing a count.
    """
    count = _count(row.get(f"{key}_n"))
    return "" if count is None else str(count)


def _variant_denominator_table(variant_summary: pd.DataFrame) -> str:
    """Per-variant scenario denominator behind every mean shown in :func:`_variant_table`.

    The means alone are not readable: a variant that returns no ranking list on a
    scenario is absent from that metric's denominator, so its ranking means are averaged
    over a different — and easier — subset of scenarios than ``full``'s. The counts are
    already in the ``<metric>_n`` columns of ``metrics/variant_summary.csv``; this table
    only exposes them next to the means they belong to.
    """
    if variant_summary.empty or "variant" not in variant_summary.columns:
        return "_No denominator data._"
    cols = list(_VARIANT_TABLE_COLS)
    lines = ["| variant | " + " | ".join(f"n({c[1]})" for c in cols) + " |",
             "|" + "---|" * (len(cols) + 1)]
    by = {r["variant"]: r for _, r in variant_summary.iterrows()}
    for v in _ordered_variants(by):
        lines.append(f"| {v} | " + " | ".join(_n_cell(by[v], key) for key, _ in cols) + " |")
    return "\n".join(lines)


#: Metrics of the §1 headline bullet, as ``(metric key, label)``. The denominator of each
#: is read from the same ``<metric>_n`` column that feeds :func:`_variant_denominator_table`.
_HEADLINE_METRICS: tuple[tuple[str, str], ...] = (
    ("ndcg_at_5", "NDCG@5"), ("hcsr", "HCSR"), ("task_success", "Task Success"),
    ("grounding", "Grounding"), ("handoff_success", "Handoff"))


def _headline_metrics(row) -> str:
    """The §1 headline metrics, each with the scenario count its mean was taken over.

    The headline printed five bare means, so it mixed denominators inside a single bullet:
    the ranking metrics are averaged over the scenarios that returned a ranked list while
    ``task_success`` is defined on every scenario. Kept as a sentence rather than promoted
    to a table -- the headline is meant to be read in one line -- and a metric whose
    ``<metric>_n`` column is absent simply prints without a count (see :func:`_n_cell`).
    """
    if row is None or not len(row):
        return "N/A"
    parts = []
    for key, label in _HEADLINE_METRICS:
        count = _n_cell(row, key)
        parts.append(f"{label} {_fmt(row.get(f'{key}_mean'))}"
                     + (f" (n={count})" if count else ""))
    return ", ".join(parts)


def _overall_delta_bullet(row: dict | None) -> str:
    """One §5.x bullet body: the Δ phrase plus the pair count the Δ was computed on.

    Printing Δ and its CI without the pair count hid the fact that a Δ against a variant
    which abandoned scenarios is estimated on the scenarios that *both* variants
    answered. ``n_pairs`` is the field ``compare()`` records for exactly that count.
    """
    if not row:
        return "N/A"
    phrase = _delta_phrase(row.get("delta"), row.get("ci_low"), row.get("ci_high"))
    pairs = _count(row.get("n_pairs"))
    return phrase if pairs is None else f"{phrase}, n={pairs} paired scenarios"


#: The dialogue-baseline comparisons §5.x reports, as ``(metric, label, other)``.
_BASELINE_BULLETS = (
    ("ndcg_at_5", "NDCG@5", "profile_only"),
    ("ndcg_at_5", "NDCG@5", "one_shot"),
    ("task_success", "Task success", "profile_only"),
    ("task_success", "Task success", "one_shot"),
)


def _baseline_delta_bullets(overall: list[dict]) -> str:
    """§5.x bullets, restricted to comparisons the run actually contains.

    These were three hardcoded bullets, so an analysis whose variant set omitted a
    baseline printed ``N/A`` for it -- a claim that the comparison was attempted and
    yielded nothing, when it was never in scope. A comparison that is absent is now
    absent, and one that IS present but not estimable still prints its reason.
    """
    by_key = {(r.get("metric"), r.get("other")): r for r in overall or []}
    lines = [f"- {label}, full vs {other}: {_overall_delta_bullet(by_key[(metric, other)])}."
             for metric, label, other in _BASELINE_BULLETS
             if (metric, other) in by_key]
    if not lines:
        return ("_This analysis contains no dialogue-baseline variant, so there is no "
                "full-vs-baseline comparison to report._")
    return "\n".join(lines)


def _contrib_table(df: pd.DataFrame, subset: str, family: str = "primary") -> str:
    """Render one Δ table for a scenario ``subset`` within one outcome ``family``.

    The contribution frame carries both outcome families (see
    :data:`jobrec_eval.cli.PRIMARY` / :data:`jobrec_eval.cli.SECONDARY`) with Holm
    applied inside each family independently, so the table must be filtered by
    ``family`` for the printed p-values to be the ones that were actually corrected
    together. The default keeps §6.1/§6.2 on the pre-registered primary family. A frame
    written before the column existed carries primary outcomes only, so it renders whole
    for ``family="primary"`` and reports no data for any other family.
    """
    if df.empty or "subset" not in df.columns:
        return "_No data._"
    if "family" in df.columns:
        sub = df[(df["subset"] == subset) & (df["family"] == family)]
    elif family != "primary":
        return "_No data._"
    else:
        sub = df[df["subset"] == subset]
    if sub.empty:
        return "_No data._"
    lines = ["| metric | full | other | Δ | 95% CI | p | p(Holm) | effect | n |",
             "|---|---|---|---|---|---|---|---|---|"]
    for _, r in sub.iterrows():
        ci = f"[{_fmt(r['ci_low'])}, {_fmt(r['ci_high'])}]" if r["ci_low"] is not None else "N/A"
        eff = f"{_fmt(r.get('effect_size'))} ({r.get('effect_type')})" if r.get("effect_size") is not None else "N/A"
        lines.append(
            f"| {r['metric']} | {_fmt(r['base_mean'])} | {_fmt(r['other_mean'])} | "
            f"{_fmt(r['delta'])} | {ci} | {_fmt_p(r.get('p_value'))} | "
            f"{_fmt_p(r.get('p_value_holm'))} | "
            f"{eff} | {int(r['n_pairs'])} |")
    note = _estimand_note(sub)
    if note:
        lines += ["", note]
    return "\n".join(lines)


def _estimand_note(sub: pd.DataFrame) -> str:
    """Disclose the estimand behind the ``task_success`` row, and the rate beside it.

    Every cell of that row (mean, Δ, CI, p, effect, n) describes the scenario-level
    binary collapsed by majority vote over repeats -- the pre-registered estimand. The
    success RATE over repeats is a different quantity; it is printed here rather than
    silently occupying the mean and Δ cells, which is what it used to do while the p-value
    and n described the binaries.
    """
    if "metric" not in sub.columns:
        return ""
    rows = sub[sub["metric"] == "task_success"]
    if rows.empty:
        return ""
    row = rows.iloc[0]
    note = ("`task_success` is the **scenario-level binary** (repeats collapsed by "
            "majority vote, even-repeat ties → 0) in every column of its row, including "
            "the Δ and the CI.")
    base_rate, other_rate = row.get("base_repeat_mean"), row.get("other_repeat_mean")
    if base_rate is not None and not pd.isna(base_rate):
        note += (f" For reference, the mean success RATE over repeats is "
                 f"{_fmt(base_rate)} (full) vs {_fmt(other_rate)} "
                 f"(Δ {_fmt(row.get('repeat_mean_delta'))}); that is a different "
                 f"estimand and is not what the test above was computed on.")
    return note


def _compliance_cell(row) -> str:
    """One §5.2 cell: the compliance rate, the count it was taken over, unknown share.

    A bare rate is not comparable across variants here for the same reason the §5 means
    are not: ``applicable`` is the number of (recommendation, constraint) pairs the check
    could be applied to, and a variant that returned fewer recommendations has a smaller
    one. Printing ``n`` in the cell keeps a perfect rate over a handful of pairs from
    reading as constraint enforcement proven over the whole run.

    ``unk`` is the share of that denominator whose constraint value could not be
    determined. Unknowns are counted in the denominator, so a large ``unk`` means the rate
    is driven by missing data rather than by observed violations, and the cell says so
    instead of letting the rate stand alone.
    """
    rate = _fmt(row.get("compliance"))
    applicable = _count(row.get("applicable"))
    if applicable is None:
        return rate
    unknown = row.get("unknown_rate")
    unknown_note = ("" if unknown is None or pd.isna(unknown) or float(unknown) <= 0
                    else f", unk {float(unknown) * 100:.0f}%")
    return f"{rate} (n={applicable}{unknown_note})"


def _compliance_table(rows: list[dict]) -> str:
    """Per-constraint compliance for every variant the frame carries (§5.2).

    The column list was hardcoded to ``full``/``no_context``/``profile_only``, so the two
    dialogue baselines were dropped from the compliance table with no note even though
    ``metrics/constraint_compliance.csv`` carries a row per (constraint field, variant) for
    all of them. Ordering via :func:`_ordered_variants` instead of filtering keeps the
    canonical column order and makes it impossible to omit a variant silently.

    Each cell carries its own denominator (see :func:`_compliance_cell`): the rates sit on
    per-variant, per-field bases that differ by an order of magnitude, so the bare rates
    were not comparable across columns.
    """
    if not rows:
        return "_No per-constraint compliance data._"
    df = pd.DataFrame(rows)
    fields = sorted(df["constraint_field"].unique())
    variants = _ordered_variants(df["variant"])
    lines = ["| constraint field | " + " | ".join(variants) + " |",
             "|" + "---|" * (len(variants) + 1)]
    for f in fields:
        cells = []
        for v in variants:
            r = df[(df.constraint_field == f) & (df.variant == v)]
            cells.append(_compliance_cell(r.iloc[0]) if len(r) else "N/A")
        lines.append(f"| {f} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _pr_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    """§5.3 precision/recall tables, in the canonical variant order.

    This table carried its own row order, which disagreed with §5, §5.2 and §7, and it
    skipped any variant outside that list. :func:`_ordered_variants` both fixes the
    disagreement and removes the filtering, so every variant in the frame is rendered.
    """
    if not rows:
        return "_No data._"
    head = "| variant | " + " | ".join(c[1] for c in cols) + " |"
    lines = [head, "|" + "---|" * (len(cols) + 1)]
    by = {r["variant"]: r for r in rows}
    for v in _ordered_variants(by):
        r = by[v]
        lines.append(f"| {v} | " + " | ".join(_fmt(r.get(c[0])) for c in cols) + " |")
    return "\n".join(lines)


def _clarification_efficiency_table(eff_rows: list[dict], clar_rows: list[dict]) -> str:
    """Clarification efficiency joined with the response-turn distribution (R7.4/R7.5).

    Columns are the six figures the readiness checklist asks for: median and IQR of
    response turns (from ``metrics/clarification_efficiency.csv``), necessary
    clarification recall (from ``metrics/clarification_metrics.csv``), the unnecessary-ask
    count, the efficiency score, and the repeated-slot guard activations.

    ``Abandoned`` (``asked_unresolved``) and ``AnsweredRate`` sit immediately before
    ``EffScore``, so the score is never read without the two figures that say whether the
    variant's dialogues actually resolved (R7.4/R7.5).
    """
    if not eff_rows:
        return "_No clarification-efficiency data._"
    clar_by_variant = {r["variant"]: r for r in clar_rows or []}
    cols = ["variant", "MedTurns", "IQR(Q1-Q3)", "NecRecall", "NecAsked", "UnnecAsked",
            "NecMissed", "RepeatGuard", "Abandoned", "AnsweredRate",
            "Tier res/aband/skip", "MedEff", "MeanEff", "n(runs)"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    by_variant = {r["variant"]: r for r in eff_rows}
    for v in _ordered_variants(by_variant):
        r = by_variant[v]
        c = clar_by_variant.get(v, {})
        iqr = (f"{_fmt(r.get('iqr_response_turns'), 2)} "
               f"({_fmt(r.get('q1_response_turns'), 2)}-{_fmt(r.get('q3_response_turns'), 2)})")
        tiers = (f"{_fmt(_count(r.get('tier_resolved')))}/"
                 f"{_fmt(_count(r.get('tier_abandoned')))}/"
                 f"{_fmt(_count(r.get('tier_skipped')))}")
        lines.append(
            f"| {v} | {_fmt(r.get('median_response_turns'), 2)} | {iqr} | "
            f"{_fmt(c.get('necessary_recall'))} | {_fmt(r.get('necessary_asked'))} | "
            f"{_fmt(r.get('unnecessary_asked'))} | {_fmt(r.get('necessary_missed'))} | "
            f"{_fmt(c.get('repeated'))} | {_fmt(r.get('asked_unresolved'))} | "
            f"{_fmt(c.get('answered_rate'))} | {tiers} | "
            f"{_fmt(r.get('median_efficiency_score'), 2)} | "
            f"{_fmt(r.get('efficiency_score'), 2)} | "
            f"{_fmt(r.get('runs'))} |")
    lines += ["", _EFFICIENCY_SCALE_NOTE]
    return "\n".join(lines)


#: How to read the efficiency columns. Without this the mean invites a magnitude
#: interpretation it cannot support: on a scale whose skip penalty is 1e6, one skipped run
#: in twenty produces a variant mean near -50000, which says nothing about "how
#: inefficient" the variant is -- only that some runs fell into the worst tier.
_EFFICIENCY_SCALE_NOTE = (
    "`Tier res/aband/skip` counts runs by efficiency TIER: asked-and-resolved, "
    "asked-then-abandoned, and necessary-clarification-skipped. This is the primary "
    "reading. `MedEff` is the median score (where the typical run sits) and `MeanEff` "
    "the mean. **`MeanEff` is a penalty scale, not a rate**: the skip tier costs 1e6 and "
    "the abandon tier 1e3, so a five-figure mean encodes the SHARE of runs in the worst "
    "tier, not a magnitude of inefficiency. Compare variants on the tier counts and the "
    "median; use the mean only for the ordering it guarantees (resolved > abandoned > "
    "skipped)."
)


def _count(value) -> int | None:
    """A count cell as ``int``, or ``None`` when it is missing.

    A ``None`` written into a numeric DataFrame column round-trips as ``NaN``, so both
    spellings of "not measured" must collapse to ``None`` before a count is tested for
    truthiness -- ``NaN`` is truthy and would otherwise read as "faults were injected".
    """
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    return int(value)


def _failure_metrics_table(rows: list[dict]) -> str:
    """The four R10 failure-path rates with the numerator/denominator behind each."""
    if not rows:
        return "_No failure-path metrics._"
    lines = ["| metric | source | numerator | denominator | value |",
             "|---|---|---|---|---|"]
    for r in rows:
        denominator = _count(r.get("denominator"))
        value = (_fmt(r.get("value")) if denominator else "N/A (empty denominator)")
        lines.append(f"| {r['metric']} | {r.get('source', '')} | "
                     f"{_fmt(_count(r.get('numerator')))} | {_fmt(denominator)} | {value} |")
    return "\n".join(lines)


def _injection_denominators(rows: list[dict]) -> dict[str, int | None]:
    """Denominator of each failure-path rate, keyed by metric name."""
    return {r["metric"]: _count(r.get("denominator")) for r in rows or []}


def _fault_injection_prose(rows: list[dict]) -> str:
    """One paragraph stating whether the LOADED runs injected anything (R10.8/10.9).

    When the denominators are empty the section says so plainly instead of printing a
    detection rate of 0.0 or a recovery rate of 1.000 that no observation supports.
    """
    denominators = _injection_denominators(rows)
    injected = denominators.get("failure_detection_rate")
    recoverable = denominators.get("recovery_success_rate")
    if injected:
        detected = next((_count(r.get("numerator")) for r in rows
                         if r["metric"] == "failure_detection_rate"), None)
        prose = (f"The loaded runs injected {injected} fault(s), of which {_fmt(detected)} "
                 "were detected. ")
        prose += (f"{recoverable} of them were designed to be recoverable."
                  if recoverable else
                  "None of them were designed to be recoverable, so the recovery rate "
                  "reads N/A.")
        return prose
    return (
        "**The loaded runs injected no faults, so the detection and recovery rates read "
        "N/A, not 0.0 and not 1.000.** An empty denominator is reported as N/A on purpose: "
        "a main-experiment run set contains no injected faults, and printing 0.0 detected "
        "or a perfect recovery over zero observations would be misleading. The robustness "
        "evidence therefore comes from the fault-injection suite enumerated in §10.2, not "
        "from this run set.")


def _data_quality_section(dq: dict | None) -> str:
    """Counts from the data-quality report saved beside the other artifacts (R17.3)."""
    if not dq:
        return "_No data-quality report was produced for this run._"
    skipped = dq.get("checks_skipped") or {}
    lines = [
        f"- Validated {dq.get('job_count')} catalog job(s) and "
        f"{dq.get('scenario_count')} scenario(s) at reference date "
        f"{dq.get('reference_date')}: **{dq.get('error_count')} error(s), "
        f"{dq.get('warning_count')} warning(s), {dq.get('info_count')} acknowledged test "
        f"fixture(s)**.",
        "- Errors block nothing here: the pipeline reports data-quality findings, and the "
        "standalone `validate` command is the gate that exits non-zero on an "
        "error-severity finding.",
        f"- Checks run: {', '.join(dq.get('checks_run') or []) or 'none'}.",
    ]
    if skipped:
        lines.append("- Not checked: "
                     + "; ".join(f"{k} ({v})" for k, v in sorted(skipped.items())) + ".")
    counts = dq.get("counts_by_violation_type") or {}
    if counts:
        lines.append(f"- Findings by type: {_counts_phrase(counts)}.")
    lines.append("- Full report: `data_quality_report.json` (covered by `checksums.json`).")
    return "\n".join(lines)


def _pairing_provenance_table(overall: list[dict]) -> str:
    """Where each task-success ``n_pairs`` came from, so it is visibly scenario-level.

    Rendered from the bookkeeping :func:`jobrec_eval.statistics.compare` records for the
    binary task-success comparisons: scenario count, total run count, repeats per
    scenario, validly paired scenarios and discordant pairs (R6.3).
    """
    rows = [r for r in overall or [] if r.get("metric") == "task_success"]
    if not rows:
        return "_No paired task-success comparisons._"
    cols = ["comparison", "scenarios", "runs", "repeats/scenario", "valid pairs",
            "discordant", "n_pairs", "p"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append(
            f"| full vs {r.get('other')} | {_fmt(r.get('scenario_count'))} | "
            f"{_fmt(r.get('total_run_count'))} | {_fmt(r.get('repeats_per_scenario'))} | "
            f"{_fmt(r.get('valid_pairs'))} | {_fmt(r.get('discordant_pairs'))} | "
            f"{_fmt(r.get('n_pairs'))} | {_fmt_p(r.get('p_value'))} |")
    return "\n".join(lines)


# ------------------------------------------------------------ relevance source
#: ``relevance_source.selected`` values written by ``cli.run_pipeline``.
SOURCE_ORACLE = "automatic_oracle"
SOURCE_HUMAN = "human_adjudicated"

#: ``adjudication_source`` value that means the human gold came from a real adjudication
#: rather than the legacy rounded-mean heuristic
#: (:data:`jobrec_eval.annotation.ADJUDICATION_COLUMN`).
_ADJUDICATED_COLUMN_SOURCE = "adjudicated_column"

#: Metrics that are derived from the relevance label table, i.e. the ones whose values
#: depend on which relevance source was selected.
_RELEVANCE_METRIC_NAMES = "NDCG@5 / Precision@5 / Mean Graded Relevance"


def _relevance_info(data: dict) -> dict:
    """The relevance-source block, defaulted for a data dict written without one.

    A report rendered from an older data dict (or a fixture that predates the flag)
    describes the automatic oracle with no human labels, which is exactly what such a run
    was, so the defaults keep the prose truthful rather than absent.
    """
    info = dict(data.get("relevance_source") or {})
    info.setdefault("selected", SOURCE_ORACLE)
    info.setdefault("oracle_version", data.get("oracle_version"))
    info.setdefault("human_labels_available", False)
    info.setdefault("human_labels", None)
    info.setdefault("retrieval_labels", SOURCE_ORACLE)
    return info


def _uses_human_relevance(data: dict) -> bool:
    return _relevance_info(data)["selected"] == SOURCE_HUMAN


def _human_label_provenance(info: dict) -> str:
    """Which human label file produced the numbers: path, content hash and pair counts."""
    prov = info.get("human_labels") or {}
    if not prov:
        return "no human relevance labels were available for this run"
    digest = str(prov.get("sha256") or "")
    parts = [f"`{prov.get('path')}`"]
    if digest:
        parts.append(f"sha256 `{digest[:12]}`")
    parts.append(f"{prov.get('graded_pairs')} judged (scenario, job) pair(s) over "
                 f"{prov.get('scenarios')} scenario(s)")
    parts.append(f"{prov.get('adjudicated_pairs')} adjudicated, "
                 f"{prov.get('rater_concordant_pairs')} rater-concordant, "
                 f"{prov.get('unadjudicated_disagreements_dropped')} unadjudicated "
                 "disagreement(s) excluded")
    return "; ".join(parts)


def _human_label_reference(info: dict) -> str:
    """Short identification of the human label file, for the repeated inline mentions.

    The full provenance (absolute path and every count) is stated once in the header, once
    in §4 and once in the appendix; the inline mentions carry the file name, the content
    hash and the judged-pair count, which is enough to tell two label sets apart.
    """
    prov = info.get("human_labels") or {}
    name = Path(str(prov.get("path") or "relevance_labels_human.csv")).name
    digest = str(prov.get("sha256") or "")
    parts = [f"`{name}`"]
    if digest:
        parts.append(f"sha256 `{digest[:12]}`")
    parts.append(f"{prov.get('graded_pairs')} judged pair(s)")
    return ", ".join(parts)


def _relevance_source_line(data: dict) -> str:
    """One sentence naming the source behind every grade-derived metric on the page."""
    info = _relevance_info(data)
    if info["selected"] == SOURCE_HUMAN:
        return (f"Relevance source for {_RELEVANCE_METRIC_NAMES}: **adjudicated human "
                f"labels** ({_human_label_reference(info)}). The automatic oracle (version "
                f"{info.get('oracle_version')}) is reported alongside them in §5.5.")
    line = (f"Relevance source for {_RELEVANCE_METRIC_NAMES}: **automatic oracle** "
            f"(version {info.get('oracle_version')}), not human raters.")
    if info["human_labels_available"]:
        line += (" Adjudicated human labels are available for this run and are reported "
                 "side by side in §5.5, but they did not produce the numbers here.")
    return line


def _relevance_header_note(data: dict) -> str:
    """The header's relevance disclaimer, as a blockquote, correct in both modes.

    The disclaimer used to assert unconditionally that relevance is scored by an automatic
    oracle and that human annotation is future work. Under ``--relevance-source human``
    both halves are false, so the paragraph is generated from what the pipeline actually
    used.
    """
    info = _relevance_info(data)
    if info["selected"] == SOURCE_HUMAN:
        lines = [
            "**Relevance is scored by human relevance labels from two raters after "
            "adjudication, not by the automatic oracle.**",
            f"{_RELEVANCE_METRIC_NAMES} are recomputed from the adjudicated human gold",
            f"({_human_label_provenance(info)}).",
            "The deterministic automatic oracle is still computed and the two are compared",
            "side by side in §5.5. Explanation grounding uses the system's claim validator.",
            "Inter-rater and oracle-vs-human agreement are reported in §4; the remaining",
            "construct limits are in §12.",
        ]
    else:
        lines = [
            "**Relevance is scored by a deterministic automatic oracle, not human "
            "raters.**",
            f"{_RELEVANCE_METRIC_NAMES} therefore measure agreement with a versioned",
            "canonical reference, built by `oracle_reference.py` and graded by",
            "`relevance.py`. The reference is frozen with its own fingerprint and is",
            "independent of the experiment variants and of stochastic repeats, so every",
            "condition and both backends are scored against one yardstick. It is NOT",
            "independent of the system itself: its constraint values and hard/soft",
            "strengths are still produced by the system's own deterministic extraction,",
            "so the oracle remains a transparent proxy rather than ground truth (§12).",
            "Explanation grounding uses the system's claim validator.",
        ]
        lines.append(
            "Adjudicated human labels are available for this run: §4 reports rater "
            "agreement and §5.5 compares the oracle against them, but the numbers below "
            "are the oracle's."
            if info["human_labels_available"] else
            "Human annotation and inter-rater agreement are left as future work (see §4, "
            "§12).")
    return "\n".join(f"> {line}" for line in lines)


def _relevance_comparison_table(rows: list[dict]) -> str:
    """The oracle-vs-human comparison table: one row per variant x ranking metric."""
    if not rows:
        return "_No relevance-source comparison was produced for this run._"
    cols = ["variant", "metric", "oracle", "human", "Δ (human − oracle)", "n(oracle)",
            "n(human)"]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in rows:
        lines.append(
            f"| {r.get('variant')} | {r.get('metric')} | {_fmt(r.get('oracle'))} | "
            f"{_fmt(r.get('human'))} | {_fmt(r.get('delta'))} | "
            f"{_fmt(_count(r.get('n_oracle')))} | {_fmt(_count(r.get('n_human')))} |")
    return "\n".join(lines)


def _relevance_comparison_section(data: dict) -> str:
    """§5.5: the two label sources side by side, with the caveats that make Δ readable."""
    info = _relevance_info(data)
    rows = data.get("relevance_source_comparison", [])
    out = [_relevance_source_line(data), "",
           "One row per variant x ranking metric: `oracle` is the variant mean under the "
           "automatic oracle, `human` the variant mean under the adjudicated human labels, "
           "`Δ (human − oracle)` the human value minus the oracle value, and the two `n` "
           "columns the scenario counts behind each mean. The primary results in §5-§7 are "
           "the "
           + ("**human** column." if info["selected"] == SOURCE_HUMAN
              else "**oracle** column."),
           "",
           _relevance_comparison_table(rows)]
    if not info["human_labels_available"]:
        out += ["",
                "The `human` and Δ cells are empty because no adjudicated human labels "
                "were available for this run (searched "
                f"`{info.get('human_label_search_path', 'evaluation/data/relevance_labels_human.csv')}`). "
                "Nothing is imputed in their place: an empty cell means unmeasured, and "
                "requesting `--relevance-source human` in this state fails the run rather "
                "than reporting the oracle under a human heading."]
    else:
        out += ["",
                "Two caveats keep the Δ column readable. First, the human labels cover the "
                "(scenario, job) pairs that were actually returned, so under human labels "
                "the ideal DCG of NDCG@5 is computed over that judged pool, while the "
                "oracle grades the whole catalog; the Δ therefore mixes a change of label "
                "source with a change of label universe and is read as an agreement "
                "diagnostic, not as an unbiased effect estimate. Second, unadjudicated "
                "rater disagreements are excluded from the human gold rather than averaged, "
                "so the human column is computed over the adjudicated subset only (counts "
                "above and in §4)."]
    out += ["",
            "Retrieval recall (`metrics/retrieval_metrics.csv`) is computed against the "
            "**automatic oracle in both modes**, on purpose: recall needs the full-catalog "
            "label universe, and human judgements exist only for returned pairs, which "
            "would make recall@pool trivially near 1.000. This is recorded as "
            "`retrieval_recall_relevance_source` in `manifests/analysis_plan.yaml`.",
            "",
            "Source: `metrics/relevance_source_comparison.csv`."]
    return "\n".join(out)


def _adjudication_note(agreement: dict | None) -> str:
    """How the human gold was resolved, so a heuristic is never read as adjudication."""
    if not agreement:
        return ""
    source = agreement.get("adjudication_source")
    unadjudicated = _count(agreement.get("unadjudicated_disagreements")) or 0
    if source == _ADJUDICATED_COLUMN_SOURCE:
        note = (f" Human gold: {_fmt(_count(agreement.get('n_adjudicated')))} adjudicated "
                f"row(s) plus {_fmt(_count(agreement.get('n_rater_concordant')))} row(s) "
                "the two raters already agreed on")
        note += (f"; **{unadjudicated} disagreement(s) remain unadjudicated** and are "
                 "excluded from the gold, never averaged."
                 if unadjudicated else "; no disagreement is left unadjudicated.")
        return note
    return (f" **Adjudication is incomplete for this file** (`adjudication_source="
            f"{source}`): it carries no `adjudicated` column, so the "
            f"{unadjudicated} rater disagreement(s) were resolved by the legacy "
            "rounded-mean heuristic. That heuristic is a fallback for the agreement "
            "figures only; it is never used to build the label table behind the ranking "
            "metrics.")


def _reliability_section(data: dict) -> str:
    info = _relevance_info(data)
    rel = data.get("relevance_agreement")
    clm = data.get("claim_agreement")
    # The "no human raters" statement is only made when no human annotation reached this
    # run at all -- neither an agreement computation nor an adjudicated label file.
    if not rel and not clm and not info["human_labels_available"]:
        return (
            "No human raters were used in this run. Relevance uses an automatic "
            f"oracle (version {data['oracle_version']}); grounding uses the claim "
            "validator. Inter-rater agreement is therefore **not reported** and is "
            "flagged as a construct-validity threat. Annotation templates are emitted "
            "under `annotation/` (relevance_template.csv, claim_template.csv); drop in "
            "`relevance_labels_human.csv` / `claim_annotations_human.csv` (with the "
            "`adjudicated` column filled in) to compute weighted Cohen's kappa, "
            "oracle-vs-human agreement and human-labelled ranking metrics automatically.")
    if _uses_human_relevance(data):
        lead = f"Human relevance labels were used in this run: {_human_label_provenance(info)}."
    elif info["human_labels_available"]:
        lead = ("Human annotation files are present for this run: "
                f"{_human_label_provenance(info)}.")
    else:
        lead = ("Human annotation files are present for this run, but no adjudicated "
                "relevance label table was loaded from them.")
    out = [f"{lead} {_relevance_source_line(data)}", ""]
    if not rel and not clm:
        out.append("- No rater-agreement figures were computed for this run, so "
                   "inter-rater reliability is **not reported** here; the label file's "
                   "provenance is above and in §5.5.")
    if rel:
        out.append(f"- Relevance ({rel['n_items']} items): raw rater agreement "
                   f"{_fmt(rel['raw_agreement_raters'])}, weighted Cohen's kappa "
                   f"{_fmt(rel['weighted_kappa_raters'])}; oracle-vs-human weighted kappa "
                   f"{_fmt(rel['oracle_vs_human_weighted_kappa'])} over "
                   f"{_fmt(_count(rel.get('n_gold_items')))} gold item(s)."
                   + _adjudication_note(rel))
    if clm:
        out.append(f"- Claims ({clm['n_items']} items): raw agreement "
                   f"{_fmt(clm['raw_agreement'])}, Cohen's kappa {_fmt(clm['cohens_kappa'])}"
                   + (f"; validator-vs-human kappa {_fmt(clm.get('validator_vs_human_kappa'))}."
                      if clm.get('validator_vs_human_kappa') is not None else ".")
                   + _adjudication_note(clm))
    return "\n".join(out)


def _construct_threat_bullet(data: dict) -> str:
    """The §12 construct bullet, which must not deny human raters when they were used."""
    info = _relevance_info(data)
    if info["selected"] == SOURCE_HUMAN:
        rel = data.get("relevance_agreement") or {}
        unadjudicated = _count(rel.get("unadjudicated_disagreements")) or 0
        tail = (f"and {unadjudicated} rater disagreement(s) are excluded from the gold "
                "rather than resolved" if unadjudicated else
                "while no rater disagreement was left unadjudicated")
        return (
            "- **Construct:** relevance uses human labels from two raters after "
            "adjudication (§4 reports rater and oracle-vs-human agreement), so NDCG/P@5 "
            "measure agreement with that adjudicated human gold. Only the returned "
            "(scenario, job) pairs were judged, so the ideal DCG is computed over the "
            f"judged pool rather than the whole catalog (§5.5), {tail}. Grounding measures "
            "evidence support, not perceived explanation quality or user trust.")
    bullet = (
        "- **Construct:** relevance uses an automatic oracle, not human judgement; NDCG/P@5\n"
        "  measure agreement with a rule-based reference. That reference is canonical --\n"
        "  frozen, fingerprinted, and identical across variants, repeats and backends, so\n"
        "  no condition is graded against its own output and no repeat decides the labels\n"
        "  -- but it is derived by the system's OWN deterministic extraction, so the\n"
        "  hard/soft strength of a stated constraint is the extractor's reading of the\n"
        "  utterance rather than an independently declared truth. The residual dependency\n"
        "  is therefore stabilised, not eliminated: a metric here measures agreement with\n"
        "  that reading. Grounding measures evidence support, not perceived explanation\n"
        "  quality or user trust.")
    if info["human_labels_available"]:
        bullet += ("\n  Adjudicated human labels exist for this run and are compared against\n"
                   "  the oracle in §5.5, but the reported ranking metrics are the oracle's.")
    return bullet


def _is_deterministic(data: dict) -> bool:
    """Whether this analysis describes the deterministic (model-free) backend."""
    return str(data.get("llm_mode") or "deterministic").lower() == "deterministic"


def _internal_threat_bullet(data: dict) -> str:
    """The §12 internal-validity bullet, phrased for the backend that actually ran.

    It used to assert a "deterministic mock provider" unconditionally, so the hybrid
    report claimed the run "does not exercise a real model" while it had just spent
    hundreds of real model calls -- a false limitation, and one that would have made the
    hybrid experiment look pointless.
    """
    if _is_deterministic(data):
        return ("- **Internal:** the deterministic mock provider removes LLM stochasticity\n"
                "  but also does not exercise a real model; variant behaviour is controlled\n"
                "  by feature flags on one code path.")
    identities = list(data.get("model_call_identities") or [])
    models = ", ".join(f"`{r.get('model')}`" for r in identities) or "the configured model"
    return (f"- **Internal:** a real model backend ({data.get('llm_mode')}, {models}) is\n"
            "  exercised, so responses are stochastic: repeats measure that variance rather\n"
            "  than adding independent samples, and a single repeat cannot be read as the\n"
            "  system's behaviour. Variant behaviour is controlled by feature flags on one\n"
            "  code path, and every model call is recorded so the run is replayable.")


def _conclusion_opening(data: dict) -> str:
    """The §13 opening clause, which must not call a real-model run deterministic."""
    if _is_deterministic(data):
        return "Within this controlled, deterministic setup"
    return (f"Within this controlled setup on a real model backend "
            f"({data.get('llm_mode')})")


def _next_steps_sentence(data: dict) -> str:
    """The conclusion's next-steps sentence, which must not ask for what was already done."""
    human = _uses_human_relevance(data)
    deterministic = _is_deterministic(data)
    if human and deterministic:
        return ("Claims are limited to this configuration; a real LLM backend and a larger "
                "human-judged label pool (the current judgements cover the returned pairs "
                "only) are the natural next steps.")
    if human:
        return ("Claims are limited to this configuration; a larger human-judged label pool "
                "(the current judgements cover the returned pairs only) is the natural next "
                "step.")
    if deterministic:
        return ("Claims are limited to this configuration; human-annotated relevance and a "
                "real LLM backend are the natural next steps.")
    # Real backend already exercised: asking for it again would deny what this run did.
    return ("Claims are limited to this configuration; human-annotated relevance is the "
            "natural next step, the real-model backend having already been exercised here.")


def _relevance_label_appendix(data: dict) -> str:
    """Appendix pointers to the label tables that actually exist for this run."""
    info = _relevance_info(data)
    lines = [f"- Relevance labels (automatic oracle, version {info.get('oracle_version')}): "
             "`../normalized/relevance_labels.csv`"]
    if info["human_labels_available"]:
        prov = info.get("human_labels") or {}
        lines.append(
            "- Relevance labels (adjudicated human, as consumed): "
            "`../normalized/relevance_labels_human.csv`; source file "
            f"`{prov.get('path')}` (sha256 `{str(prov.get('sha256') or '')[:12]}`)")
    lines.append("- Oracle vs human comparison: "
                 "`metrics/relevance_source_comparison.csv`")
    lines.append(f"- Relevance source used for the reported ranking metrics: "
                 f"`{info['selected']}`; retrieval recall labels: "
                 f"`{info['retrieval_labels']}`")
    return "\n".join(lines)


def _md_table(rows: list[dict], columns: list[tuple[str, str]],
              empty: str, nd: int = 3) -> str:
    """A markdown table of ``rows`` restricted to ``columns`` as ``(key, header)`` pairs."""
    if not rows:
        return empty
    head = "| " + " | ".join(header for _key, header in columns) + " |"
    sep = "|" + "---|" * len(columns)
    lines = [head, sep]
    for row in rows:
        lines.append("| " + " | ".join(
            _fmt(row.get(key), nd) for key, _header in columns) + " |")
    return "\n".join(lines)


#: Columns rendered for each of the three pipeline-stage tables. These metrics were
#: computed and written to ``metrics/*.csv`` on every run but appeared NOWHERE in the
#: report, so the retrieval stage, the ranking features' contributions and the
#: rule-vs-model extraction split were all invisible to a reader of the analysis.
_RETRIEVAL_COLUMNS = [
    ("variant", "variant"), ("retrieval_runs", "runs w/ retrieval"),
    ("mean_initial_pool_size", "mean recalled"), ("mean_pool_size", "mean pool"),
    ("mean_retrieval_score", "mean score"), ("fallback_rate", "full-catalog fallback"),
    ("median_retrieval_latency_ms", "median lat(ms)"),
    ("recall_at_pool", "recall@pool"), ("relevant_job_coverage", "relevant coverage"),
    ("retrieval_error_rate", "retr err"), ("ranking_error_rate", "rank err"),
]

_TOPK_COLUMNS = [
    ("feature", "ranking feature"), ("mean_weight", "mean weight"),
    ("mean_normalized_score", "mean norm score"),
    ("mean_contribution", "mean contribution"),
    ("contribution_share", "share of total"), ("inactive_jobs", "inactive jobs"),
    ("dominant_explanation_code", "dominant explanation"),
]

_EXTRACTION_COLUMNS = [
    ("variant", "variant"), ("fields", "fields"), ("rule_share", "rule share"),
    ("llm_share", "model share"), ("repaired_fields", "repaired"),
    ("rule_fallback_fields", "rule fallback"), ("unresolved_fields", "unresolved"),
    ("schema_failure_rate", "schema failure"), ("fallback_rate", "fallback rate"),
]


def _pipeline_stage_sections(data: dict) -> str:
    """Render the retrieval / top-k / extraction-source tables (§5.6).

    All three come from tables the pipeline already wrote; only the rendering was
    missing. The top-k table is shown for the ``full`` variant at variant scope, because
    the per-rank breakdown is five times longer and adds nothing a reader can act on --
    the CSV carries it.
    """
    retrieval = list(data.get("retrieval_metrics") or [])
    topk_all = list(data.get("topk_contribution") or [])
    extraction = [r for r in (data.get("extraction_source_metrics") or [])
                  if str(r.get("scope")) == "variant"]

    topk = [r for r in topk_all
            if str(r.get("scope")) == "variant" and str(r.get("variant")) == "full"]
    topk.sort(key=lambda r: (r.get("contribution_share") or 0.0), reverse=True)

    parts = [
        "**Retrieval layer.** Pool sizes, retrieval scores and the empty-recall "
        "fallback, per variant. `recall@pool` and `relevant coverage` are computed "
        "against the automatic-oracle label universe in both relevance modes; they read "
        "`N/A` when no labelled scenario returned a pool to score.",
        "",
        _md_table(retrieval, _RETRIEVAL_COLUMNS,
                  "_No retrieval metrics available._"),
        "",
        "Source: `metrics/retrieval_metrics.csv`.",
        "",
        "**Ranking-feature contributions (variant `full`, top-k results).** `share of "
        "total` is the feature's mean contribution divided by the mean total score, so "
        "a feature with weight 0 or no applicable jobs contributes 0 by construction "
        "rather than by failing.",
        "",
        _md_table(topk, _TOPK_COLUMNS,
                  "_No top-k contribution data available._"),
        "",
        "Source: `metrics/topk_contribution.csv` (also carries the per-rank breakdown "
        "and the other variants).",
        "",
        "**Extraction source.** How each variant's constraint fields were obtained. "
        "Under the deterministic backend `rule share` is 1.0 by construction and the "
        "model columns are 0; under a hybrid backend this table is where the model's "
        "share, its schema-repair rate and its fallbacks to the rule extractor are read.",
        "",
        _md_table(extraction, _EXTRACTION_COLUMNS,
                  "_No extraction-source metrics available._"),
        "",
        "Source: `metrics/extraction_source_metrics.csv` (also carries the "
        "per-scenario-type scope).",
    ]
    return "\n".join(parts)


def _completeness_line(data: dict) -> str:
    """A loud §1 bullet when the experiment is short of the runs it planned.

    A crashed run leaves no bundle, so it is simply absent from every table -- the
    aggregate silently describes fewer runs than the design. The count comes from the
    runner's manifest, so this cannot be forgotten by whoever reads the tables.
    """
    exp = data.get("experiment") or {}
    crashed = _count(exp.get("crashed_run_count")) or 0
    actual = _count(exp.get("run_count"))
    expected = _count(exp.get("expected_run_count"))
    if expected is None:
        # A manifest written before the runner recorded it (an analysis re-run over older
        # bundles) still has the design, so derive the count rather than printing "n/a".
        variants = exp.get("variants") or []
        scenarios = _count(exp.get("scenario_count"))
        repeats = _count(exp.get("repeat_count"))
        if variants and scenarios and repeats:
            expected = len(variants) * scenarios * repeats
    if expected is None or actual is None:
        return (f"- Completeness: {actual if actual is not None else 'an unknown number of'} "
                f"runs are analysed; the planned run count could not be determined from "
                f"the manifest, so completeness is NOT established here.")
    if not crashed and expected == actual:
        return ("- Completeness: every planned run produced a bundle "
                f"({actual} of {expected}).")
    return (f"- **INCOMPLETE EXPERIMENT: {crashed} run(s) crashed and produced no "
            f"bundle**, so {actual} of {expected} planned runs are analysed below. Every "
            f"aggregate is computed over the runs that survived; see `failures.csv` in "
            f"the run-bundle directory for the affected variant/scenario pairs. Do not "
            f"cite these figures as covering the full design.")


def _backend_identity_line(data: dict) -> str:
    """Provider and the model(s) that actually answered, kept distinct (§1).

    The report used to print the configured PROVIDER under the label "model", so a
    hybrid run appeared to have used a model called ``remote``. The provider is a
    transport chosen by config; the model name only exists in the recorded calls, so it
    is reported from there or not at all.
    """
    provider = data.get("llm_provider") or data.get("llm_model") or "unknown"
    identities = list(data.get("model_call_identities") or [])
    if not identities:
        return (f"provider `{provider}`, no model calls recorded (the deterministic "
                f"backend makes none, so no model answered any request);")
    parts = []
    for row in identities:
        failed = row.get("failed_calls") or 0
        suffix = f", {failed} failed" if failed else ""
        parts.append(f"`{row.get('model')}` via `{row.get('provider')}` "
                     f"({row.get('calls')} calls{suffix})")
    return (f"provider `{provider}`; model(s) actually used, read from "
            f"`model_calls.jsonl`: " + "; ".join(parts) + ";")


def _error_taxonomy_table(rows: list[dict]) -> str:
    if not rows:
        return "_No task failures to categorize._"
    lines = ["| error category | count | % | most-affected variant |",
             "|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['error_category']} | {r['count']} | {r['percentage']} | {r['most_affected_variant']} |")
    return "\n".join(lines)


def _col_mean(group: pd.DataFrame, column: str) -> float | None:
    """Mean of ``column`` in ``group``, or ``None`` when the column/values are absent.

    Keeps the scenario-type table renderable over a frame that predates a column (the
    process measures below) instead of raising a ``KeyError``.
    """
    if column not in group.columns:
        return None
    vals = group[column].dropna()
    return float(vals.mean()) if len(vals) else None


def _scenario_type_table(sv: pd.DataFrame) -> str:
    """Scenario-type x variant summary, including the two process measures (R7.4).

    ``Turns`` (median-relevant mean of ``response_turns``) and ``ClarEff`` (mean
    ``clarification_efficiency``) are reported next to task success on purpose: neither
    is interpretable on its own.

    Every variant in the frame is rendered. The table used to filter to
    ``full``/``no_memory``/``no_context``, which dropped both dialogue baselines from the
    per-scenario-type breakdown — including the clarification rows, where their behaviour
    is most visible — with no note anywhere in the report. Rows are grouped by
    ``scenario_type`` and ordered within each group by :data:`_VARIANT_ORDER`.
    """
    lines = ["| scenario_type | variant | NDCG@5 | HCSR | TaskSucc | Grounding | "
             "Turns | ClarEff | n |",
             "|---|---|---|---|---|---|---|---|---|"]
    groups: dict[tuple[str, str], pd.DataFrame] = {}
    for (stype, variant), g in sv.groupby(["scenario_type", "variant"]):
        groups[(str(stype), str(variant))] = g
    order = _ordered_variants(variant for _, variant in groups)
    for stype in sorted({s for s, _ in groups}):
        for variant in order:
            g = groups.get((stype, variant))
            if g is None:
                continue
            lines.append(
                f"| {stype} | {variant} | {_fmt(g['ndcg_at_5'].mean())} | "
                f"{_fmt(g['hcsr'].mean())} | {_fmt(g['task_success'].mean())} | "
                f"{_fmt(g['grounding'].mean())} | "
                f"{_fmt(_col_mean(g, 'response_turns'), 2)} | "
                f"{_fmt(_col_mean(g, 'clarification_efficiency'), 2)} | "
                f"{g['scenario_id'].nunique()} |")
    return "\n".join(lines)


def generate_markdown(data: dict, plots_rel: str = "../plots") -> str:
    exp = data["experiment"]
    vs = pd.DataFrame(data["variant_summary"])
    mem = pd.DataFrame(data["memory_contribution"])
    ctx = pd.DataFrame(data["context_contribution"])
    sv = pd.DataFrame(data["scenario_variant"])
    overall = data["overall_comparisons"]

    def ocmp(metric, other):
        for r in overall:
            if r["metric"] == metric and r["other"] == other:
                return r
        return None

    full = vs[vs["variant"] == "full"].iloc[0] if (vs["variant"] == "full").any() else {}

    md = f"""# Evaluation Report: CMJCC Conversational Job Recommendation

> Generated by `jobrec_eval`. Every number is reproducible from the run bundles under
> {_run_bundle_pointer(exp['experiment_id'])} and the tables under `metrics/` and
> `statistics/` inside this directory. The run-bundle tree is a sibling of this analysis
> output directory `{exp['experiment_id']}/`: both live under the pipeline's `--out-root`.
>
{_relevance_header_note(data)}

## 1. Executive Summary

- Experiment `{exp['experiment_id']}` — {exp['scenario_count']} scenarios ×
  {len(exp['variants'])} variants × {exp['repeat_count']} repeat(s) =
  {exp['run_count']} runs. Reference date {exp['reference_date']}.
- Run mode: **{data.get('llm_mode', 'deterministic')}**; {_backend_identity_line(data)}
  config/catalog/prompt hashes frozen.
{_completeness_line(data)}
- Headline (full variant, scenario-mean, with the denominator `n` each mean was taken
  over — they differ between metrics, see §5): {_headline_metrics(full)}.
- {_relevance_source_line(data)}
- Ablation direction: memory and job-context removal are compared against full
  below with paired bootstrap CIs; small n means results are framed as observed
  differences with uncertainty, not proofs.

## 2. Research Questions and Evaluation Design

RQ4 is decomposed into job-match relevance, hard-constraint satisfaction, task
success, memory contribution, job-context contribution, agent-handoff success,
explanation grounding, response turns and latency.
{_variant_count_phrase(exp['variants'])} run over a frozen scenario set and
catalog snapshot with fixed seeds. Analysis unit is the scenario; metrics are averaged
over repeats before pairing.

## 3. Dataset and Scenario Set

- Catalog snapshot `{exp['catalog_snapshot_id']}`, hash `{exp['catalog_hash'][:12]}`.
- Scenario counts by type: {_counts_phrase(data['scenario_type_counts'])}.
- Memory-dependent (>=medium) scenarios: {data['n_memory_dependent']};
  context-dependent (high) scenarios: {data['n_context_dependent']}.

### 3.1 Data quality of the frozen inputs

{_data_quality_section(data.get('data_quality'))}

## 4. Annotation Reliability

{_reliability_section(data)}

## 5. Overall Results

{_relevance_source_line(data)}

Variant summary (scenario-mean of each metric):

{_variant_table(vs)}

Scenario denominator behind each mean above (`<metric>_n` in
`metrics/variant_summary.csv`; an empty cell means the frame carries no denominator for
that metric):

{_variant_denominator_table(vs)}

**Read the denominators before the means.** The ranking metrics — NDCG@5, P@5, HCSR and
the graded-relevance mean — are averaged over the scenarios where the variant actually
returned a ranked list, and over those only. A variant that abandons a dialogue or
declines to answer returns no list on those scenarios, so it carries a SMALLER
denominator, and the scenarios it loses are exactly the hardest ones: the ambiguous
cases whose clarification it never resolved. A higher ranking mean on a smaller
denominator is therefore survivorship, NOT better ranking, and must never be read as
that variant out-ranking `full`. **`grounding` carries the same caveat and for the same
reason:** it is averaged only over the runs that actually emitted a factual claim, and a
run that returns no recommendation registers no claims, so it has nothing to ground and
drops out of that denominator too. A perfect grounding value on a reduced denominator
therefore means FEWER claims were checked, NOT better-grounded explanations, and it is
never evidence that the variant explains itself as well as `full` does. `task_success` is
the metric defined on every scenario — an abandoned or declined scenario counts as a
failure there instead of dropping out — so `task_success` is the column to compare across
variants; the ranking columns are comparable only between variants whose `n` agree.

### 5.x Full vs dialogue baselines (relevance & task success)

{_baseline_delta_bullets(overall)}

`n` is the number of scenarios paired between the two variants, i.e. the scenarios on
which both returned a comparable value; a scenario the other variant abandoned forms no
pair and enters no Δ. A near-zero Δ against a variant whose ranking denominator is
reduced is therefore "not estimable on the scenarios that variant abandoned", not
evidence of equivalence.

![NDCG]({plots_rel}/ndcg_by_variant.png)
![HCSR]({plots_rel}/hcsr_by_variant.png)
![Task success]({plots_rel}/task_success_by_variant.png)
![Grounding]({plots_rel}/grounding_by_variant.png)

### 5.2 Per-constraint compliance (recommended jobs vs authoritative hard constraints)

{_compliance_table(data.get('constraint_compliance', []))}

Each cell is `rate (n=applicable)`, where `n` is the number of (recommended job,
constraint) pairs the check could be applied to. **The denominators differ by an order of
magnitude between variants**, because a variant that returns fewer recommendations offers
fewer pairs to check — so a perfect rate on a small `n` is not evidence of constraint
enforcement across the run, and the columns are only comparable at comparable `n`. Where a
cell also shows `unk`, that share of the denominator had a constraint value that could not
be determined; unknowns are counted as non-compliant in the denominator, so a large `unk`
means the rate is driven by missing data rather than by observed violations. Source:
`metrics/constraint_compliance.csv` (`pass`, `fail`, `unknown`, `applicable`,
`unknown_rate` per row).

### 5.3 No-match and clarification correctness

No-match precision / recall / F1 by variant:

{_pr_table(data.get('no_match_metrics', []), [('precision','Precision'), ('recall','Recall'), ('f1','F1'), ('true_no_match','TP'), ('no_match_expected','Expected')])}

Clarification precision / recall by variant:

{_pr_table(data.get('clarification_metrics', []), [('precision','Precision'), ('recall','Recall'), ('useful','Useful'), ('expected_clarification','Expected')])}

**Unit of the counts in both tables above: RUNS, not scenarios.** `TP`, `Expected` and
`Useful` count runs, so with `r` repeats per scenario an expectation held by `s` scenarios
appears as `s × r`. That is why an `Expected` of 126 over 42 scenarios is three repeats of
each, not 126 distinct scenarios — the precision and recall values themselves are
unaffected (numerator and denominator scale together), but the counts must not be read as
scenario evidence. Every statistical test in §6 and §8 pairs at the SCENARIO level
instead; see the analysis-unit note in §8.

### 5.4 Clarification efficiency and response-turn distribution

`response_turns` is interpreted **jointly with task success, never on its own**: asking
fewer questions while returning the wrong answer is NOT efficiency. The
`clarification_efficiency` score encodes exactly that — a run that skipped a *necessary*
clarification carries a skip penalty that dominates any turn-count or wasted-ask
advantage, so a short dialogue that guessed can never out-score a longer one that asked
the question it needed. Abandoning a dialogue *after* asking is likewise not efficiency:
a run that asked the necessary question but ended with it still pending (continuation
disabled, turn cap reached, the user unable to answer, or the repeated-slot guard) never
resolved the ambiguity it identified, and the score now reflects that with a separate
unresolved-dialogue penalty. The ordering the score implements is therefore *asked and
resolved* > *asked and abandoned* > *necessary clarification skipped*, and a shorter
dialogue cannot move a variant up that ordering. Read the turn columns below against the
Task Success column of §5 and §7, and read a low turn count as efficient only where task
success held up. Scores are on a penalty scale (higher = more efficient; the magnitudes
are not probabilities).

`NecRecall` is necessary clarification recall, `UnnecAsked` the unnecessary-ask count,
`NecMissed` the necessary clarifications that were skipped, `RepeatGuard` the runs where
the repeated-slot guard activated, `Abandoned` the runs that asked a clarification and
then ended with the question still pending, and `AnsweredRate` the share of asking runs
whose question the simulated user answered. A variant with a non-zero `Abandoned` count
is not to be read as efficient on the strength of `EffScore` alone.

{_clarification_efficiency_table(data.get('clarification_efficiency', []), data.get('clarification_metrics', []))}

Sources: `metrics/clarification_efficiency.csv` (turn distribution, ask classification,
score), `metrics/clarification_metrics.csv` (recall, repeated-slot activations),
`metrics/run_metrics.csv` (`response_turns` and `clarification_efficiency` per run),
`metrics/variant_summary.csv` and `metrics/scenario_type_summary.csv` (aggregates).

### 5.5 Relevance source: automatic oracle vs adjudicated human labels

{_relevance_comparison_section(data)}

### 5.6 Retrieval layer, top-k sensitivity and extraction source

{_pipeline_stage_sections(data)}

## 6. Ablation Analysis

Each ablation isolates a single framework mechanism (candidate memory or
job-context orchestration) while holding catalog, scenarios, prompts, model
settings, top-k, pool size, ranking weights and seeds fixed across the compared
variants. Every Δ reported below is therefore read as the **contribution of that
framework mechanism under the controlled prototype instantiation** — an
attribution to a specific mechanism as instantiated in this prototype, not a
general property of the mechanism and not a claim of superiority over any
external framework.

{_relevance_source_line(data)}

### 6.1 Memory Contribution: Full vs No-Memory

Δmemory(M) = M_full − M_no_memory, paired by scenario. Primary subset is
memory-dependent (multi-turn) scenarios. Each Δmemory is framed as the
candidate-memory mechanism's contribution under the controlled prototype
instantiation, not as evidence of comprehensive superiority over external
frameworks.

{_contrib_table(mem, 'memory_dependent')}

All scenarios:

{_contrib_table(mem, 'all')}

![Memory delta NDCG]({plots_rel}/memory_delta_ndcg_at_5.png)
![Memory delta task]({plots_rel}/memory_delta_task_success.png)
![Memory delta HCSR]({plots_rel}/memory_delta_hcsr.png)

### 6.2 Job-Context Contribution: Full vs No-Context

Δcontext(M) = M_full − M_no_context, paired by scenario. Primary subset is
context-dependent (high) scenarios. HCSR/violations are computed against the
authoritative hard constraints. Each Δcontext is framed as the job-context
orchestration mechanism's contribution under the controlled prototype
instantiation, not as a claim of comprehensive superiority over external
frameworks.

{_contrib_table(ctx, 'context_dependent')}

All scenarios:

{_contrib_table(ctx, 'all')}

![Context delta HCSR]({plots_rel}/context_delta_hcsr.png)
![Context delta task]({plots_rel}/context_delta_task_success.png)
![Context delta NDCG]({plots_rel}/context_delta_ndcg_at_5.png)

### 6.3 Process-measure contributions (secondary outcome family)

Δ for the two process measures — **response turns** and **clarification efficiency** —
reported as a SEPARATE, secondary outcome family. Holm correction is applied within each
family independently (see §8), so these rows do not alter a single p-value in §6.1/§6.2:
the pre-registered primary family in `manifests/analysis_plan.yaml`
(`primary_outcomes`) is corrected over its own metrics only, and this family
(`secondary_outcomes`) over its own. Rows are tagged with a `family` column in
`metrics/memory_contribution.csv` and `metrics/context_contribution.csv`, so nothing is
counted twice.

These Δ values carry the same reading as every other Δ here: the contribution of that
framework mechanism under the controlled prototype instantiation, not a general property
of the mechanism and not a claim about any external framework. A negative Δ on response
turns means the full variant used FEWER turns; it is only a gain where task success in
§6.1/§6.2 did not fall, because fewer questions with a wrong answer is not efficiency.

Δmemory, memory-dependent scenarios:

{_contrib_table(mem, 'memory_dependent', 'secondary')}

Δmemory, all scenarios:

{_contrib_table(mem, 'all', 'secondary')}

Δcontext, context-dependent scenarios:

{_contrib_table(ctx, 'context_dependent', 'secondary')}

Δcontext, all scenarios:

{_contrib_table(ctx, 'all', 'secondary')}

## 7. Results by Scenario Type

{_relevance_source_line(data)}

{_scenario_type_table(sv)}

`n` here is the number of scenarios OF THAT TYPE that the variant ran; it is NOT the
per-metric denominator, which is per (metric, variant) and is shown in the denominator
table in §5. A metric cell reading `N/A` therefore means that variant returned nothing to
score for that metric on those scenarios — the `n` beside it still counts the scenarios,
not the values that entered the mean.

## 8. Statistical Analysis

**Analysis unit.** `scenario_id` is the independent analysis unit. `repeat_index` is used
only for stability and variance analysis and is **never** treated as an independent
sample: repeats are collapsed within each scenario before anything is paired.
Deterministic runs default to one repeat per scenario; stochastic and hybrid runs may
repeat, and repeating them cannot enlarge the sample or shrink a p-value.

**Continuous metrics.** Metrics are averaged over repeats per scenario x variant, then
paired by `scenario_id`. Paired bootstrap ({exp.get('bootstrap_iterations', 5000)}
iterations, seed {exp.get('bootstrap_seed', 2026)}) gives 95% CIs for the scenario-mean
difference; the p-value is Wilcoxon signed-rank over the scenario pairs.

**Binary task success.** McNemar operates on **scenario-level paired binary outcomes**,
not on run-level pairs: within each scenario the repeats are collapsed to one binary by
majority vote (strictly more successes than failures -> 1), with even-repeat ties
resolving conservatively to not-success. The two variants are then paired on the
scenarios present in both, and the exact binomial test is applied to the discordant
scenario pairs. `n_pairs` is therefore the number of **validly paired scenarios**, not
the number of runs.

{_pairing_provenance_table(overall)}

`runs` counts the runs on BOTH sides of the comparison (`scenarios` x
`repeats/scenario` x the two variants), so it exceeds `n_pairs` whenever repeats are
used. `n_pairs` equals `valid pairs`, which is a scenario count. `discordant` is the
number of scenario pairs the exact test is computed on.

**Effect sizes.** Cohen's dz for continuous metrics, rank-biserial for binary or
degenerate ones.

**Multiplicity.** Holm correction is applied within each ablation, within each scenario
subset, and **within each outcome family independently**: the pre-registered primary
family (§6.1/§6.2) and the secondary process-measure family (§6.3) are corrected
separately, so adding the process measures leaves every primary p-value unchanged. Both
families are recorded in `manifests/analysis_plan.yaml` as `primary_outcomes` and
`secondary_outcomes`.

With small n, a CI that includes 0 is reported as "direction observed, uncertain", never
as "no effect".

## 9. Error Analysis

{data['error_summary']}

Root-cause taxonomy of task-unsuccessful runs:

{_error_taxonomy_table(data.get('error_taxonomy', []))}

### 9.1 Representative case studies

{data.get('case_studies_md', '_No case studies extracted._')}

## 10. Fault-Injection Robustness (separate robustness experiment)

This section is **deliberately kept separate from the main experiment results in §5-§7**.
The main experiment measures the system on well-formed scenarios; this section reports
whether the system detects, rejects or recovers from deliberately malformed inputs. The
two are different experiments over different inputs and are never averaged together.

Two consequences of that separation, stated explicitly so the numbers are not
misread:

- **A grounding rate of 1.000 in the main experiment is legitimate, not a defect.** Every
  factual claim in a well-formed run resolves to registered evidence because the claim
  validator drops the ones that do not, so there is nothing left to be unsupported. The
  same holds for handoff success.
- **Failure samples are NOT mixed into the main experiment to drag a metric below
  1.000.** Doing so would corrupt the main measurement to make a robustness point. The
  evidence that ungrounded claims and invalid handoffs really are caught comes from the
  fault-injection suite below, which shows both rates strictly under 1.000 over a
  failure-containing set.

### 10.1 Failure-path rates over the loaded runs

{_failure_metrics_table(data.get('failure_metrics', []))}

{_fault_injection_prose(data.get('failure_metrics', []))}

Source: `metrics/failure_metrics.csv` (one row per rate, with the numerator and
denominator behind it).

### 10.2 Fault classes covered by the robustness suite

The fault-injection helpers live in `tests/support/fault_injection.py`; the assertions
live in `tests/unit/test_failure_paths.py` (per-case) and
`tests/integration/test_failure_metrics.py` (aggregate rates over a failure-containing
set, including a generated-set property test). The classes covered are:

| fault class | injected as | asserted outcome |
|---|---|---|
| invalid evidence id | a claim citing an id registered nowhere | claim flagged unsupported, never presented |
| missing evidence source | a claim carrying no evidence ids | claim flagged unsupported |
| wrong-field evidence | an id for a different field of the same job | does not resolve; claim rejected |
| partially grounded claim | one resolvable + one dangling id | rejected as a whole (grounding is all-or-nothing) |
| unsupported salary claim | salary text with no supporting evidence | flagged unsupported |
| unsupported location claim | location text with no supporting evidence | flagged unsupported |
| unsupported skill claim | skill text with no supporting evidence | flagged unsupported |
| schema-invalid handoff | out-of-vocabulary `status`; unknown extra field | handoff rejected, offending field named |
| missing-field handoff | each required field omitted in turn | handoff rejected, every loss reported |
| agent exception | an agent raising mid-turn | failed run with an error response holding no claims |
| timeout + retry | provider raising `LLMTimeout` for the first N calls | absorbed by the bounded retry; budget is finite |
| partial failure + recovery | timeout on the first model call | recovers on the retry, or falls back to the rule extractor; run completes |

Also asserted at the aggregate level: a run containing a non-validated handoff is never
scored a success, every supported claim resolves to a registered evidence id, and over a
failure-containing set both `grounding_rate` and `handoff_success_rate` are strictly below
1.000 while a happy-path-only set reports N/A for detection and recovery.

## 11. Discussion

- **RQ2 / memory:** differences concentrate in memory-dependent (multi-turn)
  scenarios, consistent with prior-turn memory contributing to correctly
  reconstructing the active search; effects on memory-independent scenarios are
  expected to be near zero.
- **RQ2 / job context:** removing explicit hard/soft orchestration is expected
  to lower HCSR and raise violations on context-dependent scenarios; the tables
  above quantify this against the authoritative constraints.
- **Quality/latency trade-off:** turns and latency are reported alongside task
  success rather than in isolation (see turns-vs-success and latency plots).
- **Inspectability (RQ1/RQ3):** handoff success, decision-log completeness and
  recommendation trace completeness are reported as engineering-quality
  indicators, not statistical claims.

![Turns vs success]({plots_rel}/turns_vs_success.png)
![Latency breakdown]({plots_rel}/latency_breakdown.png)

## 12. Threats to Validity

{_construct_threat_bullet(data)}
{_internal_threat_bullet(data)}
- **External:** small synthetic catalog and synthetic candidates; a modest
  scenario count; results do not extrapolate to real hiring outcomes.
- **Conclusion:** small n limits statistical power; emphasis is on effect sizes,
  CIs and per-scenario plots, not single p-values.

## 13. Conclusion

{_conclusion_opening(data)}, the full architecture meets the
engineering-quality indicators and the ablations show the expected directional
contributions of candidate memory and job-context orchestration. These results
attribute observed differences to specific framework mechanisms under the
controlled prototype instantiation; they do not state or imply comprehensive
superiority over any existing external framework. {_next_steps_sentence(data)}

## Appendix

- Experiment manifest: `manifests/experiment_manifest.json`
- Analysis plan: `manifests/analysis_plan.yaml` (`primary_outcomes` +
  `secondary_outcomes`; Holm is applied within each family)
- All metric tables: `metrics/*.csv`; statistics: `statistics/*.csv`
- Clarification efficiency + response-turn distribution:
  `metrics/clarification_efficiency.csv`
- Failure-path rates: `metrics/failure_metrics.csv`
- Data-quality report: `data_quality_report.json`
{_relevance_label_appendix(data)}
- Data lineage: `audit/data_lineage.csv`; checksums: `checksums.json`
  (verify with `python -m jobrec_eval.cli verify <output_dir>`)
"""
    return md


def _variant_of(manifest: dict) -> str | None:
    flags = manifest.get("feature_flags")
    variant = flags.get("variant") if isinstance(flags, dict) else None
    return str(getattr(variant, "value", variant)) if variant else None


def _consistency_scopes(manifests: list[dict]) -> list[tuple[set[str] | None, list[dict]]]:
    """The scopes the gate verifies: the whole comparison, then each ablation pair.

    The whole comparison is checked without a target flag set (R15.1: shared
    catalog/scenarios/prompts/settings/commit). Each ablation pair present in the
    manifests is then checked against its own mechanism's flag group, so a Δ the
    report attributes to that mechanism cannot be contaminated by another flag
    (R32.7).
    """
    scopes: list[tuple[set[str] | None, list[dict]]] = [(None, manifests)]
    by_variant: dict[str | None, list[dict]] = {}
    for manifest in manifests:
        by_variant.setdefault(_variant_of(manifest), []).append(manifest)
    for base, other, target in _ABLATION_PAIRS:
        if by_variant.get(base) and by_variant.get(other):
            scopes.append((set(target), by_variant[base] + by_variant[other]))
    return scopes


def _mirror_onto_run_records(manifest_paths: list[Path]) -> list[Path]:
    """Copy each manifest's consistency block onto its run record (R15.3).

    The manifest is the authoritative location for the gate result; mirroring it
    into the sibling ``run_record.json`` populates
    :attr:`jobrec.domain.run_record.RunRecord.consistency_flags`, so a run carries
    its own verification outcome. Unreadable or unexpected payloads are skipped
    rather than failing the gate.
    """
    written: list[Path] = []
    for manifest_path in manifest_paths:
        record_path = Path(manifest_path).with_name("run_record.json")
        try:
            manifest = json.loads(Path(manifest_path).read_text())
            record = json.loads(record_path.read_text())
        except (OSError, ValueError):
            continue
        block = manifest.get("consistency") if isinstance(manifest, dict) else None
        if not isinstance(block, dict) or not isinstance(record, dict):
            continue
        record["consistency_flags"] = {
            "consistent": block.get("consistent"),
            **(block.get("flags") or {}),
            "compared_runs": block.get("compared_runs", []),
            "mismatched_fields": block.get("mismatched_fields", []),
        }
        record_path.write_text(json.dumps(record, indent=2, default=str))
        written.append(record_path)
    return written


def require_consistent_runs(manifests: list[dict]) -> None:
    """Halt unless every compared run shares the same configuration (R15.2).

    Verification stops at the first failing scope; the resulting flags are stamped
    into the manifests, persisted to disk and mirrored onto the run records before
    the mismatch is raised, so a blocked report still leaves an auditable trail
    (R15.3). Raises :class:`~jobrec_eval.consistency.ConsistencyError` on mismatch
    and :class:`ValueError` when there is nothing to verify.

    Stamping rewrites ``run_manifest.json`` and ``run_record.json`` inside run
    bundles the runner has already checksummed, so the two rewritten entries are
    restamped in the run bundle's ``checksums.json`` (R16.1). Without this the
    experiment manifest is stale for every run the moment a report is generated
    and ``cli verify <experiment_dir>`` can never pass. Only the entries written
    here are touched; every other digest stays as the runner recorded it.
    """
    if not manifests:
        raise ValueError(
            "configuration consistency cannot be verified without run manifests; "
            "refusing to generate a comparison report")
    error: ConsistencyError | None = None
    for target, subset in _consistency_scopes(manifests):
        try:
            require_consistent(subset, target)
        except ConsistencyError as exc:
            error = exc
            break
    manifest_paths = save_run_manifests(manifests)
    record_paths = _mirror_onto_run_records(manifest_paths)
    restamp_checksums([*manifest_paths, *record_paths])
    if error is not None:
        raise error


def write_report(data: dict, out_dir: str | Path, *,
                 experiment_dir: str | Path | None = None,
                 manifests: list[dict] | None = None) -> Path:
    """Verify configuration consistency, then write the report bundle.

    Args:
        data: The assembled report data dict.
        out_dir: Analysis output directory; the report lands under ``report/``.
        experiment_dir: Directory holding the run bundles whose ``run_manifest.json``
            files describe the compared runs.
        manifests: Already-loaded run manifests, as an alternative to
            ``experiment_dir``.

    Returns:
        Path of the written Markdown report.

    Raises:
        ConsistencyError: The compared runs do not match; nothing is written
            (R15.2).
        ValueError: Neither manifest source was supplied, or no manifests exist.
    """
    if manifests is None:
        if experiment_dir is None:
            raise ValueError(
                "write_report needs experiment_dir or manifests: configuration "
                "consistency must be verified before any report output is produced")
        manifests = load_run_manifests(experiment_dir)
        if not manifests:
            raise ValueError(
                f"no run_manifest.json found under {experiment_dir}; configuration "
                "consistency of the compared runs cannot be verified")
    require_consistent_runs(manifests)

    out_dir = Path(out_dir)
    (out_dir / "report").mkdir(parents=True, exist_ok=True)
    # UTF-8 explicitly: the Markdown carries Δ and − and must not depend on the
    # platform's default encoding (locale codecs such as GBK cannot encode them).
    (out_dir / "report" / "analysis_report_data.json").write_text(
        json.dumps(data, indent=2, default=str), encoding="utf-8")
    md = generate_markdown(data)
    report_path = out_dir / "report" / "analysis_report.md"
    report_path.write_text(md, encoding="utf-8")
    return report_path
