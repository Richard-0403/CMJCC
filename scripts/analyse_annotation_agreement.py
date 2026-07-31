"""Agreement analysis over the human annotation, written into the annotation workspace.

Exists because one number the pipeline reports is actively misleading on its own:
``validator_vs_human_kappa`` is 0.000, and the frozen report prints it without comment.
That zero is a DEGENERATE value. Cohen's kappa needs variance on both sides, and the claim
validator predicted ``supported`` for every one of the 11197 claims, so the statistic is
undefined in substance and collapses to 0 arithmetically. Reading it as "agreement no
better than chance" inverts the finding: raw agreement is 0.79, and what the validator
actually fails at is detection -- it flags none of the claims a human calls unsupported.

Writes ``agreement/`` into the workspace and re-stamps the workspace checksum manifest,
because this script is the last writer there.

    python scripts/analyse_annotation_agreement.py --write
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import pandas as pd

WS = Path("evaluation/annotation_workspace")
ORACLE_LABELS = Path("evaluation/outputs/exp-e748800507ef/normalized/relevance_labels.csv")

#: Seed for the dev/holdout split. Fixed and committed so the split cannot be re-drawn
#: until it flatters a change: a split chosen after seeing the result is not a split.
SPLIT_SEED = 20260731

#: Share of SCENARIOS held out. The unit is the scenario, not the (scenario, job) pair --
#: pairs inside one scenario share a candidate profile and a declared reference, so
#: splitting pairs would put near-duplicates on both sides and the holdout would measure
#: nothing.
HOLDOUT_FRACTION = 0.3

BOOTSTRAP_ITERATIONS = 2000


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def claim_analysis() -> dict[str, Any]:
    frame = pd.read_csv(WS / "labels" / "claim_raters.csv")
    frame = frame[frame["gold"].notna()]
    gold = frame["gold"].astype(int)
    validator = frame["validator"].astype(int)

    distinct = sorted(validator.unique().tolist())
    constant = len(distinct) == 1
    supported_human = int((gold == 1).sum())
    unsupported_human = int((gold == 0).sum())

    # confusion matrix, validator (rows) x human (cols)
    matrix = {
        f"validator_{v}": {f"human_{h}": int(((validator == v) & (gold == h)).sum())
                           for h in (0, 1)}
        for v in (0, 1)
    }
    flagged_unsupported = int(((validator == 0) & (gold == 0)).sum())
    detection_rate = (flagged_unsupported / unsupported_human) if unsupported_human else None

    by_type = (frame.assign(gold=gold)
               .groupby("claim_type")["gold"]
               .agg(n="count", supported_rate="mean"))
    by_type["unsupported"] = ((1 - by_type["supported_rate"]) * by_type["n"]).round().astype(int)

    return {
        "n_items": int(len(frame)),
        "raw_agreement_raters": float((frame["rater_1"] == frame["rater_2"]).mean()),
        "validator_column_distinct_values": distinct,
        "validator_is_constant": bool(constant),
        "kappa_degenerate": bool(constant),
        "kappa_degenerate_reason": (
            "The validator predicted the same class (supported) for all "
            f"{len(frame)} claims, so its predictions carry zero variance. Cohen's kappa "
            "is undefined for a constant predictor and the implementation returns 0.000. "
            "This is NOT chance-level agreement: raw agreement is "
            f"{float((gold == validator).mean()):.4f}. The substantive failure is "
            "detection, reported as unsupported_detection_rate below."
        ),
        "raw_validator_vs_human_agreement": float((gold == validator).mean()),
        "human_supported": supported_human,
        "human_unsupported": unsupported_human,
        "human_unsupported_rate": float(unsupported_human / len(frame)),
        "confusion_matrix_validator_rows_human_cols": matrix,
        "validator_flagged_unsupported": int((validator == 0).sum()),
        "unsupported_detection_rate": detection_rate,
        "unsupported_detection_note": (
            "Share of the claims a human adjudicated as unsupported that the validator "
            "also marked unsupported. It is 0.0 because the validator never marks "
            "anything unsupported."
        ),
        "by_claim_type": {
            str(k): {"n": int(v["n"]), "supported_rate": float(v["supported_rate"]),
                     "unsupported": int(v["unsupported"])}
            for k, v in by_type.iterrows()
        },
    }


def relevance_analysis() -> dict[str, Any]:
    frame = pd.read_csv(WS / "labels" / "relevance_raters.csv")
    frame = frame[frame["gold"].notna()]
    gold = frame["gold"].astype(int)
    oracle = frame["oracle_grade"].astype(int)
    delta = gold - oracle

    matrix = {f"oracle_{o}": {f"human_{h}": int(((oracle == o) & (gold == h)).sum())
                              for h in range(4)}
              for o in range(4)}
    return {
        "n_items": int(len(frame)),
        "raw_agreement_raters": float((frame["rater_1"] == frame["rater_2"]).mean()),
        "exact_grade_agreement_oracle_vs_human": float((gold == oracle).mean()),
        "mean_absolute_grade_difference": float(delta.abs().mean()),
        "human_stricter_rows": int((delta < 0).sum()),
        "human_looser_rows": int((delta > 0).sum()),
        "identical_rows": int((delta == 0).sum()),
        "delta_distribution": {str(k): int(v)
                               for k, v in delta.value_counts().sort_index().items()},
        "grade_distribution_human": {str(k): int(v)
                                     for k, v in gold.value_counts().sort_index().items()},
        "grade_distribution_oracle": {str(k): int(v)
                                      for k, v in oracle.value_counts().sort_index().items()},
        "confusion_matrix_oracle_rows_human_cols": matrix,
        "direction_note": (
            "The automatic oracle grades more generously than the adjudicated humans: it "
            f"is looser on {int((delta < 0).sum())} of {len(frame)} judged pairs and "
            f"stricter on only {int((delta > 0).sum())}. The dominant cause is the salary "
            "reading recorded in RUBRIC.md -- the humans require the posting's MINIMUM to "
            "clear a stated floor, the oracle accepts an overlapping salary range."
        ),
    }


def _weighted_kappa(a: list[int], b: list[int]) -> float | None:
    try:
        from sklearn.metrics import cohen_kappa_score
    except ImportError:  # pragma: no cover - sklearn is a declared dependency
        return None
    if len(set(a)) < 2 and len(set(b)) < 2:
        return None
    return float(cohen_kappa_score(a, b, weights="quadratic"))


def _bootstrap_kappa_ci(pairs: list[tuple[int, int]]) -> dict[str, float | None]:
    """Percentile CI for the weighted kappa, resampling pairs.

    A point estimate on a few hundred judged pairs is not a quality gate: the audit plan
    proposed pre-registering kappa >= 0.80, and on this sample size 0.78 and 0.82 are not
    distinguishable. Reporting the interval is what makes the number usable.
    """
    if not pairs:
        return {"low": None, "high": None}
    rng = random.Random(SPLIT_SEED)
    draws: list[float] = []
    size = len(pairs)
    for _ in range(BOOTSTRAP_ITERATIONS):
        sample = [pairs[rng.randrange(size)] for _ in range(size)]
        value = _weighted_kappa([p[0] for p in sample], [p[1] for p in sample])
        if value is not None:
            draws.append(value)
    if not draws:
        return {"low": None, "high": None}
    draws.sort()
    return {"low": round(draws[int(0.025 * len(draws))], 4),
            "high": round(draws[min(len(draws) - 1, int(0.975 * len(draws)))], 4)}


def oracle_calibration() -> dict[str, Any]:
    """Frozen scenario-level split, and oracle-vs-human agreement before/after per side.

    Read this for what it is. The guaranteed-minimum salary fix was made after inspecting
    the disagreement across ALL judged pairs, so the "holdout" below was already seen and
    is NOT a pre-registered held-out estimate -- no retrospective split can manufacture
    one. Two things make it worth computing anyway.

    First, as a stability check: the fix has zero free parameters -- it replaces one
    comparison rule with another stated in the rubric, with nothing fitted to the data --
    so if dev and holdout move together, the improvement is a property of the rule rather
    than of a handful of pairs. Divergence between the two sides would be the warning sign.

    Second, and more usefully, it FREEZES the split for future oracle changes, which can
    then be evaluated on a side they have not seen.
    """
    from jobrec.catalog import load_catalog
    from jobrec.config import load_config
    from jobrec_eval.oracle_reference import load_or_build_canonical_references
    from jobrec_eval.relevance import grade_catalog

    raters = pd.read_csv(WS / "labels" / "relevance_raters.csv")
    raters = raters[raters["gold"].notna()]
    human = {(r.scenario_id, r.job_id): int(r.gold) for r in raters.itertuples()}
    #: The oracle grade recorded when the workspace was built, i.e. BEFORE the fix.
    before = {(r.scenario_id, r.job_id): int(r.oracle_grade) for r in raters.itertuples()}

    config = load_config("configs/experiment_full.yaml", base_dir="configs")
    catalog = load_catalog("data/processed/jobs.jsonl")
    canonical = load_or_build_canonical_references(
        "evaluation/data/scenarios.jsonl", "data/processed/jobs.jsonl", config)
    graded = grade_catalog(catalog, canonical.references, config)
    after = {(r.scenario_id, r.job_id): int(r.relevance_grade) for r in graded.itertuples()}

    scenarios = sorted({key[0] for key in human})
    rng = random.Random(SPLIT_SEED)
    shuffled = list(scenarios)
    rng.shuffle(shuffled)
    cut = round(len(shuffled) * HOLDOUT_FRACTION)
    holdout = sorted(shuffled[:cut])
    dev = sorted(shuffled[cut:])

    def side(scenario_ids: list[str]) -> dict[str, Any]:
        keys = [k for k in sorted(human) if k[0] in set(scenario_ids) and k in after]
        h = [human[k] for k in keys]
        pre = [before[k] for k in keys]
        post = [after[k] for k in keys]
        return {
            "scenarios": len(scenario_ids),
            "judged_pairs": len(keys),
            "kappa_before": _weighted_kappa(pre, h),
            "kappa_after": _weighted_kappa(post, h),
            "kappa_after_ci95": _bootstrap_kappa_ci(list(zip(post, h, strict=True))),
            "exact_agreement_before": round(
                sum(1 for a, b in zip(pre, h, strict=True) if a == b) / len(keys), 4)
            if keys else None,
            "exact_agreement_after": round(
                sum(1 for a, b in zip(post, h, strict=True) if a == b) / len(keys), 4)
            if keys else None,
            "oracle_more_generous_after": sum(
                1 for a, b in zip(post, h, strict=True) if b < a),
            "oracle_stricter_after": sum(
                1 for a, b in zip(post, h, strict=True) if b > a),
            "grades_changed_by_the_fix": sum(1 for k in keys if before[k] != after[k]),
        }

    return {
        "split": {
            "seed": SPLIT_SEED,
            "unit": "scenario",
            "holdout_fraction": HOLDOUT_FRACTION,
            "dev_scenarios": dev,
            "holdout_scenarios": holdout,
        },
        "dev": side(dev),
        "holdout": side(holdout),
        "all": side(scenarios),
        "interpretation": (
            "NOT a pre-registered held-out estimate: the guaranteed-minimum salary fix was "
            "made after inspecting the disagreement over all judged pairs, so both sides "
            "had been seen. Read the dev/holdout comparison as a stability check -- the fix "
            "has no fitted parameters, so agreement between the two sides indicates the "
            "improvement is a property of the rule rather than of particular pairs. The "
            "split is frozen here so that the NEXT oracle change can be evaluated on a side "
            "it has not seen."
        ),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--write", action="store_true")
    args = parser.parse_args()

    result = {
        "relevance": relevance_analysis(),
        "claims": claim_analysis(),
        "oracle_calibration": oracle_calibration(),
        "forbidden_readings": [
            "Do NOT report validator_vs_human_kappa = 0.000 as chance-level agreement. "
            "It is degenerate: the validator is a constant all-supported predictor.",
            "Do NOT quote the human ranking metrics as an unbiased effect estimate. The "
            "human label universe is the 368 returned pairs, while the oracle grades the "
            "whole catalogue, so NDCG's ideal DCG is computed over a different pool. It "
            "is an agreement diagnostic.",
        ],
    }

    print("=== relevance ===")
    r = result["relevance"]
    for key in ("n_items", "raw_agreement_raters", "exact_grade_agreement_oracle_vs_human",
                "mean_absolute_grade_difference", "human_stricter_rows",
                "human_looser_rows", "identical_rows"):
        print(f"  {key:<38} {r[key]}")
    print("\n=== claims ===")
    c = result["claims"]
    for key in ("n_items", "validator_column_distinct_values", "validator_is_constant",
                "raw_validator_vs_human_agreement", "human_unsupported",
                "human_unsupported_rate", "validator_flagged_unsupported",
                "unsupported_detection_rate"):
        print(f"  {key:<38} {c[key]}")
    print("\n  by claim_type:")
    for name, stats in c["by_claim_type"].items():
        print(f"    {name:<22} n={stats['n']:<6} supported_rate={stats['supported_rate']:.4f} "
              f"unsupported={stats['unsupported']}")

    cal = result["oracle_calibration"]
    print("\n=== oracle calibration (frozen scenario-level split) ===")
    print(f"  seed {cal['split']['seed']}, holdout fraction "
          f"{cal['split']['holdout_fraction']}")
    for name in ("dev", "holdout", "all"):
        s = cal[name]
        ci = s["kappa_after_ci95"]
        print(f"  {name:<8} scenarios={s['scenarios']:<3} pairs={s['judged_pairs']:<4} "
              f"kappa {s['kappa_before']:.4f} -> {s['kappa_after']:.4f} "
              f"[{ci['low']}, {ci['high']}]  changed={s['grades_changed_by_the_fix']}")

    if not args.write:
        print("\npass --write to persist")
        return 0

    out = WS / "agreement"
    out.mkdir(parents=True, exist_ok=True)
    (out / "agreement.json").write_text(json.dumps(result, indent=2, sort_keys=True),
                                        encoding="utf-8", newline="\n")

    # Re-stamp the workspace manifest: this script is its last writer.
    manifest_path = WS / "checksums.json"
    files = sorted(p for p in WS.rglob("*") if p.is_file() and p != manifest_path)
    manifest_path.write_text(json.dumps({
        "algorithm": "sha256",
        "file_count": len(files),
        "files": {p.relative_to(WS).as_posix(): sha256(p) for p in files},
    }, indent=2, sort_keys=True), encoding="utf-8", newline="\n")
    print(f"\nwrote {out / 'agreement.json'} and re-stamped checksums "
          f"({len(files)} files recorded)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
