"""Matplotlib plots for the evaluation report (headless / Agg backend)."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_VARIANT_ORDER = ["full", "profile_only", "one_shot", "no_memory", "no_context"]


def _order(variants):
    return [v for v in _VARIANT_ORDER if v in set(variants)]


def _box_by_variant(sv: pd.DataFrame, metric: str, title: str, out: Path, ylim=(0, 1)):
    variants = _order(sv["variant"].unique())
    data = [sv[sv["variant"] == v][metric].dropna().to_numpy() for v in variants]
    if not any(len(d) for d in data):
        return None
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, tick_labels=variants, showmeans=True)
    for i, d in enumerate(data, 1):
        if len(d):
            ax.scatter(np.full(len(d), i) + np.random.default_rng(1).normal(0, 0.04, len(d)),
                       d, alpha=0.4, s=12, color="tab:blue")
    ax.set_title(f"{title} (n scenarios per box shown; boxes=median/IQR, ▲=mean)")
    ax.set_ylabel(metric)
    if ylim:
        ax.set_ylim(*ylim)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_all(sv: pd.DataFrame, latency_pct: pd.DataFrame, out_dir: str | Path) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    made = []
    for metric, title, ylim, fname in [
        ("ndcg_at_5", "NDCG@5 by variant (automatic relevance oracle)", (0, 1), "ndcg_by_variant.png"),
        ("hcsr", "Hard-Constraint Satisfaction (HCSR) by variant", (0, 1.05), "hcsr_by_variant.png"),
        ("task_success", "Task Success by variant", (0, 1.05), "task_success_by_variant.png"),
        ("grounding", "Explanation grounding by variant", (0, 1.05), "grounding_by_variant.png"),
    ]:
        p = _box_by_variant(sv, metric, title, out_dir / fname, ylim)
        if p:
            made.append(p)

    made += _delta_plot(sv, "no_memory", out_dir / "memory_delta.png", "Full - No-Memory (per scenario)")
    made += _delta_plot(sv, "no_context", out_dir / "context_delta.png", "Full - No-Context (per scenario)")
    made += _turns_vs_success(sv, out_dir / "turns_vs_success.png")
    made += _latency_breakdown(latency_pct, out_dir / "latency_breakdown.png")
    return made


def _delta_plot(sv: pd.DataFrame, other: str, out: Path, title: str) -> list[Path]:
    made = []
    for metric in ["ndcg_at_5", "hcsr", "task_success"]:
        piv = sv.pivot_table(index="scenario_id", columns="variant", values=metric)
        if "full" not in piv.columns or other not in piv.columns:
            continue
        both = piv[["full", other]].dropna()
        if both.empty:
            continue
        diffs = (both["full"] - both[other]).sort_values()
        fig, ax = plt.subplots(figsize=(7, 4))
        colors = ["tab:green" if d >= 0 else "tab:red" for d in diffs]
        ax.barh(range(len(diffs)), diffs.to_numpy(), color=colors)
        ax.axvline(0, color="black", lw=0.8)
        ax.set_yticks(range(len(diffs)))
        ax.set_yticklabels(diffs.index, fontsize=6)
        ax.set_title(f"{title}: Δ{metric}")
        ax.set_xlabel(f"Δ{metric} (full - {other})")
        fig.tight_layout()
        p = out.with_name(out.stem + f"_{metric}.png")
        fig.savefig(p, dpi=120)
        plt.close(fig)
        made.append(p)
    return made


def _turns_vs_success(sv: pd.DataFrame, out: Path) -> list[Path]:
    variants = _order(sv["variant"].unique())
    fig, ax = plt.subplots(figsize=(6, 5))
    for v in variants:
        sub = sv[sv["variant"] == v]
        ax.scatter(sub["turn_count"].mean(), sub["task_success"].mean(), s=80, label=v)
        ax.annotate(v, (sub["turn_count"].mean(), sub["task_success"].mean()), fontsize=8,
                    xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean response turns")
    ax.set_ylabel("Mean task success")
    ax.set_title("Task success vs response turns (variant means)")
    ax.set_ylim(0, 1.05)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return [out]


def _latency_breakdown(latency_pct: pd.DataFrame, out: Path) -> list[Path]:
    if latency_pct.empty:
        return []
    comps = [c for c in latency_pct["component"].unique() if c != "total"]
    variants = _order(latency_pct["variant"].unique())
    fig, ax = plt.subplots(figsize=(8, 4.5))
    bottom = np.zeros(len(variants))
    for comp in comps:
        vals = [float(latency_pct[(latency_pct["variant"] == v) & (latency_pct["component"] == comp)]["median_ms"].sum())
                for v in variants]
        ax.bar(variants, vals, bottom=bottom, label=comp)
        bottom += np.array(vals)
    ax.set_ylabel("Median component latency (ms)")
    ax.set_title("Median latency breakdown by component")
    ax.legend(fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    return [out]
