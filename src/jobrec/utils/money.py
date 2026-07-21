"""Salary normalisation helpers.

Salaries are always stored with their original currency and period plus a
normalised monthly value in the base currency, so no information is lost.
"""

from __future__ import annotations

# Monthly-equivalent multipliers for supported periods.
_PERIOD_TO_MONTH = {
    "hour": 160.0,  # ~ full-time hours per month
    "month": 1.0,
    "year": 1.0 / 12.0,
    "unknown": None,
}

# Simple fixed FX table (relative to MYR) for the curated research catalog.
# Values are illustrative and fixed for reproducibility, not live rates.
_FX_TO_MYR = {
    "MYR": 1.0,
    "SGD": 3.5,
    "USD": 4.7,
    "EUR": 5.1,
}


def can_normalize(currency: str | None, period: str | None) -> bool:
    """Return True when a salary value can be normalised to monthly MYR."""
    if currency is None or period is None:
        return False
    return currency.upper() in _FX_TO_MYR and _PERIOD_TO_MONTH.get(period) is not None


def to_monthly_myr(amount: float, currency: str, period: str) -> float | None:
    """Convert an amount to a monthly value in MYR, or None if not possible."""
    factor = _PERIOD_TO_MONTH.get(period)
    fx = _FX_TO_MYR.get(currency.upper())
    if factor is None or fx is None:
        return None
    return round(amount * factor * fx, 2)
