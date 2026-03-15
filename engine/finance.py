from __future__ import annotations

import math
from typing import Iterable, List, Optional


def pmt(rate: float, nper: int, pv: float) -> float:
    """
    Standard annual PMT.
    Returns positive annual debt service.
    """
    if nper <= 0:
        raise ValueError("nper must be > 0")
    if pv < 0:
        pv = abs(pv)

    if abs(rate) < 1e-12:
        return pv / nper

    return pv * (rate / (1 - (1 + rate) ** (-nper)))


def npv(rate: float, cashflows: Iterable[float]) -> float:
    total = 0.0
    for t, cf in enumerate(cashflows):
        total += cf / ((1 + rate) ** t)
    return total


def irr_bisection(
    cashflows: List[float],
    low: float = -0.95,
    high: float = 1.50,
    tol: float = 1e-7,
    max_iter: int = 300,
) -> Optional[float]:
    """
    Robust IRR via bisection.
    Returns None if no sign change exists in interval.
    """
    f_low = npv(low, cashflows)
    f_high = npv(high, cashflows)

    if math.isnan(f_low) or math.isnan(f_high):
        return None
    if f_low == 0:
        return low
    if f_high == 0:
        return high
    if f_low * f_high > 0:
        return None

    a, b = low, high
    fa, fb = f_low, f_high

    for _ in range(max_iter):
        mid = 0.5 * (a + b)
        fm = npv(mid, cashflows)

        if abs(fm) < tol:
            return mid

        if fa * fm < 0:
            b, fb = mid, fm
        else:
            a, fa = mid, fm

    return 0.5 * (a + b)


def triangular_sample(low: float, mode: float, high: float, rng) -> float:
    return rng.triangular(low, high, mode)
