"""Statistical evaluation for Cortex experiments.

Real stats over accumulated TRIBE v2 activation data points: two-sample t-tests,
Cohen's d effect size, and convergence criteria for the iterative research loop.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import stats as scipy_stats


def two_sample_test(values_a: list[float], values_b: list[float]) -> dict[str, Any]:
    """Welch's two-sample t-test + Cohen's d between two conditions."""
    a = np.asarray(values_a, dtype=float)
    b = np.asarray(values_b, dtype=float)

    if a.size < 2 or b.size < 2:
        return {
            "n_a": int(a.size), "n_b": int(b.size),
            "mean_a": float(a.mean()) if a.size else None,
            "mean_b": float(b.mean()) if b.size else None,
            "t": None, "p_value": None, "cohens_d": None,
            "significant": False,
            "note": "Insufficient data points (need >=2 per condition).",
        }

    t, p = scipy_stats.ttest_ind(a, b, equal_var=False)

    # Cohen's d with pooled SD.
    pooled_sd = np.sqrt(((a.size - 1) * a.var(ddof=1) + (b.size - 1) * b.var(ddof=1)) / (a.size + b.size - 2))
    d = (a.mean() - b.mean()) / pooled_sd if pooled_sd > 0 else 0.0

    return {
        "n_a": int(a.size), "n_b": int(b.size),
        "mean_a": float(a.mean()), "mean_b": float(b.mean()),
        "std_a": float(a.std(ddof=1)), "std_b": float(b.std(ddof=1)),
        "diff": float(a.mean() - b.mean()),
        "t": float(t), "p_value": float(p),
        "cohens_d": float(d),
        "effect_size_label": _effect_label(abs(d)),
        "significant": bool(p < 0.05),
        "direction": "A > B" if a.mean() > b.mean() else "B > A",
    }


def _effect_label(d: float) -> str:
    if d < 0.2:
        return "negligible"
    if d < 0.5:
        return "small"
    if d < 0.8:
        return "medium"
    return "large"


def check_convergence(
    test_result: dict[str, Any],
    iteration: int,
    max_iterations: int = 4,
    min_n_per_condition: int = 4,
) -> dict[str, Any]:
    """Decide whether the experiment has gathered enough evidence to stop.

    Converged when: enough data points AND (clear significant effect OR clear null),
    or when max iterations reached.
    """
    n_ok = test_result.get("n_a", 0) >= min_n_per_condition and test_result.get("n_b", 0) >= min_n_per_condition
    p = test_result.get("p_value")
    d = test_result.get("cohens_d")

    if iteration >= max_iterations:
        return {"converged": True, "reason": "max_iterations_reached", "decision": "stop"}

    if not n_ok:
        return {"converged": False, "reason": "need_more_data", "decision": "collect_more"}

    if p is not None and d is not None:
        if p < 0.01 and abs(d) >= 0.8:
            return {"converged": True, "reason": "strong_significant_effect", "decision": "stop"}
        if p > 0.4 and abs(d) < 0.2:
            return {"converged": True, "reason": "clear_null_result", "decision": "stop"}

    return {"converged": False, "reason": "ambiguous_collect_more", "decision": "collect_more"}
