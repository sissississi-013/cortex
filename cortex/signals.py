"""Raindrop custom signals for monitoring Cortex research quality.

Defines signal checkers that detect common failure modes in autonomous
neuroscience research:
- Unfalsifiable hypotheses
- Tool selection mismatches
- Result overinterpretation
- Hypothesis loop stalls
"""

from __future__ import annotations

import os
from typing import Any

UNFALSIFIABLE_MARKERS = [
    "brain activity will be observed",
    "some regions will activate",
    "there will be neural correlates",
    "the brain processes",
    "neural activity occurs",
    "activation will be present",
]

TOOL_MODALITY_MAP = {
    "predict_fmri": {"modality": "fMRI", "spatial_resolution": "high", "temporal_resolution": "low"},
    "search_literature": {"modality": "meta-analysis", "spatial_resolution": "moderate", "temporal_resolution": "none"},
    "meta_analyze": {"modality": "meta-analysis", "spatial_resolution": "moderate", "temporal_resolution": "none"},
    "decode_brain_region": {"modality": "meta-analysis", "spatial_resolution": "moderate", "temporal_resolution": "none"},
    "analyze_eeg": {"modality": "EEG", "spatial_resolution": "low", "temporal_resolution": "high"},
}

SPATIAL_KEYWORDS = [
    "region", "cortex", "gyrus", "sulcus", "area", "nucleus",
    "voxel", "activation map", "spatial", "localize", "lateralize",
    "fmri", "bold", "mri",
]

TEMPORAL_KEYWORDS = [
    "latency", "millisecond", "erp", "component", "p300", "n100",
    "n170", "p200", "temporal", "timing", "oscillation", "frequency",
    "alpha", "beta", "gamma", "theta", "delta", "eeg",
]


def check_unfalsifiable(hypothesis: str) -> dict[str, Any]:
    """Check if a hypothesis is unfalsifiable (trivially true or untestable)."""
    hypothesis_lower = hypothesis.lower()

    for marker in UNFALSIFIABLE_MARKERS:
        if marker in hypothesis_lower:
            return {
                "signal": "hypothesis_unfalsifiable",
                "fired": True,
                "severity": "warning",
                "reason": f"Hypothesis contains vague/unfalsifiable language: '{marker}'",
                "recommendation": "Reformulate with specific, quantitative predictions "
                                 "(e.g., 'V1 activation will be >0.8' or "
                                 "'left STG activation will exceed right STG by >20%')",
            }

    has_specific_region = any(
        word in hypothesis_lower
        for word in ["v1", "v2", "stg", "ifg", "amygdala", "hippocampus",
                     "broca", "wernicke", "fusiform", "insula", "pfc",
                     "cerebellum", "thalamus", "cingulate", "precuneus"]
    )
    has_specific_prediction = any(
        word in hypothesis_lower
        for word in ["more than", "less than", "greater", "higher", "lower",
                     "increase", "decrease", "correlate", "differ", "lateralize"]
    )

    if not has_specific_region and not has_specific_prediction:
        return {
            "signal": "hypothesis_unfalsifiable",
            "fired": True,
            "severity": "info",
            "reason": "Hypothesis lacks specific brain regions or directional predictions",
            "recommendation": "Add specific region names and directional predictions",
        }

    return {"signal": "hypothesis_unfalsifiable", "fired": False}


def check_tool_mismatch(hypothesis: str, selected_tool: str) -> dict[str, Any]:
    """Check if the selected tool is appropriate for the hypothesis."""
    hypothesis_lower = hypothesis.lower()

    spatial_score = sum(1 for kw in SPATIAL_KEYWORDS if kw in hypothesis_lower)
    temporal_score = sum(1 for kw in TEMPORAL_KEYWORDS if kw in hypothesis_lower)

    tool_info = TOOL_MODALITY_MAP.get(selected_tool, {})

    if temporal_score > spatial_score and tool_info.get("temporal_resolution") == "low":
        return {
            "signal": "tool_selection_mismatch",
            "fired": True,
            "severity": "warning",
            "selected_tool": selected_tool,
            "reason": f"Hypothesis focuses on temporal dynamics but {selected_tool} has low temporal resolution",
            "recommendation": "Use analyze_eeg for temporal/ERP questions",
        }

    if spatial_score > temporal_score and tool_info.get("spatial_resolution") == "low":
        return {
            "signal": "tool_selection_mismatch",
            "fired": True,
            "severity": "warning",
            "selected_tool": selected_tool,
            "reason": f"Hypothesis focuses on spatial localization but {selected_tool} has low spatial resolution",
            "recommendation": "Use predict_fmri or meta_analyze for spatial localization questions",
        }

    return {"signal": "tool_selection_mismatch", "fired": False}


def check_overinterpretation(
    claimed_significant: bool,
    p_value: float | None = None,
    z_score: float | None = None,
    effect_size: float | None = None,
) -> dict[str, Any]:
    """Check if results are being overinterpreted."""
    issues: list[str] = []

    if claimed_significant and p_value is not None and p_value > 0.05:
        issues.append(f"Claimed significance but p = {p_value:.4f} > 0.05")

    if claimed_significant and z_score is not None and abs(z_score) < 3.1:
        issues.append(f"Claimed significance but |z| = {abs(z_score):.2f} < 3.1")

    if effect_size is not None and effect_size < 0.2 and claimed_significant:
        issues.append(f"Small effect size ({effect_size:.2f}) treated as definitive")

    if issues:
        return {
            "signal": "result_overinterpretation",
            "fired": True,
            "severity": "warning",
            "issues": issues,
            "recommendation": "Re-interpret with appropriate caveats. "
                             "Report effect sizes alongside p-values. "
                             "Qualify weak results as 'suggestive' not 'conclusive'.",
        }

    return {"signal": "result_overinterpretation", "fired": False}


OVERCLAIM_TERMS = [
    "significant", "significantly", "proves", "proven", "clearly shows", "demonstrates",
    "confirms", "strong evidence", "definitively", "established", "conclusive", "robustly",
]


def check_overinterpretation_prose(interpretation: str, stats: dict[str, Any]) -> dict[str, Any]:
    """Catch the agent over-claiming: strong language while the stats are weak/non-significant."""
    text = (interpretation or "").lower()
    p = stats.get("p_value")
    overclaims = [t for t in OVERCLAIM_TERMS if t in text]
    weak = (p is None) or (p >= 0.05)
    if overclaims and weak:
        return {
            "fired": True, "name": "result_overinterpretation", "sentiment": "NEGATIVE",
            "reason": f"Interpretation uses strong language {overclaims} but p={p} (not significant)",
            "recommendation": "Qualify as suggestive; report effect size alongside p-value.",
        }
    return {"fired": False, "name": "result_overinterpretation"}


def check_low_power_stats(stats: dict[str, Any], min_n: int = 4) -> dict[str, Any]:
    """Flag a decision being drawn on too few data points."""
    na, nb = stats.get("n_a", 0), stats.get("n_b", 0)
    if na < min_n or nb < min_n:
        return {
            "fired": True, "name": "low_statistical_power", "sentiment": "NEGATIVE",
            "reason": f"n={na}/{nb} below {min_n} per condition — underpowered.",
            "recommendation": "Collect more clips before drawing conclusions.",
        }
    return {"fired": False, "name": "low_statistical_power"}


def check_loop_stall(hypothesis_id: str, iteration: int, max_iterations: int = 3) -> dict[str, Any]:
    """Check if the hypothesis loop is stalled."""
    if iteration >= max_iterations:
        return {
            "signal": "hypothesis_loop_stall",
            "fired": True,
            "severity": "warning",
            "hypothesis_id": hypothesis_id,
            "iteration": iteration,
            "reason": f"Hypothesis refined {iteration} times without convergence",
            "recommendation": "Discard this hypothesis and pivot to an alternative approach",
        }

    return {"signal": "hypothesis_loop_stall", "fired": False}


def init_raindrop():
    """Initialize Raindrop tracing for the Cortex agent."""
    write_key = os.getenv("RAINDROP_WRITE_KEY")
    if not write_key:
        print("[Cortex] RAINDROP_WRITE_KEY not set — tracing disabled")
        return None

    try:
        import raindrop.analytics as raindrop
        raindrop.init(write_key, tracing_enabled=True)
        print("[Cortex] Raindrop tracing initialized")

        try:
            from raindrop_openai_agents import create_raindrop_openai_agents
            rd = create_raindrop_openai_agents(
                api_key=write_key,
                user_id="cortex-agent",
            )
            print("[Cortex] Raindrop OpenAI Agents integration active")
            return rd
        except ImportError:
            print("[Cortex] raindrop-openai-agents not installed — using core SDK only")
            return raindrop

    except ImportError:
        print("[Cortex] raindrop-ai not installed — tracing disabled")
        return None


def trace_hypothesis_cycle(
    hypothesis: str,
    tool: str,
    result: dict[str, Any],
    iteration: int,
) -> list[dict[str, Any]]:
    """Run all signal checks for a hypothesis cycle and return fired signals."""
    fired: list[dict[str, Any]] = []

    sig = check_unfalsifiable(hypothesis)
    if sig["fired"]:
        fired.append(sig)

    sig = check_tool_mismatch(hypothesis, tool)
    if sig["fired"]:
        fired.append(sig)

    p_val = None
    z_val = None
    if isinstance(result, dict):
        clusters = result.get("significant_clusters") or result.get("all_clusters") or []
        if clusters and isinstance(clusters[0], dict):
            p_val = clusters[0].get("p_value")
            z_val = clusters[0].get("peak_z_score")

    sig = check_overinterpretation(
        claimed_significant=True,
        p_value=p_val,
        z_score=z_val,
    )
    if sig["fired"]:
        fired.append(sig)

    sig = check_loop_stall("current", iteration)
    if sig["fired"]:
        fired.append(sig)

    return fired
