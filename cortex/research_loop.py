"""The Cortex autonomous research loop.

This is the real auto-research engine. Given a research question, it:
  1. Designs falsifiable experiments (LLM) — hypothesis, conditions A/B, focal ROI, stimulus prompts
  2. Generates REAL video stimuli (Sora / image->video)
  3. Runs each stimulus through REAL TRIBE v2 on Modal GPU
  4. Extracts the focal ROI activation from the real cortical map
  5. Accumulates data points and runs real stats (t-test, Cohen's d)
  6. Decides keep/refine/discard and iterates with more stimuli until convergence
  7. Synthesizes a report grounded in the real measured data

Everything is streamed as structured events for the live frontend.

If Modal/TRIBE is not connected, the loop reports that honestly and stops —
it never fabricates activation numbers.
"""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any, AsyncGenerator

from dotenv import load_dotenv
from openai import OpenAI

import uuid
from contextlib import contextmanager
from pathlib import Path

from cortex.stats import two_sample_test
from cortex.signals import check_overinterpretation_prose, check_low_power_stats

load_dotenv()

STIMULI_DIR = Path(__file__).parent.parent / "stimuli"
STIMULI_DIR.mkdir(exist_ok=True)

MODEL = os.getenv("CORTEX_MODEL", "gpt-4.1")
MAX_ITERATIONS = int(os.getenv("CORTEX_MAX_ITERATIONS", "2"))
N_PER_CONDITION = int(os.getenv("CORTEX_N_PER_CONDITION", "5"))


class TribeUnavailable(Exception):
    """Raised when the real TRIBE v2 Modal backend cannot be reached."""


def _client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Workshop tracing — full internal trajectory via direct OTLP (no API key)
# ---------------------------------------------------------------------------

from cortex.workshop_trace import WorkshopTrace  # noqa: E402


# ---------------------------------------------------------------------------
# Real TRIBE v2 invocation (Modal)
# ---------------------------------------------------------------------------

def _get_batch_fn():
    """Look up the deployed Modal run_experiment_batch function. Raises TribeUnavailable if not reachable."""
    try:
        import modal
    except ImportError as e:
        raise TribeUnavailable("modal package not installed") from e
    try:
        return modal.Function.from_name("cortex-tribe", "run_experiment_batch")
    except Exception as e:
        raise TribeUnavailable(
            f"Could not reach deployed Modal app 'cortex-tribe'. Ensure you ran "
            f"`modal token new` and `modal deploy cortex/modal_app.py`. ({type(e).__name__}: {e})"
        ) from e


def _save_b64(b64: str | None, name: str) -> str | None:
    if not b64:
        return None
    import base64
    p = STIMULI_DIR / name
    p.write_bytes(base64.b64decode(b64))
    return f"/stimuli/{p.name}"


async def _run_batch(prompts_a, prompts_b, name_a, name_b, focal_roi, n, exclude, tag) -> dict[str, Any]:
    """Run a parallel A-vs-B batch experiment on Modal and persist clips + surface maps."""
    fn = _get_batch_fn()
    res = await asyncio.to_thread(
        fn.remote, prompts_a=prompts_a, prompts_b=prompts_b, name_a=name_a, name_b=name_b,
        focal_roi=focal_roi, n_per_condition=n, exclude_ids=exclude,
    )
    # Persist per-clip videos.
    for det in res.get("clips_a", []) + res.get("clips_b", []):
        vb = det.pop("video_b64", None)
        if vb:
            import base64
            out = STIMULI_DIR / f"{det['clip_id']}.mp4"
            if not out.exists():
                out.write_bytes(base64.b64decode(vb))
            det["url"] = f"/stimuli/{out.name}"
    # Persist surface maps.
    res["group_map_a_url"] = _save_b64(res.pop("group_map_a_b64", None), f"{tag}_groupA.png")
    res["group_map_b_url"] = _save_b64(res.pop("group_map_b_b64", None), f"{tag}_groupB.png")
    res["contrast_map_url"] = _save_b64(res.pop("contrast_map_b64", None), f"{tag}_contrast.png")
    return res


# ---------------------------------------------------------------------------
# LLM reasoning steps
# ---------------------------------------------------------------------------

DESIGN_SYSTEM = """You are Cortex, an autonomous neuroscience researcher. Design rigorous, \
falsifiable experiments that can be tested by predicting fMRI brain activation (TRIBE v2) \
to video stimuli and measuring activation in a specific focal brain region (ROI).

Available focal ROIs (cortical, fsaverage5): FFA (fusiform face area), STG (superior temporal \
gyrus), STS, A1 (auditory), IFG (Broca/inferior frontal), AG (angular gyrus), SMG, dlPFC, ACC, \
PCC, M1 (motor), V1, V2, precuneus, mPFC, OFC, insula, MTG, PPA (place area).

For the research question, design 1-2 experiments. Each is a controlled A vs B contrast where \
you predict the focal ROI responds differently. Write 2-3 concrete stimulus DESCRIPTIONS per \
condition — these are used to retrieve matching real naturalistic video clips from the MSR-VTT \
dataset (10k captioned clips), then fed to TRIBE v2. Describe what is happening in the clip in \
natural language (as a video caption would). CRITICAL for clean contrasts: make the conditions \
differ ONLY in the variable of interest and avoid confounds — e.g. for a faces-vs-houses contrast, \
the house descriptions must explicitly have NO people/faces ("an empty house interior, no people").

Respond ONLY as JSON:
{
  "experiments": [
    {
      "hypothesis": "FFA shows higher activation to faces than to houses",
      "prediction": "Mean FFA activation for face clips > house clips (Cohen's d > 0.8)",
      "focal_roi": "FFA",
      "condition_a": {"name": "faces", "stimulus_prompts": ["a close up of a person's face talking to the camera", "a woman smiling and looking at the camera"]},
      "condition_b": {"name": "houses", "stimulus_prompts": ["an empty house with rooms and furniture", "a building exterior with no people"]}
    }
  ]
}"""

DECIDE_SYSTEM = """You are Cortex evaluating results from a real fMRI-prediction experiment. \
Given the hypothesis, the focal ROI, and the current statistics over real TRIBE v2 activation \
data points, decide what to do. Be rigorous: do not overinterpret weak/non-significant results.

Respond ONLY as JSON:
{
  "interpretation": "1-3 sentence honest read of the current evidence",
  "status": "supported | refined | discarded | inconclusive",
  "decision": "stop | collect_more",
  "additional_stimuli_a": ["optional extra prompt for condition A if collecting more"],
  "additional_stimuli_b": ["optional extra prompt for condition B if collecting more"]
}"""


def _design_experiments(question: str) -> list[dict]:
    r = _client().chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": DESIGN_SYSTEM},
            {"role": "user", "content": f"Research question: {question}"},
        ],
    )
    data = json.loads(r.choices[0].message.content or "{}")
    return data.get("experiments", [])


def _interpret_and_decide(hypothesis: str, focal_roi: str, test_result: dict, iteration: int, convergence: dict) -> dict:
    r = _client().chat.completions.create(
        model=MODEL,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": DECIDE_SYSTEM},
            {"role": "user", "content": json.dumps({
                "hypothesis": hypothesis,
                "focal_roi": focal_roi,
                "iteration": iteration,
                "statistics": test_result,
                "convergence_check": convergence,
            })},
        ],
    )
    return json.loads(r.choices[0].message.content or "{}")


REFINE_SYSTEM = """You are Cortex improving an underpowered/confounded experiment. The current \
A-vs-B contrast on the focal ROI is not yet significant. You are shown each clip's caption and its \
measured focal-ROI activation. Identify confounds (e.g. a "house" clip that actually contains a \
person inflates a face-region ROI) and propose CLEANER, more specific retrieval prompts for the \
next batch that better isolate each condition.

Respond ONLY as JSON:
{
  "diagnosis": "1-2 sentences on why the contrast is weak (confounds/variance)",
  "prompts_a": ["cleaner description 1", "cleaner description 2"],
  "prompts_b": ["cleaner description 1", "cleaner description 2"]
}"""


def _refine_prompts(hypothesis: str, focal_roi: str, name_a: str, name_b: str,
                    clips_a: list, clips_b: list, stats: dict) -> dict:
    payload = {
        "hypothesis": hypothesis, "focal_roi": focal_roi,
        "condition_a": name_a, "condition_b": name_b, "statistics": stats,
        "condition_a_clips": [{"caption": c.get("caption"), "focal_value": c.get("focal_value")} for c in clips_a],
        "condition_b_clips": [{"caption": c.get("caption"), "focal_value": c.get("focal_value")} for c in clips_b],
    }
    r = _client().chat.completions.create(
        model=MODEL, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": REFINE_SYSTEM},
                  {"role": "user", "content": json.dumps(payload)}],
    )
    return json.loads(r.choices[0].message.content or "{}")


def _interpret_result(hypothesis: str, focal_roi: str, name_a: str, name_b: str, stats: dict) -> str:
    r = _client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are a neuroscientist. In ONE natural sentence, "
             "interpret this single experiment's result based on the statistics provided."},
            {"role": "user", "content": json.dumps({"hypothesis": hypothesis, "focal_roi": focal_roi,
                "condition_a": name_a, "condition_b": name_b, "statistics": stats})},
        ],
    )
    return (r.choices[0].message.content or "").strip()


def _correct_interpretation(interpretation: str, stats: dict) -> str:
    r = _client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "Rewrite this interpretation in ONE sentence so it does NOT "
             "overstate the evidence. Given the p-value and sample size, qualify appropriately "
             "(e.g. 'suggestive, not significant') and report the effect size honestly."},
            {"role": "user", "content": json.dumps({"interpretation": interpretation, "statistics": stats})},
        ],
    )
    return (r.choices[0].message.content or "").strip()


def _synthesize_report(question: str, experiments: list[dict]) -> str:
    r = _client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are Cortex. Write a rigorous research report in "
             "markdown grounded ONLY in the real measured data provided. Include: Research Question, "
             "Experiments Conducted (with real n, means, p-values, Cohen's d per experiment), "
             "Hypotheses (SUPPORTED/REFINED/DISCARDED), Key Findings, Limitations, Conclusions. "
             "Do not invent numbers beyond what is given."},
            {"role": "user", "content": json.dumps({"question": question, "experiments": experiments})},
        ],
    )
    return r.choices[0].message.content or ""


# ---------------------------------------------------------------------------
# The loop
# ---------------------------------------------------------------------------

def _get_infer_bytes_fn():
    """Modal run_tribe_inference (takes raw video bytes). Raises TribeUnavailable if unreachable."""
    try:
        import modal
    except ImportError as e:
        raise TribeUnavailable("modal package not installed") from e
    try:
        return modal.Function.from_name("cortex-tribe", "run_tribe_inference")
    except Exception as e:
        raise TribeUnavailable(f"Modal app 'cortex-tribe' not reachable ({type(e).__name__}: {e})") from e


RANKING_DESIGN_SYSTEM = """You are Cortex, designing a CONTROLLED visual-stimulus ranking experiment \
to discover which kind of visual content most strongly engages the human brain.

IMPORTANT: TRIBE v2 predicts the CORTICAL SURFACE only. You may ONLY choose a target ROI from this \
measurable cortical list (subcortical regions like amygdala/hippocampus are NOT available):
  FFA (faces), PPA (scenes/places), V1, V2 (low-level vision), STG, A1, STS, MTG (auditory/temporal), \
  IFG (language), AG, SMG, dlPFC, ACC, PCC, M1, precuneus, mPFC, OFC, insula.
For threat/fear/emotion questions, use a cortical proxy: insula, ACC, OFC, or mPFC (the cortical \
salience/affective network) — never amygdala.

Using neuroscience domain knowledge:
1. Choose a TARGET ROI FROM THE LIST ABOVE whose engagement is the primary ranking metric \
   (e.g. FFA for faces, PPA for scenes, V1 for low-level features, insula/ACC for threat salience).
2. Design 3-5 controlled visual CATEGORIES spanning a meaningful range (e.g. simple patterns, isolated \
   objects, faces, social scenes, threatening/fearful scenes). Keep them comparable on low-level features \
   where possible so the contrast is about content, not luminance/clutter.
3. For each category, write 2 concrete IMAGE generation prompts (photorealistic, centered, single clear \
   subject/scene, no text).
4. Give a one-line neuroscience rationale per category (cite known findings from domain knowledge).

Respond ONLY as JSON:
{
  "target_roi": "insula",
  "target_roi_rationale": "Anterior insula is the cortical hub for threat salience/interoception (Craig).",
  "categories": [
    {"name": "neutral patterns", "rationale": "Low salience baseline; minimal affective drive.",
     "image_prompts": ["a plain grid of gray squares", "a simple blue geometric pattern"]},
    {"name": "threatening scenes", "rationale": "High threat salience drives anterior insula/ACC (Ohman).",
     "image_prompts": ["a menacing snarling predator lunging", "a dark threatening figure in an alley"]}
  ]
}"""


def _design_ranking(question: str) -> dict:
    r = _client().chat.completions.create(
        model=MODEL, response_format={"type": "json_object"},
        messages=[{"role": "system", "content": RANKING_DESIGN_SYSTEM},
                  {"role": "user", "content": f"Research question: {question}"}],
    )
    return json.loads(r.choices[0].message.content or "{}")


def _global_engagement(roi_activation: dict) -> float:
    vals = [v.get("mean", 0.0) for v in roi_activation.values()]
    return float(sum(vals) / len(vals)) if vals else 0.0


async def run_generated_ranking(question: str) -> AsyncGenerator[dict, None]:
    """Generative stimulus exploration: agent designs controlled visual categories, generates
    real images, runs TRIBE v2, and ranks categories by target-ROI + whole-cortex engagement."""
    from cortex.tools.stimulus_gen import generate_image_clip
    import base64 as _b64

    trace = WorkshopTrace("cortex_stimulus_ranking", question)
    yield {"kind": "workshop", "url": "http://localhost:5899"}

    yield {"kind": "phase", "phase": "design", "message": "Designing controlled stimulus categories..."}
    try:
        with trace.span("design_ranking"):
            design = await asyncio.to_thread(_design_ranking, question)
    except Exception as e:
        yield {"kind": "error", "message": f"Design failed: {e}"}
        return

    target_roi = design.get("target_roi", "V1")
    categories = design.get("categories", [])
    if not categories:
        yield {"kind": "error", "message": "No categories could be designed."}
        return

    yield {"kind": "ranking_designed", "target_roi": target_roi,
           "target_roi_rationale": design.get("target_roi_rationale"),
           "categories": [{"name": c.get("name"), "rationale": c.get("rationale")} for c in categories]}

    try:
        infer_fn = _get_infer_bytes_fn()
    except TribeUnavailable as e:
        yield {"kind": "tribe_unavailable", "message": str(e)}
        return

    cat_results = []

    for ci, cat in enumerate(categories):
        name = cat.get("name", f"cat{ci}")
        prompts = list(cat.get("image_prompts", []))[:2]
        yield {"kind": "category_start", "index": ci, "name": name, "rationale": cat.get("rationale")}

        # Generate images -> clips (in parallel) for this category.
        for p in prompts:
            yield {"kind": "stimulus_generating", "category": name, "prompt": p}
        with trace.span(f"generate_images: {name}", category=name, n=len(prompts)):
            gens = await asyncio.gather(*[asyncio.to_thread(generate_image_clip, p) for p in prompts])
        gens = [g for g in gens if g.get("generated")]
        for g in gens:
            yield {"kind": "stimulus_ready", "category": name, "image_url": g.get("image_url"),
                   "url": g.get("url"), "stimulus_id": g.get("stimulus_id")}

        if not gens:
            yield {"kind": "category_result", "index": ci, "name": name, "n": 0,
                   "target_mean": None, "global_mean": None}
            continue

        # Run TRIBE v2 on the generated clips in parallel.
        bytes_list = [g["video_bytes"] for g in gens]
        ids_list = [g["stimulus_id"] for g in gens]
        roi_list = [[target_roi]] * len(gens)
        with trace.span(f"tribe_v2_inference: {name}", category=name, n=len(gens), target_roi=target_roi):
            results = await asyncio.to_thread(lambda: list(infer_fn.map(bytes_list, ids_list, roi_list)))

        targets, globals_ = [], []
        for g, res in zip(gens, results):
            roi_act = res.get("roi_activation", {})
            tval = roi_act.get(target_roi, {}).get("mean")
            if tval is None:
                tval = (res.get("top_rois") or [{}])[0].get("mean", 0.0)
            gval = _global_engagement(roi_act)
            targets.append(float(tval)); globals_.append(float(gval))
            surf_url = _save_b64(res.get("surface_png_b64"), f"{g['stimulus_id']}_surf.png")
            yield {"kind": "stimulus_result", "category": name, "stimulus_id": g["stimulus_id"],
                   "image_url": g.get("image_url"), "url": g.get("url"), "surface_url": surf_url,
                   "target_roi": target_roi, "target_value": float(tval),
                   "global_value": float(gval), "peak_roi": res.get("peak_roi")}

        tmean = sum(targets) / len(targets)
        gmean = sum(globals_) / len(globals_)
        cat_results.append({"name": name, "rationale": cat.get("rationale"),
                            "target_mean": tmean, "global_mean": gmean, "n": len(targets)})
        yield {"kind": "category_result", "index": ci, "name": name,
               "target_mean": tmean, "global_mean": gmean, "n": len(targets)}

    # Rank.
    ranked = sorted(cat_results, key=lambda c: c["target_mean"], reverse=True)
    for rank, c in enumerate(ranked):
        c["rank"] = rank + 1
    yield {"kind": "ranking", "target_roi": target_roi, "ordered": ranked}

    # Record the ranking outcome as signals so it shows on the Workshop timeline.
    for c in ranked:
        trace.signal(f"rank #{c['rank']}: {c['name']}", sentiment="POSITIVE" if c["rank"] == 1 else "",
                     reason=f"{target_roi}={c['target_mean']:.3f}, global={c['global_mean']:.3f}")

    yield {"kind": "phase", "phase": "synthesize", "message": "Writing ranked findings..."}
    with trace.span("synthesize_report"):
        report = await asyncio.to_thread(_synthesize_ranking_report, question, target_roi,
                                         design.get("target_roi_rationale"), ranked)
    yield {"kind": "report", "content": report}
    trace.finish(report)
    yield {"kind": "phase", "phase": "complete", "message": "Exploration complete"}


def _synthesize_ranking_report(question: str, target_roi: str, roi_rationale: str, ranked: list) -> str:
    r = _client().chat.completions.create(
        model=MODEL,
        messages=[
            {"role": "system", "content": "You are Cortex. Write a concise markdown report for a "
             "visual-stimulus RANKING experiment, grounded ONLY in the provided measured data. "
             "Include: Research Question, Method (controlled generated stimuli + TRIBE v2, target ROI), "
             "a markdown TABLE ranking categories by target-ROI engagement and whole-cortex engagement, "
             "Key Findings, and Limitations (static generated images; model-predicted not empirical). "
             "Do not invent numbers."},
            {"role": "user", "content": json.dumps({"question": question, "target_roi": target_roi,
                "target_roi_rationale": roi_rationale, "ranked_categories": ranked})},
        ],
    )
    return r.choices[0].message.content or ""


async def run_research(question: str) -> AsyncGenerator[dict, None]:
    """Run the full autonomous research loop, yielding structured events."""
    trace = WorkshopTrace("cortex_research", question)
    yield {"kind": "workshop", "url": "http://localhost:5899"}

    yield {"kind": "phase", "phase": "design", "message": "Designing experiments..."}

    try:
        with trace.span("design_experiments"):
            experiments = await asyncio.to_thread(_design_experiments, question)
    except Exception as e:
        yield {"kind": "error", "message": f"Experiment design failed: {e}"}
        return

    if not experiments:
        yield {"kind": "error", "message": "No experiments could be designed for this question."}
        return

    yield {"kind": "experiments_designed", "count": len(experiments),
           "experiments": [{"hypothesis": e.get("hypothesis"), "focal_roi": e.get("focal_roi"),
                            "condition_a": e.get("condition_a", {}).get("name"),
                            "condition_b": e.get("condition_b", {}).get("name")} for e in experiments]}

    completed_results = []

    for exp_idx, exp in enumerate(experiments):
        hypothesis = exp.get("hypothesis", "")
        focal_roi = exp.get("focal_roi", "FFA")
        cond_a = exp.get("condition_a", {})
        cond_b = exp.get("condition_b", {})
        name_a = cond_a.get("name", "A")
        name_b = cond_b.get("name", "B")
        prompts_a = list(cond_a.get("stimulus_prompts", [])) or [name_a]
        prompts_b = list(cond_b.get("stimulus_prompts", [])) or [name_b]

        yield {"kind": "experiment_start", "index": exp_idx, "hypothesis": hypothesis,
               "focal_roi": focal_roi, "prediction": exp.get("prediction"),
               "condition_a": name_a, "condition_b": name_b,
               "stop_condition": "p < 0.05 and |Cohen's d| >= 0.5, else expand to max iterations"}

        data_a: list[float] = []
        data_b: list[float] = []
        used_clip_ids: list[str] = []
        iteration = 0
        test_result: dict = {}
        status = "inconclusive"
        roi_meta: dict = {}

        while iteration < MAX_ITERATIONS:
            iteration += 1
            yield {"kind": "iteration_start", "experiment": exp_idx, "iteration": iteration,
                   "n_per_condition": N_PER_CONDITION, "prompts_a": prompts_a, "prompts_b": prompts_b}

            try:
                with trace.span(f"tribe_v2_batch_inference [exp{exp_idx} it{iteration}]",
                                focal_roi=focal_roi, n_per_condition=N_PER_CONDITION):
                    batch = await _run_batch(prompts_a, prompts_b, name_a, name_b, focal_roi,
                                             N_PER_CONDITION, used_clip_ids, tag=f"e{exp_idx}_i{iteration}")
            except TribeUnavailable as e:
                yield {"kind": "tribe_unavailable", "message": str(e)}
                return
            except Exception as e:
                yield {"kind": "error", "message": f"TRIBE batch failed: {e}"}
                return

            used_clip_ids.extend(batch.get("used_clip_ids", []))
            roi_meta = batch.get("roi_meta", {})
            clips_a = batch.get("clips_a", [])
            clips_b = batch.get("clips_b", [])

            # Stream each real clip + its measured focal-ROI value.
            for cond_name, clips in ((name_a, clips_a), (name_b, clips_b)):
                for c in clips:
                    yield {"kind": "datapoint", "experiment": exp_idx, "iteration": iteration,
                           "condition": cond_name, "stimulus_id": c.get("clip_id"),
                           "roi": focal_roi, "value": float(c.get("focal_value", 0.0)),
                           "peak_roi": c.get("peak_roi"), "caption": c.get("caption"),
                           "url": c.get("url"), "similarity": c.get("similarity")}

            # Real group-average + contrast cortical surface maps for this batch.
            yield {"kind": "maps", "experiment": exp_idx, "iteration": iteration,
                   "focal_roi": focal_roi, "roi_meta": roi_meta,
                   "group_a_url": batch.get("group_map_a_url"), "group_a_label": name_a,
                   "group_b_url": batch.get("group_map_b_url"), "group_b_label": name_b,
                   "contrast_url": batch.get("contrast_map_url")}

            data_a.extend(batch.get("values_a", []))
            data_b.extend(batch.get("values_b", []))

            test_result = two_sample_test(data_a, data_b)
            yield {"kind": "stats", "experiment": exp_idx, "iteration": iteration,
                   "focal_roi": focal_roi, "result": test_result}

            # Power-based stop condition.
            p = test_result.get("p_value")
            d = test_result.get("cohens_d")
            significant = (p is not None and p < 0.05 and d is not None and abs(d) >= 0.5)

            if significant:
                status = "supported" if (d or 0) > 0 else "discarded"
                stop_reason = f"significant (p={p:.3f}, d={d:.2f})"
            elif iteration >= MAX_ITERATIONS:
                status = "inconclusive"
                stop_reason = "max iterations reached without significance"
            else:
                status = "refined"
                stop_reason = "not yet significant — refining stimuli to reduce confounds"

            # Generate a natural-language interpretation, then self-monitor it for over-claiming.
            with trace.span("interpret_result", iteration=iteration):
                interp = await asyncio.to_thread(_interpret_result, hypothesis, focal_roi,
                                                 name_a, name_b, test_result)

            # --- Self-monitoring signals (Workshop spans + mirrored to UI) ---
            low_power = check_low_power_stats(test_result)
            if low_power["fired"]:
                trace.signal(low_power["name"], sentiment=low_power.get("sentiment", ""),
                             reason=low_power.get("reason", ""))
                yield {"kind": "signal", "experiment": exp_idx, "iteration": iteration, **low_power}

            overinterp = check_overinterpretation_prose(interp, test_result)
            if overinterp["fired"]:
                trace.signal(overinterp["name"], sentiment=overinterp.get("sentiment", ""),
                             reason=overinterp.get("reason", ""))
                yield {"kind": "signal", "experiment": exp_idx, "iteration": iteration, **overinterp}
                # Self-heal: rewrite the interpretation honestly.
                with trace.span("self_correct_interpretation"):
                    corrected = await asyncio.to_thread(_correct_interpretation, interp, test_result)
                trace.signal("self_correction", sentiment="POSITIVE",
                             reason="Rewrote over-stated interpretation to match the evidence.",
                             before=interp, after=corrected)
                yield {"kind": "signal", "experiment": exp_idx, "iteration": iteration,
                       "name": "self_correction", "sentiment": "POSITIVE",
                       "reason": "Caught over-interpretation and corrected it.",
                       "before": interp, "after": corrected}
                interp = corrected

            yield {"kind": "decision", "experiment": exp_idx, "iteration": iteration,
                   "status": status, "stop_reason": stop_reason, "interpretation": interp}

            if significant or iteration >= MAX_ITERATIONS:
                break

            # Feedback loop: inspect clips, diagnose confounds, propose cleaner prompts.
            yield {"kind": "phase", "phase": "refine", "message": "Refining stimuli to reduce confounds..."}
            refine = await asyncio.to_thread(_refine_prompts, hypothesis, focal_roi, name_a, name_b,
                                             clips_a, clips_b, test_result)
            if refine.get("prompts_a"):
                prompts_a = refine["prompts_a"]
            if refine.get("prompts_b"):
                prompts_b = refine["prompts_b"]
            if refine.get("diagnosis"):
                trace.signal("confound_detected", sentiment="POSITIVE", reason=refine.get("diagnosis"))
                yield {"kind": "signal", "experiment": exp_idx, "iteration": iteration,
                       "name": "confound_detected", "sentiment": "POSITIVE",
                       "reason": refine.get("diagnosis")}
            yield {"kind": "refinement", "experiment": exp_idx, "iteration": iteration,
                   "diagnosis": refine.get("diagnosis"), "prompts_a": prompts_a, "prompts_b": prompts_b}

        completed_results.append({
            "hypothesis": hypothesis,
            "prediction": exp.get("prediction"),
            "focal_roi": focal_roi,
            "roi_definition": roi_meta,
            "condition_a": name_a, "condition_b": name_b,
            "iterations": iteration,
            "n_a": len(data_a), "n_b": len(data_b),
            "final_statistics": test_result,
            "final_status": status,
        })
        # Final outcome signal for this experiment.
        trace.signal(f"hypothesis_{status}",
                     sentiment="POSITIVE" if status == "supported" else "NEGATIVE",
                     reason=f"{hypothesis} → {status}")
        yield {"kind": "experiment_complete", "index": exp_idx, "status": status,
               "result": completed_results[-1]}

    yield {"kind": "phase", "phase": "synthesize", "message": "Synthesizing report from real data..."}
    with trace.span("synthesize_report"):
        report = await asyncio.to_thread(_synthesize_report, question, completed_results)
    yield {"kind": "report", "content": report}
    trace.finish(report)

    yield {"kind": "phase", "phase": "complete", "message": "Research complete"}
