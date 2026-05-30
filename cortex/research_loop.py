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

from pathlib import Path

from cortex.stats import two_sample_test, check_convergence

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

async def run_research(question: str) -> AsyncGenerator[dict, None]:
    """Run the full autonomous research loop, yielding structured events."""
    yield {"kind": "phase", "phase": "design", "message": "Designing experiments..."}

    try:
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

            interp = test_result.get("note") or (
                f"{name_a} mean={test_result.get('mean_a')}, {name_b} mean={test_result.get('mean_b')}, "
                f"p={p}, d={d}")
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
        yield {"kind": "experiment_complete", "index": exp_idx, "status": status,
               "result": completed_results[-1]}

    yield {"kind": "phase", "phase": "synthesize", "message": "Synthesizing report from real data..."}
    report = await asyncio.to_thread(_synthesize_report, question, completed_results)
    yield {"kind": "report", "content": report}
    yield {"kind": "phase", "phase": "complete", "message": "Research complete"}
