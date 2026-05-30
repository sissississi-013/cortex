"""FastAPI backend for Cortex web dashboard.

Streams the agent's hypothesis-test-revise loop as granular SSE events
so the frontend can render each step in real-time.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time
import uuid
from typing import Any

from dotenv import load_dotenv
from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

load_dotenv()

app = FastAPI(title="Cortex API", version="0.2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STIMULI_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "stimuli")
os.makedirs(STIMULI_DIR, exist_ok=True)
app.mount("/stimuli", StaticFiles(directory=STIMULI_DIR), name="stimuli")


def _sse(event_type: str, data: dict) -> dict:
    return {"event": event_type, "data": json.dumps(data)}


# ---------------------------------------------------------------------------
# Direct tool endpoints (for the Tool Explorer tab)
# ---------------------------------------------------------------------------

@app.get("/api/tools/predict_fmri")
async def api_predict_fmri(stimulus: str = Query(...), top_n: int = 10):
    from cortex.tools.tribe_fmri import predict_fmri_activation
    return predict_fmri_activation(stimulus, top_n=top_n)


@app.get("/api/tools/search_literature")
async def api_search_literature(query: str = Query(...)):
    from cortex.tools.nimare_literature import search_literature
    return search_literature(query)


@app.get("/api/tools/meta_analyze")
async def api_meta_analyze(term: str = Query(...)):
    from cortex.tools.nimare_literature import meta_analyze
    return meta_analyze(term)


@app.get("/api/tools/decode_brain_region")
async def api_decode_region(x: int = Query(...), y: int = Query(...), z: int = Query(...)):
    from cortex.tools.nimare_literature import decode_brain_region
    return decode_brain_region(x, y, z)


@app.get("/api/tools/analyze_eeg")
async def api_analyze_eeg(analysis_type: str = Query(...), condition: str = "auditory"):
    from cortex.tools.mne_eeg import analyze_eeg
    return analyze_eeg(analysis_type, condition)


@app.get("/api/tools/generate_image")
async def api_gen_image(description: str = Query(...)):
    from cortex.tools.stimulus_gen import generate_image_stimulus
    return generate_image_stimulus(description)


@app.get("/api/tools/generate_audio")
async def api_gen_audio(description: str = Query(...)):
    from cortex.tools.stimulus_gen import generate_audio_stimulus
    return generate_audio_stimulus(description)


# ---------------------------------------------------------------------------
# Research stream — the main event
# ---------------------------------------------------------------------------

@app.get("/api/research")
async def research_stream(question: str = Query(...)):
    """Stream the REAL autonomous research loop as granular SSE events.

    Designs experiments -> generates real video stimuli -> runs real TRIBE v2 on Modal
    -> extracts focal ROI -> real stats -> iterates to convergence -> report.
    """
    from cortex.research_loop import run_research

    async def event_generator():
        try:
            async for event in run_research(question):
                yield _sse(event.get("kind", "event"), event)
        except Exception as e:
            yield _sse("error", {"message": str(e)})

    return EventSourceResponse(event_generator())


@app.get("/api/health")
async def health():
    return {"status": "ok", "service": "cortex-api", "version": "0.2.0"}
