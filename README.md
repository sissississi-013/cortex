# Cortex

<p align="center">
  <img src="assets/cortex-logo.png" alt="Cortex logo" width="220">
</p>

<p align="center">
  <a href="https://www.python.org"><img alt="Python 3.12+" src="https://img.shields.io/badge/Python-3.12%2B-3776AB?style=flat-square&logo=python&logoColor=white"></a>
  <a href="https://modelcontextprotocol.io"><img alt="MCP Server" src="https://img.shields.io/badge/MCP-Server-111111?style=flat-square"></a>
  <a href="https://openai.github.io/openai-agents-python/"><img alt="OpenAI Agents SDK" src="https://img.shields.io/badge/OpenAI-Agents_SDK-111111?style=flat-square"></a>
  <a href="https://modal.com"><img alt="Modal GPU" src="https://img.shields.io/badge/Modal-GPU-7C3AED?style=flat-square"></a>
  <a href="https://www.raindrop.ai"><img alt="Raindrop Tracing" src="https://img.shields.io/badge/Raindrop-Tracing-4ADE80?style=flat-square"></a>
</p>

Cortex closes the loop between hypothesis generation, stimulus design, brain simulation, and evidence accumulation.

It is an autonomous neuroscience research agent that can generate a testable claim, create stimuli, run brain tools, evaluate the result, and decide whether to keep, refine, or discard the hypothesis.

Built at the Autoresearch Systems Hackathon on May 30, 2026 with Modal, OpenAI, Raindrop, and Antler.



## Why Cortex

Autoresearch is not just an agent editing code until validation loss improves. The deeper pattern is a closed loop: generate a candidate, test it against reality, measure the result, and decide what survives.

For Cortex, the candidate is a neuroscience hypothesis. Reality is a brain simulation, EEG analysis, literature meta-analysis, or generated stimulus experiment. That makes neuroscience legible to agents as an optimization problem.

1. Turn a question into falsifiable neuroscience hypotheses.
2. Generate controlled stimuli for the experiment.
3. Predict or analyze brain activation with specialized tools.
4. Cross-check against literature and meta-analysis.
5. Keep, refine, or discard each hypothesis.
6. Produce a structured research report with limitations.

## What This Unlocks

- **Replication screening:** Before spending scanner time, Cortex can extract a claim from the literature and test whether it is computationally plausible across simulated brain responses.
- **BCI stimulus optimization:** Cortex can search images, audio, text, and timing parameters for stimuli that produce distinctive neural patterns.
- **Computational validation:** Instead of only summarizing papers, Cortex compares claims against brain tools and flags convergence, contradictions, and weak evidence.
- **Autonomous stimulus search:** Generative AI creates candidate stimuli, TRIBE v2 predicts cortical responses, and the agent keeps the stimuli that best test the hypothesis.

## Main Features

- **MCP brain tools:** Exposes fMRI prediction, literature search, ALE meta-analysis, coordinate decoding, EEG analysis, and stimulus generation through FastMCP.
- **Autonomous research loop:** Uses the OpenAI Agents SDK to decompose questions, run experiments, interpret results, and iterate.
- **Live demo UI:** Streams experiment design, generated stimuli, TRIBE results, statistics, decisions, and final report into a React dashboard.
- **GPU brain inference:** Runs TRIBE v2 jobs through Modal for cortical activation maps.
- **Traceable research quality:** Integrates with Raindrop Workshop signals for hypothesis quality, tool selection, overinterpretation, and loop stalls.

## Installation

Cortex requires Python 3.12 or higher.

```bash
git clone https://github.com/sissississi-013/cortex.git
cd cortex

python3.12 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

For the web dashboard:

```bash
cd web
npm install
```

## Quickstart

Run the local brain tools test. This does not require an OpenAI key.

```bash
python -m cortex test-tools
```

Run the MCP server for other agents:

```bash
python -m cortex server
```

Or run it over HTTP:

```bash
python -m cortex server --http 8742
```

Run the autonomous researcher:

```bash
export OPENAI_API_KEY=sk-...
python -m cortex research "How does the brain process speech vs music?"
```

## Web Demo

Start the API server:

```bash
uvicorn cortex.api:app --reload --port 8000
```

Start the frontend:

```bash
cd web
npm run dev
```

Open the Vite URL and ask a falsifiable neuroscience question, for example:

```text
Does the fusiform face area respond more to faces than houses?
```

The dashboard shows the full workflow: experiment design, generated stimuli, cortical maps, live logs, statistical decisions, and final report.

<img width="1928" height="3134" alt="image" src="https://github.com/user-attachments/assets/43c411f4-aa4a-4a37-af29-99d7ed46d9e1" />

## Generative Stimulus Exploration

The **Stimulus Exploration** mode answers a different kind of question: *which visual content most strongly engages the brain?* Instead of testing a single A-vs-B hypothesis, Cortex designs and runs a controlled, generative experiment end to end:

1. **Designs a controlled experiment.** From your question, the agent picks a measurable **target ROI** and a set of controlled visual categories (e.g. patterns, objects, faces, social scenes, threatening scenes), each with a neuroscience rationale. Because TRIBE v2 predicts the cortical surface only, the agent is constrained to measurable cortical regions — for threat/fear it uses cortical proxies like the insula or ACC rather than the (subcortical, unmeasurable) amygdala.
2. **Generates its own stimuli.** For each category it generates real images with OpenAI image generation and wraps them into short clips TRIBE v2 can ingest.
3. **Runs TRIBE v2 in parallel.** Every generated clip is scored on Modal GPUs (parallel fan-out), producing real `fsaverage5` cortical activation.
4. **Ranks by engagement.** Categories are ranked by **target-ROI activation** and by **whole-cortex engagement** (mean across all regions), shown as live ranking bars with per-category cortical surface maps.
5. **Reports.** A markdown findings report with a ranked results table and honest limitations (static generated images, model-predicted rather than empirical).

Run it from the **Stimulus Exploration** tab, or via the API:

```bash
curl -N "http://localhost:8000/api/explore?question=Which%20visual%20categories%20drive%20the%20strongest%20FFA%20response%3F"
```

This is generative-AI-driven stimulus search: the agent invents candidate stimuli, measures the predicted neural response, and tells you what the brain cares about most.

## MCP Tools

| Tool | Purpose | Backend |
| --- | --- | --- |
| `predict_fmri` | Predict cortical activation for a stimulus | TRIBE v2 on Modal |
| `search_literature` | Search neuroimaging studies | NiMARE and Neurosynth |
| `meta_analyze` | Run ALE coordinate meta-analysis | NiMARE |
| `decode_brain_region` | Decode cognitive functions from MNI coordinates | Neurosynth reverse inference |
| `analyze_eeg` | Analyze ERP, time-frequency, or source localization | MNE-Python |
| `generate_image` | Create controlled visual stimuli | OpenAI image generation |
| `generate_audio` | Create auditory stimuli | OpenAI text-to-speech |
| `generate_text` | Create reading stimuli | OpenAI text generation |
| `generate_stimulus_pair` | Create matched A/B experiment stimuli | Cortex stimulus generator |

## Connect From Cursor Or Claude Code

Add Cortex to your MCP config:

```json
{
  "cortex-brain-tools": {
    "command": "python",
    "args": ["-m", "cortex.mcp_server"],
    "cwd": "/path/to/cortex"
  }
}
```

Then ask your agent to use the Cortex tools for neuroscience questions, experiment design, or brain-region interpretation.

## Raindrop Workshop

A research agent is a black box: the part that matters is everything it does *after* you send the question — the hypotheses it forms, the tools it calls, the errors it catches, and how it corrects course. Cortex streams that entire internal trajectory into **Raindrop Workshop** as a nested, navigable trace.

**No API key required.** Workshop ingests raw OTLP at `http://localhost:5899/v1/traces`, so Cortex sends OpenTelemetry spans directly to your local Workshop — bypassing the cloud key gate. Just have Workshop running; Cortex auto-detects it.

A single research run appears as a span tree like:

```text
cortex_research
├─ design_experiments              # forming falsifiable hypotheses
├─ tribe_v2_batch_inference        # the real GPU brain simulation (per iteration)
├─ interpret_result                # reading its own results
├─ ⚠ signal: low_statistical_power      # caught: too few data points
├─ ⚠ signal: result_overinterpretation  # caught: over-claiming vs the stats
├─ ✓ signal: self_correction            # rewrote the claim to match evidence
├─ ✓ signal: confound_detected          # spotted a bad stimulus, re-aligned
└─ synthesize_report
```

The self-monitoring signals make the agent's reliability legible:

- `result_overinterpretation` — flags claims that go beyond the evidence (e.g. calling p = 0.14 "significant"), then triggers a self-correction.
- `low_statistical_power` — flags conclusions drawn on too little data.
- `confound_detected` — spots stimuli that undermine the contrast and refines them.
- `self_correction` — records the agent fixing its own over-statement.

Everything works locally with no key. If you do have a Raindrop write key, set it for full cloud tracing plus auto-instrumented LLM spans:

```bash
export RAINDROP_WRITE_KEY=rk_...   # optional — local Workshop tracing works without it
```

Open Workshop at `http://localhost:5899` and inspect any run to watch the agent reason, catch its own mistakes, and get back on track.

## Project Structure

```text
cortex/
  cortex/
    api.py                 # FastAPI server: /api/research + /api/explore (SSE)
    mcp_server.py          # FastMCP brain tools server
    agent.py               # OpenAI Agents SDK orchestrator
    research_loop.py       # Hypothesis-test-revise loop + generative ranking mode
    modal_app.py           # Modal GPU functions for TRIBE v2 (parallel inference, surface maps)
    stats.py               # t-test, Cohen's d, convergence / stop condition
    signals.py             # Self-monitoring signal checks
    workshop_trace.py      # Direct OTLP tracing into Raindrop Workshop (no key)
    tools/
      stimulus_gen.py      # Image, audio, text, and paired stimulus generation
      tribe_fmri.py        # fMRI prediction interface
      nimare_literature.py # Literature search, meta-analysis, decoding
      mne_eeg.py           # EEG analysis
  web/
    src/                   # React dashboard
  assets/                  # README visuals
```

## License

MIT
