"""Cortex MCP Server — Brain tools for any agent.

Exposes TRIBE v2 fMRI prediction, NiMARE literature search / meta-analysis / decoding,
and MNE-Python EEG analysis as MCP tools via FastMCP.

Run standalone:
    python -m cortex.mcp_server              # stdio (for Claude Code, Cursor, etc.)
    python -m cortex.mcp_server --http 8742  # HTTP (for remote access / demos)
"""

from __future__ import annotations

import json
import sys
from typing import Any

from fastmcp import FastMCP

from cortex.tools.tribe_fmri import predict_fmri_activation
from cortex.tools.nimare_literature import (
    search_literature as _search_literature,
    meta_analyze as _meta_analyze,
    decode_brain_region as _decode_brain_region,
)
from cortex.tools.mne_eeg import analyze_eeg as _analyze_eeg
from cortex.tools.stimulus_gen import (
    generate_image_stimulus as _gen_image,
    generate_audio_stimulus as _gen_audio,
    generate_text_stimulus as _gen_text,
    generate_stimulus_pair as _gen_pair,
)

mcp = FastMCP("Cortex Brain Tools")


# ---------------------------------------------------------------------------
# MCP Tools
# ---------------------------------------------------------------------------


@mcp.tool
def predict_fmri(
    stimulus_text: str,
    subject_id: str = "average",
    top_n: int = 10,
) -> dict[str, Any]:
    """Predict fMRI brain activation for a stimulus using TRIBE v2.

    Given a natural-language description of a stimulus (speech, music, visual scene,
    face, text, emotion, motor action, etc.), predicts the fMRI BOLD response across
    ~20k cortical vertices on the fsaverage5 surface.

    Returns the top activated brain regions with MNI coordinates, activation intensity,
    hemisphere laterality, and anatomical labels.

    Args:
        stimulus_text: Natural language description of the stimulus
            (e.g. "a person speaking English", "classical piano music",
             "a photograph of a mountain landscape")
        subject_id: Subject identifier — 'average' uses population-level predictions
        top_n: Number of top activated regions to return (default: 10)
    """
    return predict_fmri_activation(stimulus_text, subject_id, top_n)


@mcp.tool
def search_literature(query: str, max_studies: int = 50) -> dict[str, Any]:
    """Search the Neurosynth database (~15k neuroimaging studies) for studies
    matching a cognitive or neuroscience term.

    Returns the number of matching studies and peak activation coordinates from
    coordinate-based meta-analysis.

    Args:
        query: Cognitive/neuroscience term to search
            (e.g. "speech perception", "fear", "working memory", "face perception")
        max_studies: Maximum number of peak clusters to return
    """
    return _search_literature(query, max_studies)


@mcp.tool
def meta_analyze(
    term: str,
    correction: str = "fwe",
    n_iters: int = 1000,
) -> dict[str, Any]:
    """Run ALE (Activation Likelihood Estimation) coordinate-based meta-analysis
    on neuroimaging studies matching a term.

    Computes convergent activation across studies using the ALE algorithm
    (Eickhoff et al., 2012) as implemented in NiMARE, with multiple comparisons
    correction.

    Returns significant activation clusters with MNI coordinates, z-scores,
    p-values, cluster sizes, and anatomical labels.

    Args:
        term: Cognitive/neuroscience term to meta-analyze
            (e.g. "speech perception", "music perception", "fear")
        correction: Multiple comparisons correction method — 'fwe' or 'fdr'
        n_iters: Number of permutations for Monte Carlo correction
    """
    return _meta_analyze(term, correction, n_iters)


@mcp.tool
def decode_brain_region(
    x: int,
    y: int,
    z: int,
    radius_mm: int = 10,
) -> dict[str, Any]:
    """Functional decoding: given MNI coordinates, identify what cognitive functions
    are associated with this brain region using reverse inference on Neurosynth.

    Uses the Neurosynth database to perform reverse inference — computing the
    posterior probability that a cognitive term is relevant given activation at
    the specified location.

    Args:
        x: MNI x coordinate (left-right, negative = left hemisphere)
        y: MNI y coordinate (posterior-anterior, negative = posterior)
        z: MNI z coordinate (inferior-superior, negative = inferior)
        radius_mm: Radius of the spherical ROI in millimeters
    """
    return _decode_brain_region(x, y, z, radius_mm)


@mcp.tool
def analyze_eeg(
    analysis_type: str,
    condition: str = "auditory",
    freq_band: str = "all",
) -> dict[str, Any]:
    """Analyze EEG/MEG data using MNE-Python on the MNE sample dataset.

    Supports three analysis types:
    - 'erp': Event-Related Potential analysis — peak latencies, amplitudes, components
    - 'time_frequency': Time-frequency decomposition — power changes in frequency bands
    - 'source_localization': Inverse solution — map scalp activity to cortical sources

    Args:
        analysis_type: 'erp', 'time_frequency', or 'source_localization'
        condition: 'auditory' or 'visual' (stimulus type in the sample dataset)
        freq_band: For time_frequency: 'delta', 'theta', 'alpha', 'beta', 'gamma', or 'all'
    """
    return _analyze_eeg(analysis_type, condition, freq_band)


# ---------------------------------------------------------------------------
# Stimulus Generation Tools
# ---------------------------------------------------------------------------


@mcp.tool
def generate_image(
    description: str,
    style: str = "photorealistic",
) -> dict[str, Any]:
    """Generate a visual stimulus image using DALL-E for fMRI/EEG experiments.

    The agent uses this to autonomously create controlled visual stimuli.
    For example, generating matched images of faces vs. scenes, or simple vs. complex patterns.

    Args:
        description: What the image should depict (e.g. "a human face with neutral expression",
            "a mountain landscape", "a complex geometric pattern")
        style: Image style — 'photorealistic', 'scientific', 'abstract'
    """
    return _gen_image(description, style)


@mcp.tool
def generate_audio(
    description: str,
    voice: str = "alloy",
    text_content: str = "",
) -> dict[str, Any]:
    """Generate an auditory stimulus using text-to-speech for auditory experiments.

    Creates speech stimuli for studying auditory processing, speech perception,
    language comprehension, etc.

    Args:
        description: Context for the audio (e.g. "natural English speech about weather")
        voice: TTS voice — 'alloy', 'echo', 'fable', 'onyx', 'nova', 'shimmer'
        text_content: Exact text to speak (if empty, agent auto-generates appropriate text)
    """
    return _gen_audio(description, voice, text_content if text_content else None)


@mcp.tool
def generate_text(
    description: str,
    length: str = "short",
) -> dict[str, Any]:
    """Generate a text stimulus for reading/language processing experiments.

    Creates controlled text passages for studying reading, semantic processing,
    syntactic parsing, etc.

    Args:
        description: What the text should be about (e.g. "a factual paragraph about the ocean")
        length: 'short' (1-2 sentences), 'medium' (paragraph), 'long' (2-3 paragraphs)
    """
    return _gen_text(description, length)


@mcp.tool
def generate_stimulus_pair(
    condition_a: str,
    condition_b: str,
    modality: str = "image",
) -> dict[str, Any]:
    """Generate a matched pair of stimuli for contrast experiments.

    Creates two stimuli that differ only in the experimental variable,
    enabling controlled A vs. B comparisons. The agent uses this to
    design its own experiments autonomously.

    Args:
        condition_a: First condition (e.g. "a human face with happy expression")
        condition_b: Second condition (e.g. "a human face with fearful expression")
        modality: 'image', 'audio', or 'text'
    """
    return _gen_pair(condition_a, condition_b, modality)


# ---------------------------------------------------------------------------
# MCP Resources
# ---------------------------------------------------------------------------


@mcp.resource("cortex://brain-atlas")
def brain_atlas() -> str:
    """Reference atlas of major brain regions, their functions, and MNI coordinates."""
    from cortex.tools.tribe_fmri import BRAIN_REGIONS

    lines = ["# Cortex Brain Atlas", "", "| Region | Label | Hemisphere | MNI (x,y,z) |", "|--------|-------|------------|-------------|"]
    for rid, info in BRAIN_REGIONS.items():
        coords = info["coords"]
        lines.append(f"| {rid} | {info['label']} | {info['hemisphere']} | ({coords[0]}, {coords[1]}, {coords[2]}) |")
    return "\n".join(lines)


@mcp.resource("cortex://available-tools")
def available_tools() -> str:
    """List of all brain tools available through the Cortex MCP server."""
    return """# Cortex Brain Tools

## predict_fmri
Predict fMRI brain activation for any stimulus using TRIBE v2.
Input: stimulus description (text). Output: activated brain regions with coordinates.

## search_literature
Search 15,000+ neuroimaging studies in the Neurosynth database.
Input: cognitive term. Output: matching studies and peak activation coordinates.

## meta_analyze
Run ALE meta-analysis across neuroimaging studies.
Input: cognitive term. Output: significant activation clusters with statistics.

## decode_brain_region
Functional decoding — what cognitive functions map to a brain location?
Input: MNI coordinates. Output: associated cognitive terms with probabilities.

## analyze_eeg
Analyze EEG data — ERP components, time-frequency, or source localization.
Input: analysis type + condition. Output: neural markers and statistics.
"""


@mcp.resource("cortex://available-datasets")
def available_datasets() -> str:
    """List of datasets available for analysis."""
    return """# Available Datasets

## MNE Sample Dataset
- Modality: MEG + EEG (306 MEG + 60 EEG channels)
- Paradigm: Auditory and visual stimulation
- Conditions: auditory/left, auditory/right, visual/left, visual/right
- Subjects: 1 (sample subject)
- Duration: ~5 minutes

## Neurosynth Database
- Type: Coordinate-based neuroimaging meta-analysis database
- Studies: 14,371 published fMRI/PET studies
- Coordinates: ~500,000 activation peaks
- Terms: 3,228 cognitive/neuroscience terms
- Access: via search_literature and meta_analyze tools

## TRIBE v2 Model
- Type: Multimodal brain encoding model (Meta, 2026)
- Training data: 1,000+ hours fMRI across 720 subjects
- Input: text, audio, or video stimuli
- Output: predicted BOLD response on fsaverage5 (~20k vertices)
- Access: via predict_fmri tool
"""


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """Run the MCP server."""
    if "--http" in sys.argv:
        idx = sys.argv.index("--http")
        port = int(sys.argv[idx + 1]) if idx + 1 < len(sys.argv) else 8742
        mcp.run(transport="http", host="0.0.0.0", port=port)
    else:
        mcp.run()


if __name__ == "__main__":
    main()
