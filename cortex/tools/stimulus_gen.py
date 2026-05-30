"""Autonomous stimulus generation — creates REAL video/image/audio/text stimuli.

TRIBE v2 ingests video (vision + audio + language), so the primary path generates
actual mp4 clips via OpenAI's Sora video API. If video generation is unavailable,
falls back to generating an image (gpt-image-1) and assembling it into a short
video clip locally with ffmpeg.

These files are what get fed into the real TRIBE v2 model on Modal.
"""

from __future__ import annotations

import base64
import os
import time
import uuid
from pathlib import Path
from typing import Any

from openai import OpenAI

STIMULI_DIR = Path(__file__).parent.parent.parent / "stimuli"
STIMULI_DIR.mkdir(exist_ok=True)


def _client() -> OpenAI:
    return OpenAI(api_key=os.getenv("OPENAI_API_KEY"))


# ---------------------------------------------------------------------------
# Video (primary path for TRIBE v2)
# ---------------------------------------------------------------------------

def generate_video_stimulus(
    description: str,
    seconds: int = 4,
    size: str = "720x1280",
) -> dict[str, Any]:
    """Generate a real video stimulus for TRIBE v2 using OpenAI Sora.

    Falls back to image->video if Sora is unavailable on this account.
    Returns the local mp4 path so it can be fed to the Modal TRIBE v2 job.
    """
    client = _client()
    stimulus_id = f"vid_{uuid.uuid4().hex[:8]}"
    filepath = STIMULI_DIR / f"{stimulus_id}.mp4"
    prompt = (
        f"A clear, well-lit, centered shot for a neuroscience visual experiment: "
        f"{description}. Steady camera, single subject, no text overlays."
    )

    try:
        video = client.videos.create_and_poll(
            model="sora-2",
            prompt=prompt,
            seconds=str(seconds),
            size=size,
        )
        if getattr(video, "status", None) == "completed":
            content = client.videos.download_content(video.id, variant="video")
            content.write_to_file(str(filepath))
            return {
                "stimulus_id": stimulus_id,
                "type": "video",
                "description": description,
                "filepath": str(filepath),
                "url": f"/stimuli/{filepath.name}",
                "seconds": seconds,
                "generated": True,
                "method": "sora-2",
            }
        # Not completed -> fall through to fallback.
        raise RuntimeError(f"Sora status={getattr(video, 'status', 'unknown')}")
    except Exception as e:
        fallback = _image_to_video(description, filepath, seconds)
        fallback["sora_error"] = str(e)[:200]
        fallback["stimulus_id"] = stimulus_id
        return fallback


def _image_to_video(description: str, out_path: Path, seconds: int) -> dict[str, Any]:
    """Fallback: generate an image with gpt-image-1 and assemble a short clip."""
    client = _client()
    try:
        img = client.images.generate(
            model="gpt-image-1",
            prompt=f"Photorealistic, centered, neuroscience visual stimulus: {description}. No text.",
            n=1,
            size="1024x1024",
        )
        b64 = img.data[0].b64_json
        img_path = out_path.with_suffix(".png")
        if b64:
            img_path.write_bytes(base64.b64decode(b64))
        else:
            return {"type": "video", "description": description, "generated": False,
                    "error": "no image data returned"}

        # Assemble a static clip (subtle is fine; TRIBE samples frames).
        import imageio.v3 as iio
        import numpy as np

        frame = iio.imread(img_path)
        fps = 8
        n_frames = max(1, seconds * fps)
        frames = np.stack([frame] * n_frames)
        iio.imwrite(out_path, frames, fps=fps, codec="libx264")

        return {
            "type": "video",
            "description": description,
            "filepath": str(out_path),
            "url": f"/stimuli/{out_path.name}",
            "image_url": f"/stimuli/{img_path.name}",
            "seconds": seconds,
            "generated": True,
            "method": "image-to-video (gpt-image-1 + ffmpeg)",
        }
    except Exception as e:
        return {"type": "video", "description": description, "generated": False, "error": str(e)}


def read_stimulus_bytes(filepath: str) -> bytes:
    """Read a generated stimulus file as bytes (for sending to Modal)."""
    return Path(filepath).read_bytes()


# ---------------------------------------------------------------------------
# Image / audio / text (secondary modalities)
# ---------------------------------------------------------------------------

def generate_image_stimulus(description: str, style: str = "photorealistic", size: str = "1024x1024") -> dict[str, Any]:
    """Generate a still-image stimulus using gpt-image-1."""
    client = _client()
    stimulus_id = f"img_{uuid.uuid4().hex[:8]}"
    try:
        resp = client.images.generate(
            model="gpt-image-1",
            prompt=f"{style} neuroscience experiment stimulus: {description}. Clean, centered, no text.",
            n=1,
            size=size,
        )
        b64 = resp.data[0].b64_json
        filepath = STIMULI_DIR / f"{stimulus_id}.png"
        if b64:
            filepath.write_bytes(base64.b64decode(b64))
        return {
            "stimulus_id": stimulus_id, "type": "image", "description": description,
            "filepath": str(filepath), "url": f"/stimuli/{filepath.name}",
            "generated": True, "method": "gpt-image-1",
        }
    except Exception as e:
        return {"stimulus_id": stimulus_id, "type": "image", "description": description,
                "generated": False, "error": str(e)}


def generate_audio_stimulus(description: str, voice: str = "alloy", text_content: str | None = None) -> dict[str, Any]:
    """Generate an auditory stimulus using OpenAI TTS."""
    client = _client()
    stimulus_id = f"aud_{uuid.uuid4().hex[:8]}"
    if text_content is None:
        r = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": f"Write a 1-2 sentence speech sample for an auditory neuroscience experiment about: {description}. Output only the speech text."}],
            max_tokens=100,
        )
        text_content = (r.choices[0].message.content or "").strip()
    try:
        filepath = STIMULI_DIR / f"{stimulus_id}.mp3"
        resp = client.audio.speech.create(model="tts-1", voice=voice, input=text_content)
        resp.stream_to_file(str(filepath))
        return {"stimulus_id": stimulus_id, "type": "audio", "description": description,
                "text_content": text_content, "voice": voice, "filepath": str(filepath),
                "url": f"/stimuli/{filepath.name}", "generated": True, "method": "tts-1"}
    except Exception as e:
        return {"stimulus_id": stimulus_id, "type": "audio", "description": description,
                "text_content": text_content, "generated": False, "error": str(e)}


def generate_text_stimulus(description: str, length: str = "short") -> dict[str, Any]:
    """Generate a text stimulus for reading experiments."""
    client = _client()
    stimulus_id = f"txt_{uuid.uuid4().hex[:8]}"
    guide = {"short": "1-2 sentences", "medium": "a paragraph", "long": "2-3 paragraphs"}.get(length, "1-2 sentences")
    try:
        r = client.chat.completions.create(
            model="gpt-4.1-mini",
            messages=[{"role": "user", "content": f"Write {guide} of text for a reading neuroscience experiment about: {description}. Output only the stimulus text."}],
            max_tokens=300,
        )
        text = (r.choices[0].message.content or "").strip()
        filepath = STIMULI_DIR / f"{stimulus_id}.txt"
        filepath.write_text(text)
        return {"stimulus_id": stimulus_id, "type": "text", "description": description,
                "content": text, "word_count": len(text.split()), "filepath": str(filepath),
                "generated": True}
    except Exception as e:
        return {"stimulus_id": stimulus_id, "type": "text", "description": description,
                "generated": False, "error": str(e)}
