"""REAL TRIBE v2 inference on Modal GPUs.

This runs Meta's TRIBE v2 multimodal brain-encoding model on actual video stimuli
and returns real predicted fMRI activation on the fsaverage5 cortical surface.

The agent can parametrize which ROIs / surface views to focus on per research question.

Deploy:
    modal deploy cortex/modal_app.py

Then it is callable from the rest of the app via:
    fn = modal.Function.from_name("cortex-tribe", "run_tribe_inference")
    result = fn.remote(video_bytes=..., roi_focus=["FFA", "STG"])

TRIBE v2: d'Ascoli et al. 2026, "A foundation model of vision, audition, and
language for in-silico neuroscience". Weights: huggingface.co/facebook/tribev2
(CC BY-NC). Predicts ~20k fsaverage5 cortical vertices, offset 5s for hemodynamic lag.
"""

from __future__ import annotations

import modal

app = modal.App("cortex-tribe")

# Persisted cache for HF weights + atlases so cold starts after the first are fast.
weights_volume = modal.Volume.from_name("cortex-tribe-weights", create_if_missing=True)
WEIGHTS_DIR = "/weights"

# Persisted bank of real naturalistic video stimuli (MSR-VTT) + caption embedding index.
bank_volume = modal.Volume.from_name("cortex-stimulus-bank", create_if_missing=True)
BANK_DIR = "/bank"

# Image with TRIBE v2 and its scientific stack.
# The tribev2 repo may not be pip-installable directly, so we clone it and add to
# PYTHONPATH (and attempt an editable install, ignoring failure).
tribe_image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("git", "ffmpeg")
    .pip_install(
        "torch",
        "torchvision",
        "torchaudio",
        "numpy",
        "scipy",
        "nibabel",
        "nilearn",
        "huggingface_hub",
        "pandas",
        "einops",
        "transformers",
        "decord",
        "av",
        "pytorch-lightning",
        "sentence-transformers",
    )
    .run_commands(
        "git clone https://github.com/facebookresearch/tribev2.git /opt/tribev2",
        "pip install -e /opt/tribev2 || true",
    )
    .env({
        "HF_HOME": WEIGHTS_DIR,
        "HUGGINGFACE_HUB_CACHE": WEIGHTS_DIR,
        "PYTHONPATH": "/opt/tribev2",
    })
)

# Destrieux surface-atlas label -> friendly functional ROI name.
# Maps common functional regions the agent asks about to anatomical parcels on fsaverage5.
DESTRIEUX_TO_ROI = {
    "G_temp_sup-Lateral": "STG",            # superior temporal gyrus
    "G_temp_sup-Plan_tempo": "A1",          # planum temporale / auditory
    "S_temporal_sup": "STS",                # superior temporal sulcus
    "G_oc-temp_lat-fusifor": "FFA",         # fusiform (face area territory)
    "G_oc-temp_med-Parahip": "PPA",         # parahippocampal place area territory
    "G_front_inf-Triangul": "IFG",          # Broca / inferior frontal
    "G_front_inf-Opercular": "IFG",
    "G_pariet_inf-Angular": "AG",           # angular gyrus
    "G_pariet_inf-Supramar": "SMG",         # supramarginal gyrus
    "G_front_middle": "dlPFC",              # dorsolateral PFC
    "G_and_S_cingul-Ant": "ACC",            # anterior cingulate
    "G_cingul-Post-dorsal": "PCC",          # posterior cingulate
    "G_precentral": "M1",                   # primary motor
    "G_occipital_middle": "V2",
    "Pole_occipital": "V1",                 # occipital pole / primary visual
    "S_calcarine": "V1",
    "G_precuneus": "precuneus",
    "G_front_sup": "mPFC",
    "G_orbital": "OFC",
    "G_Ins_lg_and_S_cent_ins": "insula",
    "G_temporal_middle": "MTG",
}


_MODEL = None  # cached across warm-container calls


def _load_tribe_model():
    """Download (cached) + load TRIBE v2 once per container."""
    global _MODEL
    if _MODEL is not None:
        return _MODEL
    import os
    from huggingface_hub import snapshot_download
    from tribev2 import TribeModel

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    local_dir = snapshot_download(repo_id="facebook/tribev2", cache_dir=WEIGHTS_DIR)
    _MODEL = TribeModel.from_pretrained(
        checkpoint_dir=local_dir,
        checkpoint_name="best.ckpt",
        cache_folder=WEIGHTS_DIR,
        device="auto",
    )
    return _MODEL


@app.function(
    image=tribe_image,
    gpu="A10G",
    volumes={WEIGHTS_DIR: weights_volume},
    timeout=900,
    scaledown_window=300,
)
def run_tribe_inference(
    video_bytes: bytes,
    stimulus_id: str = "stim",
    roi_focus: list[str] | None = None,
) -> dict:
    """Run TRIBE v2 on a single video stimulus and return real fsaverage5 activation.

    Args:
        video_bytes: raw mp4 bytes of the stimulus
        stimulus_id: identifier for logging
        roi_focus: optional list of functional ROI names the agent wants to focus on
            (e.g. ["FFA", "STG"]). If None, returns all parcellated ROIs.

    Returns dict with per-ROI mean activation, peak ROI, and surface summary.
    """
    import os
    import tempfile
    import numpy as np

    os.makedirs(WEIGHTS_DIR, exist_ok=True)

    # ---- Write video to a temp file for the model loader ----
    with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
        f.write(video_bytes)
        video_path = f.name

    # ---- Load TRIBE v2 + run inference + aggregate ROIs ----
    model = _load_tribe_model()
    result = _infer_video_file(model, video_path, roi_focus)
    result["stimulus_id"] = stimulus_id
    return result


def _infer_video_file(model, video_path: str, roi_focus: list | None) -> dict:
    """Run TRIBE v2 on a video file and aggregate into named ROIs. Shared by all entry points."""
    import numpy as np

    events = model.get_events_dataframe(video_path=video_path)
    preds, segments = model.predict(events=events)
    preds = np.asarray(preds)  # (time, n_vertices) on fsaverage5

    vertex_mean = preds.mean(axis=0)
    n_vertices = int(vertex_mean.shape[0])
    roi_activation = _aggregate_rois(vertex_mean, weights_dir=WEIGHTS_DIR)

    # Render the REAL fsaverage5 cortical surface activation map.
    surface_png_b64 = _render_surface_png(vertex_mean, weights_dir=WEIGHTS_DIR)

    weights_volume.commit()

    sorted_rois = sorted(roi_activation.items(), key=lambda kv: kv[1]["mean"], reverse=True)
    focus = None
    if roi_focus:
        focus = {r: roi_activation[r] for r in roi_focus if r in roi_activation}

    return {
        "model": "TRIBE_v2",
        "real_inference": True,
        "surface": "fsaverage5",
        "n_vertices": n_vertices,
        "n_timepoints": int(preds.shape[0]),
        "roi_activation": roi_activation,
        "roi_focus": focus,
        "peak_roi": sorted_rois[0][0] if sorted_rois else None,
        "surface_png_b64": surface_png_b64,
        "top_rois": [
            {"roi": r, "mean": float(v["mean"]), "label": v["label"]}
            for r, v in sorted_rois[:10]
        ],
    }


def _infer_video_file_light(model, video_path: str, roi_focus: list | None) -> dict:
    """Inference WITHOUT surface rendering — returns ROI activation + raw vertex map.
    Used for parallel batch runs where we render only group/contrast maps afterward."""
    import numpy as np
    events = model.get_events_dataframe(video_path=video_path)
    preds, _ = model.predict(events=events)
    preds = np.asarray(preds)
    vertex_mean = preds.mean(axis=0)
    roi_activation = _aggregate_rois(vertex_mean, weights_dir=WEIGHTS_DIR)
    sorted_rois = sorted(roi_activation.items(), key=lambda kv: kv[1]["mean"], reverse=True)
    return {
        "roi_activation": roi_activation,
        "peak_roi": sorted_rois[0][0] if sorted_rois else None,
        "vertex_mean": [float(x) for x in vertex_mean],
        "n_vertices": int(vertex_mean.shape[0]),
    }


@app.function(
    image=tribe_image, gpu="A10G",
    volumes={WEIGHTS_DIR: weights_volume, BANK_DIR: bank_volume},
    timeout=900, scaledown_window=300, max_containers=12,
)
def infer_clip_path(clip_path: str, roi_focus: list | None = None) -> dict:
    """Run TRIBE v2 on a clip already present in the bank volume. Parallel-friendly."""
    model = _load_tribe_model()
    out = _infer_video_file_light(model, clip_path, roi_focus)
    out["clip_path"] = clip_path
    return out


def _retrieve_clips(prompts: list, n: int, exclude: set) -> list:
    """Retrieve n distinct MSR-VTT clips best matching a list of prompts (round-robin)."""
    import numpy as np
    bank = _load_bank()
    embedder = _load_embedder()
    qs = embedder.encode(prompts, normalize_embeddings=True)
    sims = np.stack([bank["embs"] @ np.asarray(q, dtype="float32") for q in qs])  # (P, Nclips)
    used = set(exclude or [])
    picked = []
    pi = 0
    guard = 0
    while len(picked) < n and guard < n * len(prompts) + 50:
        guard += 1
        order = np.argsort(-sims[pi % len(prompts)])
        for idx in order:
            vid = str(bank["ids"][idx])
            if vid not in used:
                used.add(vid)
                picked.append({"clip_id": vid, "path": str(bank["paths"][idx]),
                               "caption": str(bank["texts"][idx]),
                               "similarity": float(sims[pi % len(prompts)][idx])})
                break
        pi += 1
    return picked


@app.function(
    image=tribe_image,
    volumes={WEIGHTS_DIR: weights_volume, BANK_DIR: bank_volume},
    timeout=2400,
)
def run_experiment_batch(
    prompts_a: list,
    prompts_b: list,
    name_a: str,
    name_b: str,
    focal_roi: str,
    n_per_condition: int = 6,
    exclude_ids: list | None = None,
) -> dict:
    """Run a full A-vs-B contrast experiment with parallel TRIBE inference.

    Retrieves N real clips per condition, runs TRIBE v2 across many GPU containers
    in parallel, then renders real group-average + contrast (A-B) cortical surface maps.
    """
    import base64
    import numpy as np

    exclude = set(exclude_ids or [])
    clips_a = _retrieve_clips(prompts_a, n_per_condition, exclude)
    exclude |= {c["clip_id"] for c in clips_a}
    clips_b = _retrieve_clips(prompts_b, n_per_condition, exclude)

    all_clips = clips_a + clips_b
    paths = [c["path"] for c in all_clips]
    roi_args = [[focal_roi]] * len(paths)

    # Fan out inference across GPU containers in parallel.
    results = list(infer_clip_path.map(paths, roi_args))
    by_path = {r["clip_path"]: r for r in results}

    def collect(clips):
        vals, vmaps, details = [], [], []
        for c in clips:
            r = by_path.get(c["path"])
            if not r:
                continue
            roi = r["roi_activation"].get(focal_roi, {})
            val = roi.get("mean")
            if val is None:
                val = (sorted(r["roi_activation"].items(), key=lambda kv: kv[1]["mean"], reverse=True) or [("", {"mean": 0})])[0][1]["mean"]
            vals.append(float(val))
            vmaps.append(np.asarray(r["vertex_mean"], dtype=float))
            try:
                with open(c["path"], "rb") as f:
                    vb = base64.b64encode(f.read()).decode()
            except Exception:
                vb = None
            details.append({"clip_id": c["clip_id"], "caption": c["caption"],
                            "similarity": c["similarity"], "focal_value": float(val),
                            "peak_roi": r["peak_roi"], "video_b64": vb})
        return vals, vmaps, details

    vals_a, vmaps_a, details_a = collect(clips_a)
    vals_b, vmaps_b, details_b = collect(clips_b)

    mean_a = np.mean(vmaps_a, axis=0) if vmaps_a else None
    mean_b = np.mean(vmaps_b, axis=0) if vmaps_b else None

    surf_a = _render_surface_png(mean_a, WEIGHTS_DIR) if mean_a is not None else None
    surf_b = _render_surface_png(mean_b, WEIGHTS_DIR) if mean_b is not None else None
    contrast_png = None
    if mean_a is not None and mean_b is not None:
        contrast_png = _render_surface_png(mean_a - mean_b, WEIGHTS_DIR, diverging=True)

    # ROI provenance for transparency.
    sample_roi = (by_path.get(clips_a[0]["path"]) if clips_a else None)
    roi_meta = {}
    if sample_roi:
        info = sample_roi["roi_activation"].get(focal_roi, {})
        roi_meta = {"focal_roi": focal_roi,
                    "destrieux_parcels": info.get("destrieux_parcels"),
                    "n_vertices": info.get("n_vertices")}

    return {
        "focal_roi": focal_roi,
        "name_a": name_a, "name_b": name_b,
        "values_a": vals_a, "values_b": vals_b,
        "clips_a": details_a, "clips_b": details_b,
        "group_map_a_b64": surf_a, "group_map_b_b64": surf_b,
        "contrast_map_b64": contrast_png,
        "roi_meta": roi_meta,
        "n_a": len(vals_a), "n_b": len(vals_b),
        "used_clip_ids": [c["clip_id"] for c in all_clips],
    }


def _render_surface_png(vertex_mean, weights_dir: str, diverging: bool = False):
    """Render real TRIBE v2 fsaverage5 surface activation as a brain map PNG (base64)."""
    import io
    import base64
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
        from nilearn import datasets, plotting

        fs = datasets.fetch_surf_fsaverage("fsaverage5", data_dir=weights_dir)
        vm = np.asarray(vertex_mean, dtype=float)
        half = len(vm) // 2
        lh, rh = vm[:half], vm[half:half * 2]
        vmax = float(np.percentile(np.abs(vm), 99)) or float(np.abs(vm).max() or 1.0)
        cmap = "cold_hot" if diverging else "inferno"
        thr = vmax * 0.05

        bg = "#0a0f0c"
        fig, axes = plt.subplots(1, 4, subplot_kw={"projection": "3d"}, figsize=(13, 3.0))
        specs = [
            (fs["infl_left"], lh, "left", "lateral", fs.get("sulc_left"), axes[0]),
            (fs["infl_left"], lh, "left", "medial", fs.get("sulc_left"), axes[1]),
            (fs["infl_right"], rh, "right", "lateral", fs.get("sulc_right"), axes[2]),
            (fs["infl_right"], rh, "right", "medial", fs.get("sulc_right"), axes[3]),
        ]
        for mesh, data, hemi, view, bgmap, ax in specs:
            plotting.plot_surf_stat_map(
                mesh, data, hemi=hemi, view=view, bg_map=bgmap,
                cmap=cmap, vmax=vmax, threshold=thr,
                colorbar=False, bg_on_data=True, axes=ax, figure=fig,
            )
            ax.set_facecolor(bg)
        fig.patch.set_facecolor(bg)
        fig.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0)
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=110, facecolor=bg, bbox_inches="tight", pad_inches=0)
        plt.close(fig)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception as e:
        print(f"[surface render failed] {type(e).__name__}: {e}")
        return None


# ---------------------------------------------------------------------------
# Real video stimulus bank (MSR-VTT) — retrieve real clips by caption similarity
# ---------------------------------------------------------------------------

_BANK = None  # cached (embeddings, ids, captions) per container
_EMBEDDER = None


def _load_embedder():
    global _EMBEDDER
    if _EMBEDDER is None:
        from sentence_transformers import SentenceTransformer
        _EMBEDDER = SentenceTransformer("all-MiniLM-L6-v2", cache_folder=BANK_DIR)
    return _EMBEDDER


@app.function(image=tribe_image, volumes={BANK_DIR: bank_volume}, timeout=3600)
def prepare_stimulus_bank() -> dict:
    """Download MSR-VTT clips + captions and build a caption embedding index (one-time)."""
    import os
    import json
    import zipfile
    import numpy as np
    from huggingface_hub import hf_hub_download

    os.makedirs(BANK_DIR, exist_ok=True)
    videos_dir = os.path.join(BANK_DIR, "videos")

    if not os.path.isdir(videos_dir) or len(os.listdir(videos_dir)) < 100:
        zip_path = hf_hub_download("friedrichor/MSR-VTT", "MSRVTT_Videos.zip",
                                   repo_type="dataset", cache_dir=BANK_DIR)
        os.makedirs(videos_dir, exist_ok=True)
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(videos_dir)

    data_path = hf_hub_download("friedrichor/MSR-VTT", "raw_data/MSRVTT_data.json",
                                repo_type="dataset", cache_dir=BANK_DIR)
    with open(data_path) as f:
        data = json.load(f)

    # Map video_id -> joined captions.
    caps: dict[str, list[str]] = {}
    for s in data.get("sentences", []):
        caps.setdefault(s["video_id"], []).append(s["caption"])

    # Resolve actual file paths (videos may be nested).
    file_index: dict[str, str] = {}
    for root, _, fnames in os.walk(videos_dir):
        for fn in fnames:
            if fn.endswith((".mp4", ".avi", ".webm")):
                vid = os.path.splitext(fn)[0]
                file_index[vid] = os.path.join(root, fn)

    ids, texts, paths = [], [], []
    for vid, path in file_index.items():
        if vid in caps:
            ids.append(vid)
            texts.append(" ".join(caps[vid][:5]))
            paths.append(path)

    embedder = _load_embedder()
    embs = embedder.encode(texts, batch_size=128, show_progress_bar=False, normalize_embeddings=True)

    np.savez(os.path.join(BANK_DIR, "index.npz"),
             ids=np.array(ids), texts=np.array(texts, dtype=object),
             paths=np.array(paths, dtype=object), embs=np.asarray(embs, dtype="float32"))
    bank_volume.commit()
    return {"status": "ready", "n_clips": len(ids)}


def _load_bank():
    global _BANK
    if _BANK is None:
        import os
        import numpy as np
        d = np.load(os.path.join(BANK_DIR, "index.npz"), allow_pickle=True)
        _BANK = {"ids": d["ids"], "texts": d["texts"], "paths": d["paths"], "embs": d["embs"]}
    return _BANK


@app.function(
    image=tribe_image,
    gpu="A10G",
    volumes={WEIGHTS_DIR: weights_volume, BANK_DIR: bank_volume},
    timeout=900,
    scaledown_window=300,
)
def retrieve_and_run(
    condition_prompt: str,
    roi_focus: list | None = None,
    exclude_ids: list | None = None,
) -> dict:
    """Retrieve the best-matching real MSR-VTT clip for a condition and run TRIBE v2 on it.

    Returns real ROI activation + which clip was used (id, caption) + the clip bytes
    so the frontend can display the actual stimulus the model saw.
    """
    import os
    import numpy as np

    bank = _load_bank()
    embedder = _load_embedder()
    q = embedder.encode([condition_prompt], normalize_embeddings=True)[0]
    sims = bank["embs"] @ np.asarray(q, dtype="float32")

    exclude = set(exclude_ids or [])
    order = np.argsort(-sims)
    chosen = None
    for idx in order:
        vid = str(bank["ids"][idx])
        if vid not in exclude:
            chosen = idx
            break
    if chosen is None:
        chosen = int(order[0])

    vid = str(bank["ids"][chosen])
    path = str(bank["paths"][chosen])
    caption = str(bank["texts"][chosen])
    similarity = float(sims[chosen])

    model = _load_tribe_model()
    result = _infer_video_file(model, path, roi_focus)
    result.update({
        "stimulus_id": vid,
        "clip_id": vid,
        "caption": caption,
        "similarity": similarity,
        "condition_prompt": condition_prompt,
        "source": "MSR-VTT",
    })
    try:
        with open(path, "rb") as f:
            result["video_bytes"] = f.read()
    except Exception:
        result["video_bytes"] = None
    return result


def _aggregate_rois(vertex_mean, weights_dir: str) -> dict:
    """Aggregate per-vertex activation into named functional ROIs using Destrieux atlas."""
    import numpy as np
    from nilearn import datasets

    fsaverage = datasets.fetch_surf_fsaverage("fsaverage5", data_dir=weights_dir)
    destrieux = datasets.fetch_atlas_surf_destrieux(data_dir=weights_dir)

    labels = [l.decode() if isinstance(l, bytes) else l for l in destrieux["labels"]]
    annot_lh = np.asarray(destrieux["map_left"])
    annot_rh = np.asarray(destrieux["map_right"])
    full_annot = np.concatenate([annot_lh, annot_rh])

    # Align lengths defensively.
    n = min(len(full_annot), len(vertex_mean))
    full_annot = full_annot[:n]
    vm = np.asarray(vertex_mean)[:n]

    n_lh = len(annot_lh)

    roi_activation: dict = {}
    for parcel_idx, parcel_label in enumerate(labels):
        roi_name = DESTRIEUX_TO_ROI.get(parcel_label)
        if roi_name is None:
            continue
        mask = full_annot == parcel_idx
        if not mask.any():
            continue
        vals = vm[mask]
        # Hemisphere split for laterality.
        lh_mask = mask.copy()
        lh_mask[n_lh:] = False
        rh_mask = mask.copy()
        rh_mask[:n_lh] = False

        entry = roi_activation.setdefault(roi_name, {
            "label": roi_name,
            "destrieux_parcels": [],
            "_vals": [],
            "_lh": [],
            "_rh": [],
        })
        entry["destrieux_parcels"].append(parcel_label)
        entry["_vals"].extend(vals.tolist())
        if lh_mask.any():
            entry["_lh"].extend(vm[lh_mask].tolist())
        if rh_mask.any():
            entry["_rh"].extend(vm[rh_mask].tolist())

    # Finalize stats.
    out: dict = {}
    for roi, e in roi_activation.items():
        vals = np.asarray(e["_vals"], dtype=float)
        lh = np.asarray(e["_lh"], dtype=float) if e["_lh"] else np.array([])
        rh = np.asarray(e["_rh"], dtype=float) if e["_rh"] else np.array([])
        out[roi] = {
            "label": roi,
            "mean": float(vals.mean()) if vals.size else 0.0,
            "std": float(vals.std()) if vals.size else 0.0,
            "n_vertices": int(vals.size),
            "lh_mean": float(lh.mean()) if lh.size else None,
            "rh_mean": float(rh.mean()) if rh.size else None,
            "destrieux_parcels": e["destrieux_parcels"],
        }
    return out


@app.function(image=tribe_image, volumes={WEIGHTS_DIR: weights_volume}, timeout=1800, gpu="A10G")
def warm_weights() -> dict:
    """Pre-download TRIBE v2 weights + atlases into the volume to speed up later runs."""
    import os
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    from nilearn import datasets

    _load_tribe_model()
    datasets.fetch_surf_fsaverage("fsaverage5", data_dir=WEIGHTS_DIR)
    datasets.fetch_atlas_surf_destrieux(data_dir=WEIGHTS_DIR)
    weights_volume.commit()
    return {"status": "warmed"}


@app.function(image=tribe_image, volumes={WEIGHTS_DIR: weights_volume}, timeout=1800)
def download_and_inspect_weights() -> dict:
    """Download facebook/tribev2 from HF and report the checkpoint layout + load test."""
    import os
    from huggingface_hub import snapshot_download

    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    local_dir = snapshot_download(repo_id="facebook/tribev2", cache_dir=WEIGHTS_DIR)

    files = []
    for root, _, fnames in os.walk(local_dir):
        for fn in fnames:
            full = os.path.join(root, fn)
            rel = os.path.relpath(full, local_dir)
            files.append({"path": rel, "size_mb": round(os.path.getsize(full) / 1e6, 2)})

    out = {"local_dir": local_dir, "files": files}

    # Try to load using the discovered checkpoint.
    ckpts = [f["path"] for f in files if f["path"].endswith((".ckpt", ".pt", ".pth"))]
    out["checkpoints"] = ckpts
    if ckpts:
        try:
            from tribev2 import TribeModel
            ckpt_rel = ckpts[0]
            ckpt_dir = os.path.join(local_dir, os.path.dirname(ckpt_rel)) or local_dir
            ckpt_name = os.path.basename(ckpt_rel)
            model = TribeModel.from_pretrained(
                checkpoint_dir=ckpt_dir,
                checkpoint_name=ckpt_name,
                cache_folder=WEIGHTS_DIR,
                device="auto",
            )
            out["load_ok"] = True
            out["loaded_ckpt_dir"] = ckpt_dir
            out["loaded_ckpt_name"] = ckpt_name
        except Exception as e:
            out["load_ok"] = False
            out["load_err"] = f"{type(e).__name__}: {e}"

    weights_volume.commit()
    return out


@app.function(image=tribe_image, timeout=300)
def inspect_api() -> dict:
    """Introspect the real tribev2 API so we can match signatures exactly."""
    import inspect
    import tribev2

    out = {"tribev2_attrs": [a for a in dir(tribev2) if not a.startswith("_")]}

    try:
        from tribev2 import TribeModel
        out["TribeModel_methods"] = [m for m in dir(TribeModel) if not m.startswith("_")]
        try:
            out["from_pretrained_sig"] = str(inspect.signature(TribeModel.from_pretrained))
        except Exception as e:
            out["from_pretrained_sig_err"] = str(e)
        for meth in ("get_events_dataframe", "predict", "__init__"):
            try:
                out[f"{meth}_sig"] = str(inspect.signature(getattr(TribeModel, meth)))
            except Exception as e:
                out[f"{meth}_sig_err"] = str(e)
    except Exception as e:
        out["TribeModel_err"] = str(e)

    # demo_utils often holds the inference helper.
    try:
        import tribev2.demo_utils as du
        out["demo_utils_attrs"] = [a for a in dir(du) if not a.startswith("_")]
    except Exception as e:
        out["demo_utils_err"] = str(e)

    return out


@app.local_entrypoint()
def main():
    """Quick smoke test: warm the weights volume."""
    print(warm_weights.remote())
