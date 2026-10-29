# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "colour==0.1.5",
#     "diffusers==0.42.1",
#     "gradio==6.31.0",
#     "huggingface-hub==1.33.0",
#     "imageio-ffmpeg==0.7.0",
#     "ipython==9.17.1",
#     "ltx-pipelines==1.0.0",
#     "numpy==2.5.4",
#     "openimageio==3.2.1.3",
#     "torch==2.15.0",
#     "torchaudio==2.11.1",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():
    def _():
        import requests
        import zipfile
        import io
        from pathlib import Path

        WORKDIR = Path("/tmp/ltx23-workspace")
        WORKDIR.mkdir(parents=True, exist_ok=True)

        url = "https://github.com/Lightricks/LTX-2/archive/refs/heads/main.zip"

        response = requests.get(url, timeout=180)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
            archive.extractall(WORKDIR)

        matches = list(WORKDIR.glob("LTX-2-main"))

        if not matches:
            raise RuntimeError("سورس رسمی پیدا نشد؛ فایل ZIP را بررسی کن.")

        SOURCE_DIR = matches[0]

        print("Source:", SOURCE_DIR)
        print("Files:", len(list(SOURCE_DIR.rglob("*"))))
        return print("Downloaded successfully.")


    _()
    return


@app.cell
def _():
    def _():
        import subprocess
        import sys

        packages = [
            "accelerate>=1.2.1",
            "gradio",
            "imageio",
            "imageio-ffmpeg",
            "av",
            "sentencepiece",
        ]

        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--disable-pip-version-check",
                *packages,
            ],
            capture_output=True,
            text=True,
        )

        print(result.stdout[-12000:])

        if result.returncode != 0:
            print(result.stderr[-8000:])
            raise RuntimeError("نصب وابستگی‌ها موفق نبود.")
        return print("نصب وابستگی‌ها با موفقیت تمام شد.")


    _()
    return


@app.cell
def _():
    def _():
        from getpass import getpass
        from huggingface_hub import login

        token = getpass("Hugging Face access token: ").strip()

        if not token:
            print("No token entered.")
            return

        login(token=token)
        print("Hugging Face authentication completed.")

    _()
    return


@app.cell
def _():
    def _():
        from huggingface_hub import hf_hub_download
        from pathlib import Path

        MODEL_DIR = Path("/tmp/ltx23-model")
        MODEL_DIR.mkdir(parents=True, exist_ok=True)

        files = [
            "ltx-2.3-22b-distilled-1.1.safetensors",
            "ltx-2.3-spatial-upscaler-x2-1.1.safetensors",
        ]

        for filename in files:
            print("Downloading:", filename)

            path = hf_hub_download(
                repo_id="Lightricks/LTX-2.3",
                filename=filename,
                local_dir=str(MODEL_DIR),
            )

            print("Saved:", path)

    _()
    return


@app.cell
def _():

    def _():
        from huggingface_hub import snapshot_download
        from pathlib import Path

        GEMMA_DIR = Path("/tmp/ltx23-gemma")

        path = snapshot_download(
            repo_id="google/gemma-3-12b-it-qat-q4_0-unquantized",
            local_dir=str(GEMMA_DIR),
        )

        print("Gemma downloaded to:", path)

    _()
    return


@app.cell
def _():
    def _():
        import sys
        import subprocess

        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--force-reinstall", "--no-deps",
                "colour-science==0.4.7",
            ],
            capture_output=True, text=True,
        )

        print(result.stdout[-2000:])

        if result.returncode != 0:
            print(result.stderr[-2000:])
            raise RuntimeError("Reinstall colour-science failed.")

        print("colour-science reinstalled successfully.")


    _()
    return


@app.cell
def _():
    def _():
        import sys
        import subprocess

        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--no-cache-dir",
                "--force-reinstall",
                "--no-deps",
                "torchaudio==2.11.1+cu132",
                "--index-url",
                "https://download.pytorch.org/whl/test/cu132",
            ],
            capture_output=True, text=True,
        )

        print(result.stdout[-2000:])

        if result.returncode != 0:
            print(result.stderr[-2000:])
            raise RuntimeError("نصب torchaudio ناموفق بود.")

        print("torchaudio installed successfully.")


    _()
    return


@app.cell
def _():
    def _():
        import sys
        import subprocess

        cmd = [
            sys.executable, "-m", "pip", "install",
            "--no-cache-dir",
            "--force-reinstall",
            "--no-deps",
            "torchvision==0.30.0.dev20261028+cu132",
            "--index-url",
            "https://download.pytorch.org/whl/nightly/cu132",
        ]

        result = subprocess.run(cmd, capture_output=True, text=True)

        print(result.stdout[-2000:])

        if result.returncode != 0:
            print(result.stderr[-2000:])
            raise RuntimeError("نصب torchvision ناموفق بود.")

        print("torchvision installed successfully.")


    _()
    return


@app.cell
def _():
    import marimo as _mo
    _mo.md("""
    ### ⚠️ Restart the kernel

    After installing the packages above, **restart the marimo kernel** (from the menu: **Kernel → Restart**), then continue running the cells below.

    This is needed so that `torch`, `torchvision`, `torchaudio`, and `colour` are freshly imported with their new versions.
    """)
    return


@app.cell
def _():
    import subprocess, sys

    subprocess.run([
        sys.executable, "-m", "pip", "install",
        "--force-reinstall",
        "torch", "torchvision", "torchaudio",
        "--index-url", "https://download.pytorch.org/whl/cu130",
    ], check=True)
    return


@app.cell
def _():
    def _():
        import sys
        import subprocess
        import importlib

        # --- Step 1: Check current torch version ---
        import torch
        print("Torch:", torch.__version__)
        print("Torch CUDA:", torch.version.cuda)
        print("CUDA available:", torch.cuda.is_available())

        # --- Step 2: Check torchaudio compatibility ---
        _ta_ok = False
        try:
            import torchaudio
            print("TorchAudio:", torchaudio.__version__)
            # Force-load the C++ extension
            from torchaudio._extension import _load_lib
            _load_lib("libtorchaudio")
            _ta_ok = True
            print("TorchAudio C++ library: OK")
        except Exception as _e:
            print("TorchAudio C++ library: FAILED —", str(_e)[:200])

        # --- Step 3: Check torchvision compatibility ---
        _tv_ok = False
        try:
            import torchvision
            print("Torchvision:", torchvision.__version__)
            from torchvision.ops import nms as _nms
            _b = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
            _s = torch.tensor([0.9])
            _nms(_b, _s, 0.5)
            _tv_ok = True
            print("Torchvision NMS operator: OK")
        except Exception as _e:
            print("Torchvision NMS operator: FAILED —", str(_e)[:200])

        # --- Step 4: Fix torchaudio by installing matching torch ---
        if not _ta_ok:
            _cuda_tag = (torch.version.cuda or "").replace(".", "")
            if _cuda_tag:
                _torch_channels = [
                    f"https://download.pytorch.org/whl/test/cu{_cuda_tag}",
                    f"https://download.pytorch.org/whl/nightly/cu{_cuda_tag}",
                ]
            else:
                _torch_channels = ["https://download.pytorch.org/whl/cpu"]

            _ta_fixed = False
            for _idx in _torch_channels:
                print(f"Installing torch+torchaudio from {_idx} ...")
                _r = subprocess.run(
                    [
                        sys.executable, "-m", "pip", "install",
                        "--no-cache-dir", "--force-reinstall", "--no-deps",
                        "torch", "torchaudio",
                        "--index-url", _idx,
                    ],
                    capture_output=True, text=True,
                )
                print(_r.stdout[-1500:])
                if _r.returncode == 0:
                    _ta_fixed = True
                    break
                print("  channel failed:", _r.stderr[-500:])

            if _ta_fixed:
                # Clear cached modules so fresh imports pick up new versions
                for _mod in list(sys.modules):
                    if _mod in ("torch", "torchaudio") or _mod.startswith("torch.") or _mod.startswith("torchaudio."):
                        del sys.modules[_mod]
                importlib.invalidate_caches()
                try:
                    import torch as _torch2
                    import torchaudio as _ta2
                    from torchaudio._extension import _load_lib as _ll2
                    _ll2("libtorchaudio")
                    print("TorchAudio after reinstall:", _ta2.__version__)
                    print("Torch after reinstall:", _torch2.__version__)
                    _ta_ok = True
                except Exception as _e:
                    print("TorchAudio still FAILED after reinstall —", str(_e)[:200])

        # --- Step 5: Fix torchvision by reinstalling from matching channel ---
        if not _tv_ok:
            _cuda_tag = (torch.version.cuda or "").replace(".", "")
            if _cuda_tag:
                _tv_channels = [
                    f"https://download.pytorch.org/whl/test/cu{_cuda_tag}",
                    f"https://download.pytorch.org/whl/nightly/cu{_cuda_tag}",
                    f"https://download.pytorch.org/whl/cu{_cuda_tag}",
                ]
            else:
                _tv_channels = ["https://download.pytorch.org/whl/cpu"]

            _tv_fixed = False
            for _idx in _tv_channels:
                print(f"Reinstalling torchvision from {_idx} ...")
                _r = subprocess.run(
                    [
                        sys.executable, "-m", "pip", "install",
                        "--no-cache-dir", "--force-reinstall", "--no-deps",
                        "torchvision",
                        "--index-url", _idx,
                    ],
                    capture_output=True, text=True,
                )
                print(_r.stdout[-1000:])
                if _r.returncode == 0:
                    _tv_fixed = True
                    break
                print("  channel failed:", _r.stderr[-500:])

            if _tv_fixed:
                for _mod in list(sys.modules):
                    if _mod == "torchvision" or _mod.startswith("torchvision."):
                        del sys.modules[_mod]
                importlib.invalidate_caches()
                try:
                    import torchvision as _tv2
                    from torchvision.ops import nms as _nms2
                    _b2 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
                    _s2 = torch.tensor([0.9])
                    _nms2(_b2, _s2, 0.5)
                    print("Torchvision NMS after reinstall: OK")
                    print("New torchvision:", _tv2.__version__)
                    _tv_ok = True
                except Exception as _e:
                    print("Torchvision still FAILED after reinstall —", str(_e)[:200])

        # --- Step 6: Final check ---
        if not _ta_ok or not _tv_ok:
            raise RuntimeError(
                "Torchaudio یا torchvision با torch فعلی سازگار نیست.\n"
                "لطفاً کرنل را ری‌استارت کن (Kernel → Restart) و دوباره اجرا کن.\n"
                f"  torchaudio OK: {_ta_ok}\n"
                f"  torchvision OK: {_tv_ok}"
            )

        # --- Step 7: Verify all required imports ---
        from torchvision.io import ImageReadMode, decode_image
        from transformers import PreTrainedModel, ProcessorMixin

        print("\n=== All imports OK ===")
        print("Torch:", torch.__version__)
        print("Torchvision:", torchvision.__version__)
        print("TorchAudio:", torchaudio.__version__)
        print("CUDA:", torch.cuda.is_available())

    _()
    return


@app.cell
def _():
    def _():
        import sys
        from pathlib import Path

        import torch

        ROOT = Path("/tmp/ltx23-workspace/LTX-2-main")
        MODEL = Path("/tmp/ltx23-model/ltx-2.3-22b-distilled-1.1.safetensors")
        UPSCALER = Path("/tmp/ltx23-model/ltx-2.3-spatial-upscaler-x2-1.1.safetensors")
        GEMMA = Path("/tmp/ltx23-gemma")

        for p in [ROOT / "packages/ltx-core/src",
                  ROOT / "packages/ltx-pipelines/src"]:
            if str(p) not in sys.path:
                sys.path.insert(0, str(p))

        for p in [MODEL, UPSCALER, GEMMA / "config.json",
                  GEMMA / "model.safetensors.index.json"]:
            if not p.exists():
                raise FileNotFoundError(f"فایل پیدا نشد: {p}")

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA در دسترس نیست.")

        from ltx_pipelines.distilled import DistilledPipeline
        from ltx_pipelines.utils.model_paths import ModelPaths

        print("GPU:", torch.cuda.get_device_name(0))
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        print("در حال بارگذاری LTX-2.3؛ این مرحله ممکن است چند دقیقه طول بکشد...")

        paths = ModelPaths.from_monolith(
            checkpoint_path=str(MODEL),
            gemma_root=str(GEMMA),
        )

        pipeline = DistilledPipeline(
            model_paths=paths,
            spatial_upsampler_path=str(UPSCALER),
            loras=[],
            device=torch.device("cuda"),
            quantization=None,
            prompt_enhancer_gemma_root=str(GEMMA),
        )

        globals()["ltx_pipeline"] = pipeline
        globals()["ltx_loaded"] = True

        print("\nمدل بارگذاری شد؛ آمادهٔ ساخت ویدیو هستیم.")

    _()
    return


@app.cell
def _():
    def _():
        # ============================================================
        # LTX-2.3 STUDIO v2 — Gradio GUI with PUBLIC URL
        #  - Gemma writes the prompts (it can SEE your images / frames)
        #  - Modes: Single video | Story (multi-scene) | Edit a video
        #  - Clips of 2..60 s (long clips are chained from the last frame)
        #  - Character / environment reference images
        #  - Users + password, and a strict one-job-at-a-time queue
        # ============================================================

        # ====================== SETTINGS ============================

        # {} = بدون رمز. هر کاربر یک نام و رمز؛ حتماً رمزها را عوض کن.
        GUI_USERS = {
            "ltx": "ltx12345",
            "friend": "friend12345",
        }

        DEFAULT_STYLE = """Premium cinematic 3D animated feature film, detailed
        environments, natural materials, physically believable lighting,
        soft volumetric light, subtle film grain, cinematic color grading,
        natural depth of field, professional cinematography."""

        DEFAULT_AUDIO = """Soft warm background music that fits the mood,
        natural room sounds and sound effects matching every action,
        and spoken Persian (Farsi) dialogue if people talk."""

        FPS = 24
        SEG_SECONDS = 10      # max seconds of ONE LTX pass; longer clips are chained
        MIN_SEC, MAX_SEC = 2, 60
        MAX_QUEUE = 20

        # ============================================================
        # 1. IMPORTS
        # ============================================================

        import contextlib
        import gc
        import json
        import math
        import random
        import re
        import shutil
        import subprocess
        import sys
        import threading
        import time
        from pathlib import Path

        import torch

        try:
            import gradio as gr
        except ImportError:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "gradio"])
            import gradio as gr

        from PIL import Image, ImageOps
        import imageio_ffmpeg

        FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()

        MODEL_DIR = Path("/tmp/ltx23-model")
        CHECKPOINT = MODEL_DIR / "ltx-2.3-22b-distilled-1.1.safetensors"
        UPSCALER = MODEL_DIR / "ltx-2.3-spatial-upscaler-x2-1.1.safetensors"
        GEMMA_DIR = Path("/tmp/ltx23-gemma")
        OUT_DIR = Path("/tmp/ltx23-stories")
        OUT_DIR.mkdir(parents=True, exist_ok=True)
        GALLERY_JSON = OUT_DIR / "gallery.json"

        if not torch.cuda.is_available():
            raise RuntimeError("CUDA در دسترس نیست.")
        for p in (CHECKPOINT, UPSCALER, GEMMA_DIR):
            if not p.exists():
                raise FileNotFoundError(f"پیدا نشد: {p}")

        from ltx_pipelines.utils.media_io import encode_video

        GPU_LOCK = threading.RLock()       # one GPU job at a time
        GALLERY_LOCK = threading.Lock()
        JOB = {"cur": None}                # who is running now (for the status line)

        LONG_SIDE = {"HD": 1280, "Medium": 1024, "Low": 768}
        SQUARE_SIDE = {"HD": 1024, "Medium": 768, "Low": 512}
        PRESET_ORDER = ["HD", "Medium", "Low"]
        ASPECTS = [
            "16:9 (landscape)",
            "9:16 (portrait)",
            "1:1 (square)",
            "Auto (from start frame)",
        ]
        MODES = ["Single video", "Story (multi-scene)", "Edit a video"]
        EDIT_KINDS = [
            "Extend: continue after the end",
            "Redo from second T (keep 0..T)",
            "Restart from the first frame",
        ]

        _noop = lambda x, desc="": None


        def sub(prog, lo, hi):
            """Map a 0..1 progress callback into the lo..hi slice of its parent."""
            prog = prog or _noop
            return lambda x, desc="": prog(lo + (hi - lo) * x, desc)


        # ============================================================
        # 2. HELPERS
        # ============================================================

        def stamp_ms():
            return int(time.time() * 1000)


        def pick_size(preset, aspect, img):
            """Width/height (multiples of 64) for a preset + aspect choice."""
            def r64(x):
                return max(256, int(round(x / 64)) * 64)

            if aspect.startswith("Auto") and img is not None:
                ratio = img.width / img.height
            elif aspect.startswith("9:16"):
                ratio = 9 / 16
            elif aspect.startswith("1:1"):
                ratio = 1.0
            else:
                ratio = 16 / 9

            if abs(ratio - 1.0) < 0.02:
                s = SQUARE_SIDE[preset]
                return s, s
            L = LONG_SIDE[preset]
            if ratio >= 1:
                return L, r64(L / ratio)
            return r64(L * ratio), L


        def frames_for(duration):
            return 1 + 8 * max(1, round((duration * FPS - 1) / 8))


        def split_duration(total):
            """2..60 s -> list of segment lengths, each <= SEG_SECONDS."""
            total = int(max(MIN_SEC, min(MAX_SEC, round(float(total)))))
            n = max(1, math.ceil(total / SEG_SECONDS))
            base, extra = divmod(total, n)
            return [base + (1 if i < extra else 0) for i in range(n)]


        def clean_prompt(text):
            return re.sub(r"^```[a-zA-Z]*\s*|\s*```$", "", text.strip()).strip()


        def try_json(text):
            for pat in (r"\[.*\]", r"\{.*\}"):
                m = re.search(pat, text, re.S)
                if m:
                    try:
                        return json.loads(m.group(0))
                    except Exception:
                        pass
            return None


        def run_ffmpeg(args):
            r = subprocess.run([FFMPEG, "-y", "-loglevel", "error", *args],
                               capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(r.stderr[-1500:])


        def probe(path):
            r = subprocess.run([FFMPEG, "-i", str(path)], capture_output=True, text=True)
            err = r.stderr
            m = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", err)
            dur = round(int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3]), 2) if m else None
            v = re.search(r"Video:.*?[,\s](\d{2,5})x(\d{2,5})[,\s\[]", err)
            return {"duration": dur, "audio": "Audio:" in err,
                    "w": int(v[1]) if v else None, "h": int(v[2]) if v else None}


        def video_duration(path):
            d = probe(path)["duration"]
            return round(d, 1) if d else None


        def extract_frame(video, out, at=None):
            """Frame at `at` seconds, or the LAST frame when at is None."""
            if at is None:
                args = ["-sseof", "-0.2", "-i", str(video)]
            else:
                args = ["-ss", f"{float(at):.3f}", "-i", str(video)]
            run_ffmpeg([*args, "-frames:v", "1", "-update", "1", str(out)])
            if not Path(out).exists():
                raise RuntimeError("frame ساخته نشد.")


        def make_thumb_and_last(video_path, key):
            thumb = OUT_DIR / f"thumb_{key}.jpg"
            last = OUT_DIR / f"last_{key}.png"
            run_ffmpeg(["-i", str(video_path), "-frames:v", "1", "-update", "1",
                        "-vf", "scale=384:-2", str(thumb)])
            extract_frame(video_path, last)
            if not thumb.exists():
                raise RuntimeError("thumbnail ساخته نشد.")
            return str(thumb), str(last)


        def normalize_clip(src, dst, w, h, max_dur=None):
            """Re-encode to w x h / 24 fps / AAC (adds silence if no audio)."""
            info = probe(src)
            cmd = ["-i", str(src)]
            if not info["audio"]:
                cmd += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
            vf = (f"scale={w}:{h}:force_original_aspect_ratio=increase,crop={w}:{h},"
                  f"fps={FPS},setsar=1,format=yuv420p")
            cmd += ["-vf", vf]
            if max_dur:
                cmd += ["-t", f"{float(max_dur):.3f}"]
            cmd += ["-map", "0:v:0"]
            cmd += ["-map", "0:a:0"] if info["audio"] else ["-map", "1:a:0", "-shortest"]
            cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16",
                    "-c:a", "aac", "-ar", "48000", "-ac", "2", "-b:a", "192k", str(dst)]
            run_ffmpeg(cmd)


        def join_videos(items, out_path, size):
            """items = [(path, max_duration_or_None), ...] -> one mp4 of `size`."""
            w, h = size
            tmpd = OUT_DIR / f"tmp_{stamp_ms()}"
            tmpd.mkdir(exist_ok=True)
            try:
                parts = []
                for i, (src, mx) in enumerate(items):
                    dst = tmpd / f"p{i}.mp4"
                    normalize_clip(src, dst, w, h, mx)
                    parts.append(dst)
                lst = tmpd / "list.txt"
                lst.write_text("".join(f"file '{p.as_posix()}'\n" for p in parts))
                run_ffmpeg(["-f", "concat", "-safe", "0", "-i", str(lst),
                            "-c", "copy", str(out_path)])
            finally:
                shutil.rmtree(tmpd, ignore_errors=True)


        def load_refs(char_files, env_files):
            """-> [(label, PIL)] with labels char1.., env1.."""
            refs = []
            for prefix, files in (("char", char_files), ("env", env_files)):
                for i, f in enumerate(files or [], 1):
                    path = getattr(f, "name", f)
                    try:
                        refs.append((f"{prefix}{i}", Image.open(path).convert("RGB")))
                    except Exception as e:
                        print("bad reference image:", path, str(e)[:100])
            return refs[:8]


        # ---------------- gallery storage ----------------

        def _load_all():
            try:
                data = json.loads(GALLERY_JSON.read_text(encoding="utf-8"))
                return data if isinstance(data, list) else []
            except Exception:
                return []


        def _save_all(items):
            tmp = Path(str(GALLERY_JSON) + ".tmp")
            tmp.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")
            tmp.replace(GALLERY_JSON)


        def add_entry(entry):
            with GALLERY_LOCK:
                items = [e for e in _load_all() if e.get("video") != entry["video"]]
                items.append(entry)
                _save_all(items)


        def sync_existing_videos():
            """Add videos already in OUT_DIR (from earlier runs) to the gallery."""
            with GALLERY_LOCK:
                items = _load_all()
                known = {e.get("video") for e in items}
                added = 0
                files = sorted(list(OUT_DIR.glob("video_*.mp4")) + list(OUT_DIR.glob("movie_*.mp4")))
                for vp in files:
                    if str(vp) in known or vp.stat().st_size == 0:
                        continue
                    m = re.match(r"(?:video|movie)_(\d+)_(\d+)_(\d+)x(\d+)\.mp4$", vp.name)
                    if not m:
                        continue
                    stamp, seed, w, h = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
                    try:
                        thumb, last = make_thumb_and_last(vp, f"{stamp}_{seed}")
                    except Exception as e:
                        print("skip", vp.name, str(e)[:120])
                        continue
                    pf = OUT_DIR / f"prompt_{stamp}_{seed}.txt"
                    prompt = pf.read_text(encoding="utf-8") if pf.exists() else ""
                    items.append({
                        "video": str(vp), "thumb": thumb, "last_frame": last,
                        "prompt": prompt, "seed": int(seed), "w": w, "h": h,
                        "duration": video_duration(vp), "created": int(stamp),
                        "continued_from": None,
                    })
                    added += 1
                if added:
                    _save_all(items)
                return added


        def gallery_entries():
            items = [e for e in _load_all()
                     if Path(e.get("video", "")).exists() and Path(e.get("thumb", "")).exists()]
            items.sort(key=lambda e: e.get("created", 0), reverse=True)
            return items


        def caption(e):
            c = f"{e['w']}x{e['h']} · {e.get('duration') or '?'}s"
            if e.get("owner"):
                c += f" · {e['owner']}"
            if e.get("kind") == "movie":
                return "🎞 " + c
            return ("↪ " + c) if e.get("continued_from") else c


        def gallery_items(entries):
            return [(e["thumb"], caption(e)) for e in entries]


        def find_entry(video_path):
            for e in gallery_entries():
                if e["video"] == video_path:
                    return e
            return None


        def sel_md(e):
            if not e:
                return "_ویدیویی انتخاب نشده._"
            return (f"**Selected:** `{Path(e['video']).name}` — {e['w']}x{e['h']}, "
                    f"{e.get('duration') or '?'}s, seed {e.get('seed')}")


        def register_clip(clip, continued_from=None, owner=None, prefix="video"):
            """Move a finished temp video to its final name and add it to the gallery."""
            stamp = int(time.time())
            seed = clip["seed"]
            w, h = clip["w"], clip["h"]
            while (OUT_DIR / f"{prefix}_{stamp}_{seed}_{w}x{h}.mp4").exists():
                stamp += 1
            final = OUT_DIR / f"{prefix}_{stamp}_{seed}_{w}x{h}.mp4"
            Path(clip["video"]).replace(final)
            key = f"{stamp}_{seed}"
            (OUT_DIR / f"prompt_{key}.txt").write_text(clip["prompt"], encoding="utf-8")
            thumb, last = make_thumb_and_last(final, key)
            entry = {
                "video": str(final), "thumb": thumb, "last_frame": last,
                "prompt": clip["prompt"], "last_prompt": clip.get("last_prompt"),
                "seed": seed, "w": w, "h": h,
                "duration": video_duration(final) or clip.get("duration"),
                "created": stamp, "continued_from": continued_from,
                "elapsed": clip.get("elapsed"), "owner": owner,
                "kind": "movie" if prefix == "movie" else "clip",
            }
            add_entry(entry)
            return entry


        # ---------------- LTX pipeline ----------------

        def get_pipeline():
            pipeline = globals().get("ltx_pipeline")
            if pipeline is not None:
                return pipeline

            print("Loading LTX pipeline...")
            from ltx_pipelines.distilled import DistilledPipeline
            try:
                from ltx_pipelines.utils.model_paths import ModelPaths
                model_paths = ModelPaths.from_monolith(
                    checkpoint_path=str(CHECKPOINT),
                    gemma_root=str(GEMMA_DIR),
                )
                pipeline = DistilledPipeline(
                    model_paths=model_paths,
                    spatial_upsampler_path=str(UPSCALER),
                    loras=[],
                    device=torch.device("cuda:0"),
                )
            except (ImportError, TypeError):
                pipeline = DistilledPipeline(
                    distilled_checkpoint_path=str(CHECKPOINT),
                    spatial_upsampler_path=str(UPSCALER),
                    gemma_root=str(GEMMA_DIR),
                    loras=(),
                    device=torch.device("cuda:0"),
                )
            globals()["ltx_pipeline"] = pipeline
            return pipeline


        def make_condition(path, strength):
            try:
                from ltx_pipelines.utils.types import ImageConditioningInput
                return ImageConditioningInput.at(
                    path=str(path), frame_idx=0, strength=float(strength)
                )
            except Exception:
                return (str(path), 0, float(strength))


        # ============================================================
        # 3. GEMMA (sees images) — prompt writer + story planner
        # ============================================================

        def gemma_run(instruction, images=None, max_new_tokens=1400):
            """Load Gemma, answer once (optionally with images), free the GPU."""
            from transformers import AutoProcessor, Gemma3ForConditionalGeneration

            side = 896 if len(images or []) <= 1 else 640
            imgs = []
            for im in images or []:
                g = im.convert("RGB").copy()
                g.thumbnail((side, side))
                imgs.append(g)

            with GPU_LOCK:
                processor = AutoProcessor.from_pretrained(str(GEMMA_DIR), local_files_only=True)
                gemma = Gemma3ForConditionalGeneration.from_pretrained(
                    str(GEMMA_DIR),
                    torch_dtype=torch.bfloat16,
                    device_map={"": "cuda:0"},
                    low_cpu_mem_usage=True,
                    local_files_only=True,
                )
                gemma.eval()
                dev = torch.device("cuda:0")

                def run(content):
                    messages = [{"role": "user", "content": content}]
                    inputs = processor.apply_chat_template(
                        messages, tokenize=True, add_generation_prompt=True,
                        return_dict=True, return_tensors="pt",
                    )
                    moved = {}
                    for k, v in inputs.items():
                        if isinstance(v, torch.Tensor):
                            moved[k] = v.to(dev, dtype=torch.bfloat16) if k == "pixel_values" else v.to(dev)
                        else:
                            moved[k] = v
                    with torch.inference_mode():
                        gen = gemma.generate(**moved, max_new_tokens=max_new_tokens, do_sample=False)
                    text = processor.decode(gen[0, moved["input_ids"].shape[-1]:],
                                            skip_special_tokens=True)
                    del moved, gen
                    return clean_prompt(text)

                try:
                    if imgs:
                        try:
                            return run([{"type": "image", "image": im} for im in imgs]
                                       + [{"type": "text", "text": instruction}])
                        except Exception as e:
                            print("Gemma could not read the images; text-only:",
                                  type(e).__name__, str(e)[:200])
                    return run([{"type": "text", "text": instruction}])
                finally:
                    del gemma, processor
                    gc.collect()
                    torch.cuda.empty_cache()


        def refs_text(start_img, refs):
            """Describes the order of attached images for Gemma."""
            lines, n = [], 0
            if start_img is not None:
                n += 1
                lines.append(f"image {n} = START FRAME (exact first frame of the video)")
            for label, _ in refs:
                n += 1
                kind = "CHARACTER" if label.startswith("char") else "ENVIRONMENT"
                lines.append(f"image {n} = {kind} reference '{label}'")
            return "\n".join(lines)


        def write_prompt(idea, duration, style, audio_wishes, start_img,
                         prev_prompt=None, refs=None):
            refs = refs or []
            actual = frames_for(float(duration)) / FPS
            max_words = max(3, int(actual * 2.2))

            extra = ""
            if start_img is not None or refs:
                extra += "\nATTACHED IMAGES (in this order):\n" + refs_text(start_img, refs) + "\n"
            if start_img is not None:
                extra += """
        START FRAME: it is the EXACT first frame of the video. Describe the same
        characters (face, hair, clothing), location, objects, lighting, colors and
        framing that you SEE in it. The video continues naturally from this image;
        do not change identities, outfit, or place.
        """
            if refs:
                extra += """
        REFERENCE IMAGES: whenever such a character or place appears in the shot,
        describe it exactly as it looks in its reference image (face, hair, age,
        clothing, colors, architecture). Keep it consistent.
        """
            if prev_prompt:
                extra += f"""
        CONTINUATION: this shot directly continues a PREVIOUS shot. The start frame
        is the previous shot's LAST frame. Previous shot prompt, for reference only:
        <<<
        {prev_prompt[:2500]}
        >>>
        Keep the same location, characters, clothing, lighting, voices and music
        style. Describe ONLY what happens NEXT; do not repeat earlier actions or
        dialogue. Time markers restart at 0s.
        """

            instruction = f"""
        You are an expert prompt writer for LTX-2.3, a text-to-video model that
        generates picture AND synchronized audio (music, sound effects, spoken
        dialogue) in one pass. Turn the user's idea into ONE complete English
        prompt for a single continuous shot of about {actual:.0f} seconds.
        {extra}
        USER IDEA (Persian or English):
        {idea.strip()}

        VISUAL STYLE:
        {style.strip()}

        AUDIO WISHES:
        {audio_wishes.strip()}

        Write the prompt in this order, as flowing present-tense prose:
        1. Shot type and visual style.
        2. The setting (one single location) and the character(s): age, face,
           hair, clothing, expression. Be concrete and consistent.
        3. The action in strict chronological order with time markers such as
           "0-3s: ...", "3-6s: ...". Only physically plausible actions; nobody
           passes through furniture; objects are held before they are used.
        4. Camera: lens feel, angle, slow and smooth movement.
        5. Lighting and color palette.
        6. SOUND DESIGN: ambient room sound, sound effects matching each action,
           and background music (mood, instruments, tempo, volume under speech).
        7. DIALOGUE: if anyone speaks, write each line exactly like this:
           The [character] says in Persian, in a [voice description] voice,
           at about [time]s: "<the line written in Persian script>".
           Write natural, short, polite everyday Persian. Use at most
           {max_words} spoken words in total. The speaker's lips move in
           sync. If the idea has no speaking, do not add dialogue.

        Rules:
        - One location, one continuous shot, no cuts, no scene changes.
        - No subtitles, on-screen text, logos or watermarks.
        - Keep Persian only inside the dialogue quotes; everything else English.
        - Return ONLY the final prompt text, no title, no explanation, no markdown.
        """
            imgs = ([start_img] if start_img is not None else []) + [im for _, im in refs]
            result = gemma_run(instruction, imgs, 1400)
            if len(result) < 200:
                raise RuntimeError("Gemma پرامپت قابل استفاده‌ای نساخت؛ دوباره امتحان کن.")
            return result


        def normalize_plan(data, default_len, use_given_len=True):
            if isinstance(data, dict):
                data = data.get("scenes") or data.get("plan") or []
            out = []
            for it in data if isinstance(data, list) else []:
                if isinstance(it, str):
                    it = {"idea": it}
                if not isinstance(it, dict):
                    continue
                idea = str(it.get("idea") or it.get("description") or "").strip()
                if not idea:
                    continue
                d = default_len
                if use_given_len:
                    try:
                        d = int(round(float(it.get("duration", default_len))))
                    except Exception:
                        d = default_len
                d = int(max(MIN_SEC, min(MAX_SEC, d)))
                cont = it.get("continue")
                if isinstance(cont, str):
                    cont = cont.strip().lower() == "true"
                ref = it.get("start_ref")
                ref = str(ref).strip().lower() if ref else None
                if ref in ("", "null", "none"):
                    ref = None
                out.append({"idea": idea, "duration": d,
                            "continue": bool(cont) and len(out) > 0, "start_ref": ref})
            return out


        def plan_scenes(idea, n, scene_len, auto_len, style, audio_wishes, start_img, refs):
            """Gemma splits a story into scenes and decides which ones continue."""
            schema = '[{"idea": "...", "duration": 8, "continue": false, "start_ref": null}]'
            dur_rule = (f'"duration": integer seconds between {MIN_SEC} and {MAX_SEC} that '
                        f'fits the action (typical 4-15).' if auto_len else
                        f'"duration": always {scene_len}.')
            img_block = ""
            if start_img is not None or refs:
                img_block = "\nATTACHED IMAGES (in this order):\n" + refs_text(start_img, refs) + "\n"
            instruction = f"""
        You are the director of a short film made of consecutive AI-generated video
        shots (a model that makes picture + sound + dialogue per shot).
        Split the user's story into exactly {n} consecutive scenes.
        {img_block}
        STORY (Persian or English):
        {idea.strip()}

        VISUAL STYLE: {style.strip()}
        AUDIO: {audio_wishes.strip()}

        For every scene output an object with:
        - "idea": 2-5 English sentences: location, who is there, exactly what
          happens, mood. Describe characters consistently with the reference
          images (face, hair, clothing). Put spoken lines in Persian script
          between « » (NEVER use double-quote characters inside the strings).
        - "continue": true ONLY if this shot happens in the SAME place and moment as
          the previous shot, so it must start from the previous shot's last frame
          (continuous action). false if the place, time or setup changes: it is then
          a new fresh shot. Scene 1 is always false.
        - "start_ref": when continue is false and one of the attached reference
          images should be the first frame of the shot, its label (like "char1" or
          "env1"); otherwise null.
        - {dur_rule}

        Return ONLY a JSON array, no markdown, no commentary, in this shape:
        {schema}
        """
            imgs = ([start_img] if start_img is not None else []) + [im for _, im in refs]
            text = gemma_run(instruction, imgs, 2500)
            plan = normalize_plan(try_json(text), scene_len, use_given_len=auto_len)
            if not plan:
                print("Gemma plan unreadable -> single-scene fallback")
                plan = [{"idea": idea.strip(), "duration": scene_len,
                         "continue": False, "start_ref": None}]
            return plan[:n]


        def plan_beats(idea, durs, style, audio_wishes, start_img, refs):
            """Split ONE long continuous clip into beats (one per LTX pass)."""
            n = len(durs)
            img_block = ""
            if start_img is not None or refs:
                img_block = "\nATTACHED IMAGES (in this order):\n" + refs_text(start_img, refs) + "\n"
            instruction = f"""
        You are a screenwriter. One continuous scene (same location, same
        characters, no cuts) will be filmed in {n} consecutive parts. Part k starts
        exactly where part k-1 ended. Part lengths in seconds: {durs}.
        {img_block}
        IDEA (Persian or English):
        {idea.strip()}

        Split the idea into exactly {n} beats. Each beat is 2-4 English sentences
        describing only what happens in that part, in order, without repeating
        earlier actions. Put spoken lines in Persian script between « » (NEVER use
        double-quote characters inside the strings).
        Return ONLY a JSON array of {n} strings.
        """
            imgs = ([start_img] if start_img is not None else []) + [im for _, im in refs]
            data = try_json(gemma_run(instruction, imgs, 1500))
            beats = [str(x).strip() for x in data if str(x).strip()] if isinstance(data, list) else []
            if len(beats) < n:
                print("Gemma beats unreadable -> using the idea for each part")
                beats = [f"{idea.strip()}\n(This is part {i + 1} of {n} of one continuous scene; "
                         f"continue the action naturally.)" for i in range(n)]
            return beats[:n]


        # ============================================================
        # 4. VIDEO GENERATION
        # ============================================================

        def generate_segment(prompt, duration, preset, aspect, seed, strength,
                             start_img, force_size=None, prog=None):
            """ONE LTX pass (<= SEG_SECONDS). Returns a dict, does not touch gallery."""
            prog = prog or _noop
            duration = float(duration)
            num_frames = frames_for(duration)
            tag = stamp_ms()

            cands = []
            if force_size:
                cands.append(("same-size", force_size[0], force_size[1]))
            for name in PRESET_ORDER[PRESET_ORDER.index(preset):]:
                w, h = pick_size(name, "Auto (from start frame)" if force_size else aspect, start_img)
                if force_size and w * h >= force_size[0] * force_size[1]:
                    continue
                if (w, h) not in [(c[1], c[2]) for c in cands]:
                    cands.append((name, w, h))

            temp_imgs = []
            with GPU_LOCK:
                prog(0.05, "Loading LTX...")
                pipeline = get_pipeline()
                chunks = audio = out = None
                used = None

                for name, w, h in cands:
                    images = []
                    if start_img is not None:
                        fitted = ImageOps.fit(start_img.convert("RGB"), (w, h),
                                              method=Image.LANCZOS, centering=(0.5, 0.5))
                        img_path = OUT_DIR / f"start_{tag}_{w}x{h}.png"
                        fitted.save(img_path)
                        temp_imgs.append(img_path)
                        images = [make_condition(img_path, strength)]

                    prog(0.2, f"Generating {w}x{h} ({duration:.0f}s)...")
                    try:
                        with torch.inference_mode():
                            out = pipeline(
                                prompt=prompt, seed=seed, height=h, width=w,
                                frame_rate=float(FPS), images=images,
                                num_frames=num_frames, enhance_prompt=False,
                            )
                            if isinstance(out, tuple):
                                video, audio = out[0], out[1]
                            else:
                                video, audio = out.video, out.audio
                            chunks = [video] if isinstance(video, torch.Tensor) else list(video)
                        used = (name, w, h)
                        break
                    except torch.cuda.OutOfMemoryError:
                        print(f"Out of memory at {name} ({w}x{h}); lowering resolution.")
                        chunks = audio = out = None
                        gc.collect()
                        torch.cuda.empty_cache()

                if used is None:
                    raise gr.Error("حافظه کافی نبود؛ مدت را کمتر کن.")

                prog(0.88, "Encoding video + audio...")
                output_path = OUT_DIR / f"seg_{tag}_{seed}_{used[1]}x{used[2]}.mp4"
                enc = dict(video=iter(chunks), fps=float(FPS), audio=audio,
                           output_path=str(output_path), video_chunks_number=len(chunks))
                try:
                    encode_video(**enc, audio_sampling_rate=None)
                except TypeError:
                    enc["video"] = iter(chunks)
                    encode_video(**enc)

                del chunks, audio, out
                gc.collect()
                torch.cuda.empty_cache()

            for p in temp_imgs:
                p.unlink(missing_ok=True)
            if not output_path.exists() or output_path.stat().st_size == 0:
                raise gr.Error("فایل خروجی ساخته نشد.")
            return {"video": str(output_path), "w": used[1], "h": used[2],
                    "duration": round(num_frames / FPS, 1)}


        def render_clip(idea, duration, start_img, prev_prompt, seed, force_size,
                        preset, aspect, strength, style, audio, refs=None,
                        final_prompt=None, prog=None):
            """A clip of 2..60 s = 1 or more chained LTX passes (same seed,
            each pass starts from the previous pass's last frame)."""
            prog = prog or _noop
            t0 = time.time()
            seed = int(seed) if seed is not None and int(seed) >= 0 else random.randint(1, 2**31 - 1)
            durs = split_duration(duration)
            n = len(durs)
            idea = idea or final_prompt or ""

            if n > 1:
                prog(0.0, f"Gemma is splitting the clip into {n} parts...")
                beats = plan_beats(idea, durs, style, audio, start_img, refs or [])
            else:
                beats = [idea]

            segs, prompts = [], []
            cur_img, cur_prev, size = start_img, prev_prompt, force_size
            tag = stamp_ms()

            for i, (d, beat) in enumerate(zip(durs, beats)):
                sp = sub(prog, i / n, (i + 1) / n)
                if i == 0 and final_prompt:
                    p = final_prompt
                else:
                    sp(0.02, f"Part {i + 1}/{n}: Gemma is writing the prompt...")
                    p = write_prompt(beat, d, style, audio, cur_img, cur_prev, refs)
                seg = generate_segment(p, d, preset, aspect, seed,
                                       strength if i == 0 else 1.0,
                                       cur_img, size, prog=sub(sp, 0.1, 1.0))
                segs.append(seg)
                prompts.append(p)
                if i < n - 1:
                    lf = OUT_DIR / f"lf_{tag}_{i}.png"
                    extract_frame(seg["video"], lf)
                    cur_img = Image.open(lf).convert("RGB")
                    lf.unlink(missing_ok=True)
                    cur_prev, size = p, (seg["w"], seg["h"])

            if n == 1:
                out = segs[0]["video"]
            else:
                prog(0.98, "Joining parts...")
                out = str(OUT_DIR / f"seg_join_{tag}.mp4")
                join_videos([(s["video"], None) for s in segs], out, (segs[0]["w"], segs[0]["h"]))
                for s in segs:
                    Path(s["video"]).unlink(missing_ok=True)

            joined = "\n\n".join(f"--- part {i + 1} ---\n{p}" for i, p in enumerate(prompts)) \
                if n > 1 else prompts[0]
            return {"video": out, "w": segs[0]["w"], "h": segs[0]["h"], "seed": seed,
                    "duration": sum(durs), "prompt": joined, "last_prompt": prompts[-1],
                    "elapsed": round(time.time() - t0)}


        # ============================================================
        # 5. MODES: story + edit
        # ============================================================

        def run_story(owner, plan, seed_in, preset, aspect, strength, style, audio,
                      start_img, refs_all, prog):
            """Generator yielding (entry, info) after every scene (+ final movie)."""
            refs_dict = dict(refs_all)
            seed0 = int(seed_in) if seed_in is not None and int(seed_in) >= 0 \
                else random.randint(1, 2**31 - 2)
            total = len(plan)
            entries, prev = [], None

            for i, sc in enumerate(plan):
                sp = sub(prog, i / total, (i + 1) / total * 0.97)
                cont = bool(sc.get("continue")) and prev is not None
                start_label = None
                if cont:
                    # same shot continues: last frame of previous scene, SAME seed
                    img = Image.open(prev["last_frame"]).convert("RGB")
                    force = (prev["w"], prev["h"])
                    prev_prompt = prev.get("last_prompt") or prev.get("prompt")
                    seed, st, cfrom = prev["seed"], 1.0, prev["video"]
                else:
                    # new scene: new seed, optional reference image as first frame
                    label = sc.get("start_ref")
                    img = None
                    if label in refs_dict:
                        img, start_label = refs_dict[label], label
                    elif i == 0 and start_img is not None:
                        img = start_img
                    force, prev_prompt, cfrom, st = None, None, None, strength
                    seed = seed0 if i == 0 else (seed0 + 7919 * i) % (2**31 - 2) + 1
                refs = [(l, im) for l, im in refs_all if l != start_label]

                clip = render_clip(sc["idea"], sc["duration"], img, prev_prompt, seed, force,
                                   preset, aspect, st, style, audio, refs=refs, prog=sp)
                entry = register_clip(clip, continued_from=cfrom, owner=owner)
                entries.append(entry)
                prev = entry
                yield entry, (f"Scene {i + 1}/{total} done in {entry['elapsed']}s | "
                              f"{'continued (same seed)' if cont else 'new scene (new seed)'} | "
                              f"{entry['w']}x{entry['h']} | {entry['duration']}s")

            if len(entries) > 1:
                prog(0.98, "Joining all scenes into one movie...")
                tag = stamp_ms()
                out = str(OUT_DIR / f"seg_movie_{tag}.mp4")
                first = entries[0]
                join_videos([(e["video"], None) for e in entries], out, (first["w"], first["h"]))
                movie = register_clip({
                    "video": out, "w": first["w"], "h": first["h"], "seed": first["seed"],
                    "prompt": "\n\n".join(f"=== scene {i + 1} ===\n{e['prompt']}"
                                          for i, e in enumerate(entries)),
                    "last_prompt": entries[-1].get("last_prompt"),
                    "duration": sum(e.get("duration") or 0 for e in entries),
                }, owner=owner, prefix="movie")
                yield movie, f"Final movie: {len(entries)} scenes | {movie['duration']}s"


        def run_edit(owner, src, idea, kind, cut, duration, join, seed, preset,
                     strength, style, audio, refs, prog):
            if not src:
                raise gr.Error("اول یک ویدیو آپلود کن.")
            src = str(getattr(src, "name", src))
            info = probe(src)
            sdur = info["duration"] or 0
            if sdur < 1:
                raise gr.Error("ویدیوی ورودی خوانده نشد.")
            tag = stamp_ms()
            extend, restart = kind.startswith("Extend"), kind.startswith("Restart")
            t = 0.0 if restart else (sdur if extend else min(max(float(cut or 0), 0.5), sdur - 0.2))

            frame = OUT_DIR / f"edit_frame_{tag}.png"
            extract_frame(src, frame, None if extend else t)
            img = Image.open(frame).convert("RGB")
            frame.unlink(missing_ok=True)

            clip = render_clip(idea, duration, img, None, seed, None, preset,
                               "Auto (from start frame)", strength, style, audio,
                               refs=refs, prog=sub(prog, 0.0, 0.93))
            if join and not restart:
                prog(0.95, "Joining with the original video...")
                out = str(OUT_DIR / f"seg_edit_{tag}.mp4")
                join_videos([(src, None if extend else t), (clip["video"], None)],
                            out, (clip["w"], clip["h"]))
                Path(clip["video"]).unlink(missing_ok=True)
                clip["video"] = out
            return register_clip(clip, owner=owner)


        # ============================================================
        # 6. GUI HANDLERS
        # ============================================================

        def who(request):
            return getattr(request, "username", None) or "guest"


        @contextlib.contextmanager
        def job(user, title):
            JOB["cur"] = {"user": user, "title": title, "t0": time.time()}
            try:
                yield
            finally:
                JOB["cur"] = None
                gc.collect()
                torch.cuda.empty_cache()


        def ui_status():
            j = JOB["cur"]
            if not j:
                return "🟢 GPU آزاد است."
            mins = int((time.time() - j["t0"]) / 60)
            return (f"🔴 در حال اجرا: **{j['user']}** — {j['title']} ({mins} min). "
                    f"درخواست‌های بعدی در صف می‌مانند.")


        def resolve_context(cont, sel_path, start_img):
            """Returns (start_img, force_size, prev_prompt, prev_entry)."""
            if not cont:
                return start_img, None, None, None
            e = find_entry(sel_path) if sel_path else None
            if e is None:
                raise gr.Error("اول یک ویدیو را از گالری انتخاب کن.")
            img = Image.open(e["last_frame"]).convert("RGB")
            return img, (e["w"], e["h"]), e.get("last_prompt") or e.get("prompt") or None, e


        def pack(entry, info, plan_text=None):
            entries = gallery_entries()
            return (
                entry["video"], entry["prompt"], gr.update(value=""), info,
                gallery_items(entries), entries, entry["video"],
                entry["last_frame"], sel_md(entry),
                gr.update(value=plan_text) if plan_text else gr.update(),
            )


        def ui_mode(m):
            return (gr.update(visible=m == MODES[0]), gr.update(visible=m == MODES[1]),
                    gr.update(visible=m == MODES[2]), gr.update(visible=m == MODES[1]),
                    gr.update(visible=m == MODES[0]))


        def ui_write(idea, duration, style, audio_wishes, start_img, cont, sel_path,
                     char_files, env_files, request: gr.Request = None):
            if not idea or not idea.strip():
                raise gr.Error("ایده را بنویس.")
            img, _size, prev_prompt, _e = resolve_context(cont, sel_path, start_img)
            refs = load_refs(char_files, env_files)
            d = split_duration(duration)[0]
            with job(who(request), "writing a prompt"):
                return write_prompt(idea, d, style, audio_wishes, img, prev_prompt, refs)


        def ui_plan(idea, n_scenes, scene_len, auto_len, style, audio_wishes, start_img,
                    char_files, env_files, request: gr.Request = None):
            if not idea or not idea.strip():
                raise gr.Error("داستان را بنویس.")
            refs = load_refs(char_files, env_files)
            with job(who(request), "planning a story"):
                plan = plan_scenes(idea, int(n_scenes), int(scene_len), bool(auto_len),
                                   style, audio_wishes, start_img, refs)
            return json.dumps(plan, ensure_ascii=False, indent=2)


        def ui_run(mode, idea, prompt, plan_text, duration, preset, aspect, seed, strength,
                   style, audio_wishes, start_img, char_files, env_files, cont, sel_path,
                   n_scenes, scene_len, auto_len, edit_video, edit_kind, edit_cut, edit_join,
                   progress=gr.Progress(), request: gr.Request = None):
            owner = who(request)
            prog = lambda x, desc="": progress(min(max(x, 0.0), 1.0), desc=desc)
            seed_v = int(seed) if seed is not None and int(seed) >= 0 else -1
            refs = load_refs(char_files, env_files)
            idea = (idea or "").strip()

            with job(owner, mode):
                try:
                    # ---------------- single video ----------------
                    if mode == MODES[0]:
                        if not idea and not (prompt or "").strip():
                            raise gr.Error("ایده یا پرامپت را بنویس.")
                        img, force, prev_prompt, prev_e = resolve_context(cont, sel_path, start_img)
                        if prev_e is not None and seed_v < 0:
                            seed_v = prev_e["seed"]          # continuing -> same seed
                        clip = render_clip(idea, duration, img, prev_prompt, seed_v, force,
                                           preset, aspect, strength, style, audio_wishes,
                                           refs=refs, final_prompt=(prompt or "").strip() or None,
                                           prog=prog)
                        entry = register_clip(clip, continued_from=prev_e["video"] if prev_e else None,
                                              owner=owner)
                        yield pack(entry, f"Done in {entry['elapsed']}s | {entry['w']}x{entry['h']} | "
                                          f"{entry['duration']}s | seed {entry['seed']}"
                                          + (" | continued from previous video" if prev_e else ""))

                    # ---------------- story ----------------
                    elif mode == MODES[1]:
                        if not idea and not (plan_text or "").strip():
                            raise gr.Error("داستان را بنویس.")
                        if (plan_text or "").strip():
                            plan = normalize_plan(try_json(plan_text), int(scene_len), True)
                            if not plan:
                                raise gr.Error("JSON پلن معتبر نیست؛ دوباره «Plan» را بزن.")
                        else:
                            prog(0.0, "Gemma is planning the scenes...")
                            plan = plan_scenes(idea, int(n_scenes), int(scene_len), bool(auto_len),
                                               style, audio_wishes, start_img, refs)
                        plan_json = json.dumps(plan, ensure_ascii=False, indent=2)
                        for entry, info in run_story(owner, plan, seed_v, preset, aspect, strength,
                                                     style, audio_wishes, start_img, refs, prog):
                            yield pack(entry, info, plan_json)

                    # ---------------- edit ----------------
                    else:
                        if not idea:
                            raise gr.Error("توضیح بده با ویدیو چه کار کنم.")
                        entry = run_edit(owner, edit_video, idea, edit_kind, edit_cut, duration,
                                         bool(edit_join), seed_v, preset, strength, style,
                                         audio_wishes, refs, prog)
                        yield pack(entry, f"Edit done in {entry['elapsed']}s | {entry['w']}x{entry['h']} | "
                                          f"{entry['duration']}s")
                except torch.cuda.OutOfMemoryError:
                    raise gr.Error("حافظه GPU کم آمد؛ کیفیت یا مدت را کمتر کن.")


        def ui_refresh():
            entries = gallery_entries()
            return gallery_items(entries), entries


        def ui_select(entries, evt: gr.SelectData):
            idx = evt.index if isinstance(evt.index, int) else evt.index[0]
            if idx is None or idx >= len(entries):
                raise gr.Error("انتخاب نامعتبر؛ گالری را رفرش کن.")
            e = entries[idx]
            info = f"{e['w']}x{e['h']} | {e.get('duration') or '?'}s | seed {e.get('seed')}"
            return (e["video"], e.get("prompt", ""), info,
                    e["video"], e["last_frame"], sel_md(e))


        def ui_delete(sel_path):
            if not sel_path:
                raise gr.Error("اول یک ویدیو را انتخاب کن.")
            with GALLERY_LOCK:
                keep = []
                for e in _load_all():
                    if e.get("video") == sel_path:
                        for k in ("video", "thumb", "last_frame"):
                            if e.get(k):
                                Path(e[k]).unlink(missing_ok=True)
                    else:
                        keep.append(e)
                _save_all(keep)
            entries = gallery_entries()
            return gallery_items(entries), entries, None, None, "_ویدیویی انتخاب نشده._"


        # ============================================================
        # 7. GUI LAYOUT
        # ============================================================

        with gr.Blocks(title="LTX-2.3 Studio") as demo:
            gr.Markdown("## LTX-2.3 Studio v2\nGemma پرامپت/داستان را می‌نویسد، LTX ویدیو + صدا + دیالوگ می‌سازد.")
            status_md = gr.Markdown("🟢 GPU آزاد است.")

            entries_state = gr.State([])
            sel_state = gr.State(None)

            with gr.Row():
                with gr.Column(scale=1):
                    mode = gr.Radio(MODES, value=MODES[0], label="Mode")
                    idea = gr.Textbox(
                        label="ایده / داستان / دستور ادیت (فارسی یا انگلیسی)", lines=5, rtl=True,
                        placeholder="مثلاً: صحنه اول مرد در ساندویچی فلافل سفارش می‌دهد، بعد سر میز می‌نشیند و می‌خورد...",
                    )
                    with gr.Accordion("Reference images (کاراکتر / محیط)", open=False):
                        char_files = gr.File(label="Character images (char1, char2, ...)",
                                             file_count="multiple", file_types=["image"])
                        env_files = gr.File(label="Environment images (env1, env2, ...)",
                                            file_count="multiple", file_types=["image"])
                    start_img = gr.Image(label="Start frame (اختیاری؛ Single / صحنه اول Story)",
                                         type="pil", height=200)

                    with gr.Group(visible=True) as grp_single:
                        cont = gr.Checkbox(
                            value=False,
                            label="ادامه از آخرین فریم ویدیوی انتخاب‌شده (Continue، seed ثابت)",
                            info="Start frame نادیده گرفته می‌شود.",
                        )
                        sel_info = gr.Markdown("_ویدیویی انتخاب نشده._")
                        sel_frame = gr.Image(label="آخرین فریم ویدیوی انتخاب‌شده", height=130,
                                             interactive=False)

                    with gr.Group(visible=False) as grp_story:
                        n_scenes = gr.Slider(1, 12, value=3, step=1, label="Number of scenes")
                        scene_len = gr.Slider(MIN_SEC, MAX_SEC, value=8, step=1,
                                              label="Seconds per scene")
                        auto_len = gr.Checkbox(value=False, label="Gemma طول هر صحنه را خودش تعیین کند")

                    with gr.Group(visible=False) as grp_edit:
                        edit_video = gr.Video(label="Video to edit")
                        edit_kind = gr.Radio(EDIT_KINDS, value=EDIT_KINDS[0], label="Edit type")
                        edit_cut = gr.Number(value=3, label="T (seconds) برای «Redo from second T»")
                        edit_join = gr.Checkbox(value=True, label="بخش جدید به ویدیوی اصلی چسبانده شود")

                    strength = gr.Slider(
                        0.3, 1.0, value=1.0, step=0.05,
                        label="Start-frame strength (1.0 = دقیقاً همین فریم اول)",
                    )
                    with gr.Row():
                        duration = gr.Slider(MIN_SEC, MAX_SEC, value=8, step=1,
                                             label="Duration (s) — Single / Edit")
                        seed = gr.Number(value=-1, precision=0, label="Seed (-1 = random)")
                    with gr.Row():
                        preset = gr.Dropdown(["HD", "Medium", "Low"], value="HD", label="Quality")
                        aspect = gr.Dropdown(ASPECTS, value=ASPECTS[0], label="Aspect")
                    with gr.Accordion("Style & audio", open=False):
                        style = gr.Textbox(value=DEFAULT_STYLE, lines=4, label="Visual style")
                        audio_wishes = gr.Textbox(value=DEFAULT_AUDIO, lines=3, label="Audio wishes")

                with gr.Column(scale=1):
                    with gr.Group(visible=False) as grp_plan:
                        plan_box = gr.Textbox(
                            label="Story plan (JSON، قابل ویرایش) — خالی = Gemma خودکار می‌سازد",
                            lines=10)
                        btn_plan = gr.Button("0. Plan scenes (Gemma)")
                    prompt_box = gr.Textbox(
                        label="Final prompt (فقط Single؛ برای بخش اول، اختیاری)", lines=8,
                        placeholder="خالی بگذار تا خودکار ساخته شود، یا دکمه‌ی 1 را بزن و ویرایشش کن.",
                    )
                    with gr.Row():
                        btn_prompt = gr.Button("1. Write prompt (Gemma)")
                        btn_video = gr.Button("2. Generate", variant="primary")
                    video_out = gr.Video(label="Result", height=340)
                    info_out = gr.Textbox(label="Info", interactive=False)
                    with gr.Accordion("Prompt used", open=False):
                        used_prompt = gr.Textbox(lines=10, interactive=False, show_label=False)

            gr.Markdown("### Gallery — روی یک ویدیو کلیک کن تا پخش و برای Continue انتخاب شود")
            gallery = gr.Gallery(
                label="Gallery", columns=6, height=300, object_fit="cover",
                allow_preview=False, show_label=False,
            )
            with gr.Row():
                btn_refresh = gr.Button("Refresh gallery", size="sm")
                btn_delete = gr.Button("Delete selected", size="sm", variant="stop")

            mode.change(ui_mode, mode, [grp_single, grp_story, grp_edit, grp_plan, btn_prompt])

            btn_prompt.click(
                ui_write,
                inputs=[idea, duration, style, audio_wishes, start_img, cont, sel_state,
                        char_files, env_files],
                outputs=prompt_box, concurrency_limit=1, concurrency_id="gpu",
            )
            btn_plan.click(
                ui_plan,
                inputs=[idea, n_scenes, scene_len, auto_len, style, audio_wishes, start_img,
                        char_files, env_files],
                outputs=plan_box, concurrency_limit=1, concurrency_id="gpu",
            )
            btn_video.click(
                ui_run,
                inputs=[mode, idea, prompt_box, plan_box, duration, preset, aspect, seed, strength,
                        style, audio_wishes, start_img, char_files, env_files, cont, sel_state,
                        n_scenes, scene_len, auto_len, edit_video, edit_kind, edit_cut, edit_join],
                outputs=[video_out, used_prompt, prompt_box, info_out, gallery,
                         entries_state, sel_state, sel_frame, sel_info, plan_box],
                concurrency_limit=1, concurrency_id="gpu",
            )
            gallery.select(
                ui_select, inputs=entries_state,
                outputs=[video_out, used_prompt, info_out, sel_state, sel_frame, sel_info],
            )
            btn_refresh.click(ui_refresh, outputs=[gallery, entries_state])
            btn_delete.click(ui_delete, inputs=sel_state,
                             outputs=[gallery, entries_state, sel_state, sel_frame, sel_info])
            demo.load(ui_refresh, outputs=[gallery, entries_state])
            demo.load(ui_status, outputs=status_md)
            if hasattr(gr, "Timer"):
                gr.Timer(5).tick(ui_status, outputs=status_md)

        # ============================================================
        # 8. LAUNCH WITH PUBLIC URL
        # ============================================================

        old = globals().get("ltx_demo")
        if old is not None:
            try:
                old.close()
            except Exception:
                pass

        print("GPU:", torch.cuda.get_device_name(0))
        n_added = sync_existing_videos()
        print(f"Gallery: {len(gallery_entries())} videos ({n_added} imported from earlier runs)")
        print("Pre-loading LTX (first click will be faster)...")
        get_pipeline()

        # --- Compatibility shim -------------------------------------------------
        # Notebook environments (marimo/molab) patch asyncio.run with a version that
        # rejects `loop_factory`, which uvicorn passes on Python 3.13. That kills the
        # Gradio server thread silently. Give uvicorn its own safe runner instead.
        import asyncio
        import uvicorn.server as _uvicorn_server


        def _asyncio_run_compat(main, *, debug=None, loop_factory=None, **kwargs):
            loop = loop_factory() if loop_factory is not None else asyncio.new_event_loop()
            try:
                asyncio.set_event_loop(loop)
                return loop.run_until_complete(main)
            finally:
                try:
                    loop.run_until_complete(loop.shutdown_asyncgens())
                except Exception:
                    pass
                asyncio.set_event_loop(None)
                loop.close()


        if hasattr(_uvicorn_server, "asyncio_run"):
            _uvicorn_server.asyncio_run = _asyncio_run_compat
        # -------------------------------------------------------------------------

        # one GPU job at a time; everyone else waits in line (max MAX_QUEUE)
        demo.queue(max_size=MAX_QUEUE, default_concurrency_limit=1)
        launch_kwargs = dict(
            prevent_thread_lock=True,
            allowed_paths=[str(OUT_DIR)],
            quiet=True,
        )
        if GUI_USERS:
            launch_kwargs["auth"] = list(GUI_USERS.items())

        share_url = None
        try:
            _app, local_url, share_url = demo.launch(share=True, **launch_kwargs)
        except Exception as e:
            print("Public share failed:", type(e).__name__, str(e)[:300])
            demo.close()
            _app, local_url, share_url = demo.launch(share=False, **launch_kwargs)

        globals()["ltx_demo"] = demo

        print("\n========== GUI READY ==========")
        print("Local :", local_url)
        print("Public:", share_url or "(not available)")
        print("Users :", ", ".join(GUI_USERS) if GUI_USERS else "(no login)")

        import marimo as mo
        return mo.md(
            f"### LTX Studio is running\n\n"
            f"**Public URL:** {share_url or 'not available (share failed)'}\n\n"
            f"Local: {local_url}\n\n"
            f"Users: {', '.join(GUI_USERS) if GUI_USERS else 'no login'}"
        )


    _()
    return


if __name__ == "__main__":
    app.run()
