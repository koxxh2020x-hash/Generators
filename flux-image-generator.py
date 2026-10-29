# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "diffusers==0.42.1",
#     "huggingface-hub==1.33.0",
#     "ipython==9.17.1",
# ]
# ///

import marimo

__generated_with = "0.25.2"
app = marimo.App(width="medium", auto_download=["html"])


@app.cell
def _():

    import subprocess
    import sys

    subprocess.check_call([
        sys.executable, "-m", "pip", "install",
        "-U", "accelerate", "gradio"
    ])

    print("نصب وابستگی‌ها تمام شد.")
    print("اگر محیط نیاز به Restart داشت، آن را Restart کن.")

    return


@app.cell
def _():

    def _():
        import gc
        import os
        import getpass
        import torch
        from diffusers import Flux2Pipeline

        MODEL_ID = "black-forest-labs/FLUX.2-dev"

        if not torch.cuda.is_available():
            raise RuntimeError("GPU در دسترس نیست.")

        print("GPU:", torch.cuda.get_device_name(0))
        print(
            "VRAM:",
            round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 2),
            "GiB"
        )

        hf_token = os.environ.get("HF_TOKEN")

        if not hf_token:
            hf_token = getpass.getpass(
                "Hugging Face token: "
            ).strip()

        if not hf_token:
            raise RuntimeError("توکن Hugging Face وارد نشده است.")

        os.environ["HF_TOKEN"] = hf_token

        gc.collect()
        torch.cuda.empty_cache()

        print("در حال بارگذاری FLUX.2-dev...")

        pipe = Flux2Pipeline.from_pretrained(
            MODEL_ID,
            dtype=torch.bfloat16,
            token=hf_token,
        )

        pipe.enable_model_cpu_offload()
        pipe.vae.enable_slicing()
        pipe.vae.enable_tiling()

        print("مدل آمادهٔ تولید تصویر است.")

        return pipe


    pipe = _()

    return (pipe,)


@app.cell
def _(pipe):

    import re
    import gc
    import time
    import zipfile
    import torch
    import gradio as gr

    from pathlib import Path
    from datetime import datetime

    OUTPUT_DIR = Path("flux_outputs")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    STYLES = {
        "بدون سبک اضافه": "",
        "انیمه": "anime illustration, detailed anime art",
        "کارتونی": "stylized cartoon, vibrant colors",
        "سه‌بعدی": "high quality 3D render",
        "سینمایی": "cinematic composition, dramatic lighting",
        "واقع‌گرایانه": "photorealistic, realistic lighting",
        "Low-poly": "low-poly 3D art, geometric shapes",
        "استیک‌من": "simple stick figure illustration",
        "نقاشی دیجیتال": "detailed digital painting",
    }

    SIZES = {
        "مربع": (768, 768),
        "عمودی": (576, 1024),
        "افقی": (1024, 576),
        "پرتره": (768, 1024),
        "منظره": (1024, 768),
    }

    def parse_scenes(text):
        # هر بلوک { ... } یک صحنه است؛ خطوط داخل آن جدا نمی‌شوند.
        scenes = re.findall(r"\{(.*?)\}", text, flags=re.S)
        scenes = [s.strip() for s in scenes if s.strip()]

        if scenes:
            return scenes

        # اگر آکولاد نداشتی، پاراگراف‌های خالی جداکننده‌اند.
        return [
            p.strip()
            for p in re.split(r"\n\s*\n", text)
            if p.strip()
        ]

    def make_prompt(prompt, style, character_ref=False):
        style_text = STYLES[style]
        parts = []

        if character_ref:
            parts.append(
                "Use reference image 1 as the recurring character reference. "
                "Keep the character's recognizable face, hairstyle, clothing, "
                "colors and other visible identity traits consistent. "
                "Adapt the character's pose and expression to the scene."
            )

        parts.append(prompt)

        if style_text:
            parts.append(style_text)

        return ", ".join(parts)

    def generate_one(prompt, reference_images, width, height,
                     steps, seed, output_path):
        kwargs = {
            "prompt": prompt,
            "width": width,
            "height": height,
            "num_inference_steps": int(steps),
            "guidance_scale": 4.0,
        }

        if reference_images:
            kwargs["image"] = (
                reference_images[0]
                if len(reference_images) == 1
                else reference_images
            )

        if seed >= 0:
            kwargs["generator"] = torch.Generator(
                device="cpu"
            ).manual_seed(int(seed))

        with torch.inference_mode():
            result = pipe(**kwargs)

        image = result.images[0]
        image.save(output_path)
        return str(output_path)

    def generate_story(
        story, character, style, size_name, custom_width,
        custom_height, steps, count, seed, progress=gr.Progress()
    ):
        scenes = parse_scenes(story)

        if not scenes:
            raise gr.Error(
                "داستان یا صحنه‌ها را وارد کن. هر صحنه را داخل { } بگذار."
            )

        if len(scenes) > 100:
            raise gr.Error("حداکثر ۱۰۰ صحنه در هر صف مجاز است.")

        if size_name == "سفارشی":
            width = max(256, min(1536, int(custom_width or 768)))
            height = max(256, min(1536, int(custom_height or 1024)))
            width = (width // 16) * 16
            height = (height // 16) * 16
        else:
            width, height = SIZES[size_name]

        total = len(scenes) * int(count)
        batch = OUTPUT_DIR / datetime.now().strftime("%Y%m%d_%H%M%S")
        batch.mkdir(parents=True, exist_ok=True)

        reference_images = [character] if character is not None else []
        logs, gallery = [], []
        done = 0

        for scene_index, scene in enumerate(scenes, start=1):
            prompt = make_prompt(
                scene, style, character_ref=character is not None
            )

            for variation in range(1, int(count) + 1):
                done += 1
                progress(
                    (done - 1) / total,
                    desc=f"صحنه {scene_index}/{len(scenes)}، خروجی {variation}"
                )

                path = batch / f"scene_{scene_index:03d}_v{variation:02d}.png"
                current_seed = (
                    int(seed) + (scene_index - 1) * int(count) + variation - 1
                    if int(seed) >= 0 else -1
                )

                try:
                    generate_one(
                        prompt, reference_images, width, height,
                        steps, current_seed, path
                    )
                    gallery.append(str(path))
                    logs.append(f"موفق: صحنه {scene_index}، خروجی {variation}")
                except Exception as e:
                    logs.append(
                        f"خطا در صحنه {scene_index}، خروجی {variation}: "
                        f"{type(e).__name__}: {e}"
                    )
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                progress(done / total, desc=f"{done} از {total}")

        zip_path = batch.with_suffix(".zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            for filename in gallery:
                zf.write(filename, arcname=Path(filename).name)

        summary = (
            f"پایان صف | صحنه‌ها: {len(scenes)} | "
            f"تصاویر موفق: {len(gallery)} از {total}\n"
            f"پوشه: {batch.resolve()}\n\n"
            + "\n".join(logs)
        )

        return summary, gallery, str(zip_path)

    def edit_image(source, character, instruction, style, size_name,
                   custom_width, custom_height, steps, seed):
        if source is None:
            raise gr.Error("ابتدا تصویر موردنظر را آپلود کن.")
        if not instruction.strip():
            raise gr.Error("توضیح بده چه تغییری می‌خواهی.")

        if size_name == "سفارشی":
            width = max(256, min(1536, int(custom_width or 768)))
            height = max(256, min(1536, int(custom_height or 1024)))
            width = (width // 16) * 16
            height = (height // 16) * 16
        else:
            width, height = SIZES[size_name]

        if character is not None:
            refs = [character, source]
            prompt = (
                "Reference image 1 shows the character to preserve. "
                "Reference image 2 is the base image to edit. "
                "Apply the requested changes to the second image while "
                "preserving the character identity from the first image. "
                + instruction
            )
        else:
            refs = [source]
            prompt = (
                "Edit reference image 1 according to these instructions: "
                + instruction
            )

        style_text = STYLES[style]
        if style_text:
            prompt += ", " + style_text

        output_dir = OUTPUT_DIR / "edits"
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / (
            "edit_" + datetime.now().strftime("%Y%m%d_%H%M%S") + ".png"
        )

        try:
            generate_one(
                prompt, refs, width, height, steps,
                int(seed), output_path
            )
        except Exception as e:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise gr.Error(f"ویرایش ناموفق بود: {type(e).__name__}: {e}")

        return str(output_path), str(output_path)

    with gr.Blocks(title="FLUX Story Studio") as demo:
        gr.Markdown(
            "# FLUX Story Studio\n"
            "ساخت صحنه‌های داستانی، حفظ مرجع شخصیت و ویرایش تصویر"
        )

        with gr.Tab("داستان و صحنه‌ها"):
            story = gr.Textbox(
                label="داستان یا صحنه‌ها",
                lines=12,
                placeholder=(
                    "{\n"
                    "A small character walks through a misty forest.\n"
                    "Cinematic low-poly style.\n"
                    "}\n\n"
                    "{\n"
                    "The same character discovers an abandoned cabin.\n"
                    "Warm light shines through the window.\n"
                    "}"
                ),
            )

            character = gr.Image(
                label="تصویر مرجع کاراکتر (اختیاری)",
                type="pil",
            )

            with gr.Row():
                style = gr.Dropdown(
                    list(STYLES.keys()), value="سینمایی", label="سبک"
                )
                size = gr.Dropdown(
                    list(SIZES.keys()) + ["سفارشی"],
                    value="عمودی", label="اندازه تصویر"
                )

            with gr.Row():
                width = gr.Number(value=768, label="عرض سفارشی", precision=0)
                height = gr.Number(value=1024, label="ارتفاع سفارشی", precision=0)
                steps = gr.Slider(10, 50, value=28, step=1, label="مراحل")
                count = gr.Slider(
                    1, 4, value=1, step=1, label="تعداد تصویر برای هر صحنه"
                )

            seed = gr.Number(
                value=42, precision=0,
                label="Seed ثابت؛ عدد منفی یعنی تصادفی"
            )
            start = gr.Button("شروع ساخت تمام صحنه‌ها", variant="primary")
            status = gr.Textbox(label="گزارش صف", lines=8)
            gallery = gr.Gallery(label="تصاویر ساخته‌شده", columns=3)
            download_zip = gr.File(label="دانلود همه تصاویر به صورت ZIP")

            start.click(
                fn=generate_story,
                inputs=[
                    story, character, style, size, width, height,
                    steps, count, seed
                ],
                outputs=[status, gallery, download_zip],
            )

        with gr.Tab("ویرایش تصویر"):
            source = gr.Image(label="تصویر اصلی", type="pil")
            edit_character = gr.Image(
                label="مرجع شخصیت (اختیاری)", type="pil"
            )
            instruction = gr.Textbox(
                label="تغییر موردنظر",
                lines=4,
                placeholder=(
                    "پس‌زمینه را به جنگل شبانه تغییر بده، "
                    "شخصیت اصلی حفظ شود و نور سینمایی باشد."
                ),
            )

            with gr.Row():
                edit_style = gr.Dropdown(
                    list(STYLES.keys()), value="سینمایی", label="سبک خروجی"
                )
                edit_size = gr.Dropdown(
                    list(SIZES.keys()) + ["سفارشی"],
                    value="مربع", label="اندازه خروجی"
                )

            with gr.Row():
                edit_width = gr.Number(
                    value=768, label="عرض سفارشی", precision=0
                )
                edit_height = gr.Number(
                    value=768, label="ارتفاع سفارشی", precision=0
                )
                edit_steps = gr.Slider(
                    10, 50, value=28, step=1, label="مراحل"
                )

            edit_seed = gr.Number(value=42, precision=0, label="Seed")
            edit_button = gr.Button("ویرایش تصویر", variant="primary")
            edited = gr.Image(label="نتیجه")
            edited_file = gr.File(label="فایل تصویر")

            edit_button.click(
                fn=edit_image,
                inputs=[
                    source, edit_character, instruction, edit_style,
                    edit_size, edit_width, edit_height, edit_steps, edit_seed
                ],
                outputs=[edited, edited_file],
            )

    demo.queue(default_concurrency_limit=1)
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )

    return


if __name__ == "__main__":
    app.run()
