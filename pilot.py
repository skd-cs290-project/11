"""Offline-ready checks and inference stages for the one-card pilot."""

import argparse
import json
import platform
import sys
import time
from pathlib import Path

from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
BASE_ID = "stable-diffusion-v1-5/stable-diffusion-v1-5"
CONTROL_ID = "lllyasviel/control_v11p_sd15_canny"
CAPTION = "a rainy daytime urban street, seen from a dashboard camera, wet road and realistic reflections"
EDIT = "Turn this clear daytime street into a rainy daytime street"
ADAPTER_WEIGHT = 0.8
SEED = 290

BASE_FILES = (
    "model_index.json", "scheduler/scheduler_config.json",
    "text_encoder/config.json", "text_encoder/model.safetensors",
    "tokenizer/vocab.json", "tokenizer/merges.txt", "tokenizer/tokenizer_config.json",
    "unet/config.json", "unet/diffusion_pytorch_model.safetensors",
    "vae/config.json", "vae/diffusion_pytorch_model.safetensors",
)
CONTROL_FILES = ("config.json", "diffusion_pytorch_model.safetensors")


def model_paths(work):
    return work / "models" / "sd15", work / "models" / "canny"


def stage_models(work):
    from huggingface_hub import HfApi, snapshot_download

    base, control = model_paths(work)
    version_path = work / "models" / "revisions.json"
    revisions = json.loads(version_path.read_text()) if version_path.is_file() else {}
    hub = HfApi()
    for repo, target, files in (
        (BASE_ID, base, [*BASE_FILES, "tokenizer/*.json"]),
        (CONTROL_ID, control, list(CONTROL_FILES)),
    ):
        revision = revisions.get(repo) or hub.model_info(repo).sha
        if not revision:
            raise RuntimeError(f"Could not resolve model revision: {repo}")
        snapshot_download(repo, revision=revision, local_dir=target, allow_patterns=files)
        revisions[repo] = revision
    verify_models(work)
    version_path.write_text(json.dumps(revisions, indent=2) + "\n", encoding="utf-8")


def verify_models(work):
    from safetensors import safe_open

    for root, files in zip(model_paths(work), (BASE_FILES, CONTROL_FILES)):
        for name in files:
            path = root / name
            if name.endswith(".safetensors"):
                with safe_open(str(path), framework="pt", device="cpu") as weights:
                    if not weights.keys():
                        raise ValueError(f"Empty model checkpoint: {path}")
            elif name.endswith(".json"):
                json.loads(path.read_text(encoding="utf-8"))
            elif not path.stat().st_size:
                raise ValueError(f"Empty model component: {path}")


def sources(work):
    return sorted((work / "data" / "clear_test").glob("*.png"))


def preflight(work):
    import accelerate
    import cv2
    import datasets
    import diffusers
    import peft
    import torch
    import torchvision
    from diffusers import ControlNetModel, StableDiffusionControlNetImg2ImgPipeline

    # Diffusers exposes these classes lazily, so importing its package alone misses errors.
    print(f"Inference imports: {ControlNetModel.__name__}, {StableDiffusionControlNetImg2ImgPipeline.__name__}")

    libs = (torch, torchvision, diffusers, peft, datasets, accelerate, cv2)
    print(json.dumps({lib.__name__: lib.__version__ for lib in libs}, indent=2), flush=True)
    if torch.version.cuda != "12.1":
        raise RuntimeError(f"Expected PyTorch CUDA 12.1 wheel, got {torch.version.cuda!r}")
    print(f"Python {platform.python_version()} on {platform.platform()}", flush=True)
    verify_models(work)
    lora = work / "rain_lora" / "pytorch_lora_weights.safetensors"
    if lora.exists():
        from safetensors import safe_open

        with safe_open(str(lora), framework="pt", device="cpu") as weights:
            keys = weights.keys()
            if not keys or any(not key.startswith("unet.") for key in keys):
                raise ValueError(f"Expected the SD 1.5 Diffusers UNet LoRA format: {lora}")
    directory = work / "data"
    manifest = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    expected_test = {Path(name).stem for name in manifest["val_clear"]}
    test = sources(work)
    if not expected_test or expected_test != {p.stem for p in test}:
        raise ValueError("Prepared test images disagree with the manifest.")
    training_dir = directory / "rain_train"
    records = [
        json.loads(line) for line in
        (training_dir / "metadata.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    expected_train = {Path(name).stem for name in manifest["train_rainy"]}
    training = sorted(training_dir.glob("*.png"))
    if not expected_train or {p.stem for p in training} != expected_train:
        raise ValueError("Prepared training images disagree with the manifest.")
    if {row["file_name"] for row in records} != {p.name for p in training}:
        raise ValueError("Caption records do not cover the training images.")
    if expected_train & expected_test:
        raise ValueError("Training and test image IDs overlap.")
    for path in [*training, *test]:
        with Image.open(path) as image:
            if image.size != (768, 432):
                raise ValueError(f"Unexpected size for {path}: {image.size}")
            image.verify()
    print(f"Login-node preflight passed: {len(training)} train, {len(test)} test.", flush=True)


def cuda_check(work):
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError(f"CUDA is unavailable with torch={torch.__version__}, wheel={torch.version.cuda}")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"Expected one visible GPU, got {torch.cuda.device_count()}")
    device = torch.cuda.get_device_properties(0)
    if "A40" not in device.name:
        raise RuntimeError(f"This pilot requests an A40, got {device.name}")
    # Exercise CUDA and FP16 GEMM, not just device enumeration.
    value = torch.ones((32, 32), device="cuda", dtype=torch.float16)
    if (value @ value).mean().item() != 32:
        raise RuntimeError("CUDA FP16 multiplication check failed")
    report = {
        "gpu": device.name, "vram_bytes": device.total_memory,
        "compute_capability": [device.major, device.minor],
        "torch": torch.__version__, "torch_cuda": torch.version.cuda,
    }
    (work / "cuda.json").write_text(json.dumps(report, indent=2) + "\n")
    print(report, flush=True)


def save_png(image, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    image.save(temporary, format="PNG")
    temporary.replace(path)


def load_shared_lora(pipe, work):
    pipe.load_lora_weights(str(work / "rain_lora"), adapter_name="rain", local_files_only=True)
    pipe.set_adapters(["rain"], adapter_weights=[ADAPTER_WEIGHT])


def baseline(work):
    import cv2
    import numpy as np
    import torch
    from diffusers import ControlNetModel, DDIMScheduler, StableDiffusionControlNetImg2ImgPipeline

    base, control = model_paths(work)
    controlnet = ControlNetModel.from_pretrained(str(control), local_files_only=True, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetImg2ImgPipeline.from_pretrained(
        str(base), controlnet=controlnet, local_files_only=True,
        torch_dtype=torch.float16, safety_checker=None, feature_extractor=None,
    ).to("cuda")
    pipe.scheduler = DDIMScheduler.from_config(pipe.scheduler.config)
    load_shared_lora(pipe, work)
    with torch.inference_mode():
        for index, path in enumerate(sources(work)):
            output = work / "results" / "baseline" / path.name
            if output.is_file():
                continue
            with Image.open(path) as source:
                image = source.convert("RGB")
            greyscale = cv2.cvtColor(np.asarray(image), cv2.COLOR_RGB2GRAY)
            edge = Image.fromarray(cv2.Canny(greyscale, 100, 200)).convert("RGB")
            result = pipe(
                CAPTION, image=image, control_image=edge, strength=0.75,
                controlnet_conditioning_scale=0.8, guidance_scale=7.0, num_inference_steps=30,
                generator=torch.Generator(device="cuda").manual_seed(SEED + index),
            ).images[0]
            save_png(result, output)
            print(f"Baseline {index + 1}: {output}", flush=True)


def optimize(work, steps):
    import copy
    import numpy as np
    import torch
    from diffusers.optimization import get_cosine_schedule_with_warmup

    sys.path.insert(0, str(ROOT / "vendor" / "sms"))
    from sms import SMS, SMSConfig

    class RainSMS(SMS):
        # Reuse the upstream optimization with the standard Diffusers LoRA format.
        def load_lora_weights_civitai(self, pipeline_src, *_args, **_kwargs):
            teacher = copy.deepcopy(pipeline_src)
            load_shared_lora(teacher, work)
            return teacher.unet.eval(), teacher.text_encoder

    torch.manual_seed(SEED)
    torch.backends.cuda.matmul.allow_tf32 = True
    base, _ = model_paths(work)
    sms = RainSMS(SMSConfig(
        sd_pretrained_model_or_path=str(base), device="cuda:0",
        src_prompt=EDIT, tgt_prompt=CAPTION, guidance_scale=4.5,
        lora_path=str(work / "rain_lora"),
    ))
    fitting = [p for p in sms.unet.parameters() if p.requires_grad]
    if not fitting:
        raise RuntimeError("SMS's per-image adapter was not initialized.")
    initial = [p.detach().cpu().clone() for p in fitting]
    images = sources(work)
    for index, path in enumerate(images):
        output = work / "results" / "sms" / path.name
        if output.is_file():
            comparison(work, path)
            continue
        started = time.monotonic()
        torch.manual_seed(SEED + index)
        with torch.no_grad():
            for parameter, value in zip(fitting, initial):
                parameter.copy_(value)
            with Image.open(path) as image:
                array = np.array(image.convert("RGB"), dtype=np.float32) / 255.0
            x = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0).to("cuda")
            src = sms.encode_image(x).detach()
        tgt = src.clone().requires_grad_(True)
        image_opt = torch.optim.AdamW([tgt], lr=0.1)
        image_schedule = get_cosine_schedule_with_warmup(
            image_opt, num_warmup_steps=min(100, steps // 5), num_training_steps=int(steps * 1.5),
        )
        fake_opt = torch.optim.AdamW(fitting, lr=1e-3, weight_decay=0.0)
        for step in range(steps):
            image_opt.zero_grad(set_to_none=True)
            loss = sms(
                tgt, src, method="sms", epoch_ratio=step / steps, lambda_dct=0.0005,
            )
            loss.backward()
            image_opt.step()
            image_schedule.step()
            fake_opt.zero_grad(set_to_none=True)
            fake_loss = sms(tgt, src, method="diffusion")
            fake_loss.backward()
            fake_opt.step()
            if (step + 1) % 50 == 0 or step + 1 == steps:
                if not torch.isfinite(tgt).all().item() or not torch.isfinite(fake_loss).item():
                    raise FloatingPointError(f"Non-finite SMS value in {path.name}, step {step + 1}")
                print(f"SMS {index + 1} {path.name} {step + 1}/{steps}", flush=True)
        with torch.no_grad():
            decoded = sms.decode_latent(tgt)[0].permute(1, 2, 0).mul(255).round()
            result = Image.fromarray(decoded.to(torch.uint8).cpu().numpy())
        save_png(result, output)
        comparison(work, path)
        seconds = time.monotonic() - started
        print(f"SMS {index + 1}/{len(images)}: {output}, {seconds:.1f}s", flush=True)


def comparison(work, source):
    destination = work / "results" / "comparison" / source.name
    paths = [source, work / "results" / "baseline" / source.name, work / "results" / "sms" / source.name]
    if destination.is_file() or not all(path.is_file() for path in paths):
        return
    canvas = Image.new("RGB", (768 * 3, 432 + 28), "#f5f5f5")
    draw = ImageDraw.Draw(canvas)
    for index, (path, label) in enumerate(zip(paths, ("Source", "Rain LoRA + Canny", "Same Rain LoRA + SMS"))):
        with Image.open(path) as image:
            canvas.paste(image.convert("RGB"), (index * 768, 28))
        draw.text((index * 768 + 12, 8), label, fill="#161616")
    save_png(canvas, destination)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=("models", "preflight", "cuda", "baseline", "sms", "report"))
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=500)
    args = parser.parse_args()
    if args.steps < 1:
        parser.error("--steps must be positive")
    work = args.work.resolve()
    if args.stage == "sms":
        optimize(work, args.steps)
    elif args.stage == "report":
        names = sources(work)
        if not names:
            raise RuntimeError("No input images found")
        for path in names:
            comparison(work, path)
        count = sum((work / "results" / "comparison" / p.name).is_file() for p in names)
        print(f"{count}/{len(names)} source/baseline/SMS comparisons at {work / 'results/comparison'}")
        if count != len(names):
            raise RuntimeError("Some comparisons are incomplete")
    else:
        {"preflight": preflight, "models": stage_models, "cuda": cuda_check, "baseline": baseline}[args.stage](work)


if __name__ == "__main__":
    main()
