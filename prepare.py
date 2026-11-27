"""Select disjoint BDD100K splits and prepare a Diffusers ImageFolder."""

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image, ImageOps


def select(labels, images, weather, count, seed):
    rows = json.loads(labels.read_text(encoding="utf-8"))
    if isinstance(rows, dict):
        rows = rows["frames"]
    candidates = {}
    for row in rows:
        attributes = row.get("attributes") or {}
        if (attributes.get("scene"), attributes.get("timeofday"), attributes.get("weather")) != (
            "city street", "daytime", weather
        ):
            continue
        name = row["name"]
        if Path(name).name != name:
            raise ValueError(f"Expected image basename in {labels}: {name}")
        path = images / name
        if path.is_file():
            candidates[path.stem] = path
    if len(candidates) < count:
        raise ValueError(
            f"{images}: found {len(candidates)} {weather} daytime city street images, "
            f"need {count}. Check the image/label split or lower the requested count."
        )
    return random.Random(seed).sample(sorted(candidates.values()), count)


def write_image(source, target):
    with Image.open(source) as image:
        ImageOps.fit(image.convert("RGB"), (768, 432)).save(target)


def fetch_subset(split, weather, count, seed, revision):
    from huggingface_hub import hf_hub_download

    repo = "dronefreak/BDD100K"
    candidates = []
    for index in range({"train": 9, "val": 2}[split]):
        shard = f"data/images_100k/100k/{split}/shard_{index:03}"
        metadata = hf_hub_download(
            repo, f"{shard}/metadata.jsonl", repo_type="dataset", revision=revision,
        )
        with open(metadata, encoding="utf-8") as stream:
            for line in stream:
                row = json.loads(line)
                if (row["scene"], row["timeofday"], row["weather"]) == ("city street", "daytime", weather):
                    name = row["file_name"]
                    if Path(name).name != name:
                        raise ValueError(f"Expected image basename in mirror: {name}")
                    candidates.append(f"{shard}/{name}")
    if len(candidates) < count:
        raise ValueError(f"Mirror split {split} contains {len(candidates)} matches, need {count}")
    selected = random.Random(seed).sample(sorted(candidates), count)

    def download(filename):
        return Path(hf_hub_download(repo, filename, repo_type="dataset", revision=revision))

    with ThreadPoolExecutor(max_workers=6) as pool:
        return list(pool.map(download, selected))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hub", action="store_true", help="Fetch selected images from the BDD100K mirror")
    parser.add_argument("--train-images", type=Path)
    parser.add_argument("--val-images", type=Path)
    parser.add_argument("--train-labels", type=Path)
    parser.add_argument("--val-labels", type=Path)
    parser.add_argument("--work", type=Path, required=True)
    parser.add_argument("--train-count", type=int, default=160)
    parser.add_argument("--test-count", type=int, default=20)
    parser.add_argument("--seed", type=int, default=290)
    args = parser.parse_args()
    if min(args.train_count, args.test_count) < 1:
        parser.error("sample counts must be positive")
    data = args.work / "data"
    if data.exists():
        raise FileExistsError(f"{data} already exists. Use a fresh --work for a different split.")
    if args.hub:
        from huggingface_hub import HfApi

        revision = HfApi().dataset_info("dronefreak/BDD100K").sha
        if not revision:
            raise RuntimeError("Could not resolve the BDD100K mirror revision")
        training = fetch_subset("train", "rainy", args.train_count, args.seed, revision)
        testing = fetch_subset("val", "clear", args.test_count, args.seed, revision)
    else:
        if not all((args.train_labels, args.train_images, args.val_labels, args.val_images)):
            parser.error("provide --hub or all four local BDD100K paths")
        revision = None
        training = select(args.train_labels, args.train_images, "rainy", args.train_count, args.seed)
        testing = select(args.val_labels, args.val_images, "clear", args.test_count, args.seed)
    if {p.stem for p in training} & {p.stem for p in testing}:
        raise ValueError("Training and test image IDs overlap.")
    staging = data.with_name(".prepare-data")
    training_dir, testing_dir = staging / "rain_train", staging / "clear_test"
    training_dir.mkdir(parents=True)
    testing_dir.mkdir()
    records = []
    for source in training:
        name = f"{source.stem}.png"
        write_image(source, training_dir / name)
        records.append(json.dumps({
            "file_name": name,
            "text": "a rainy daytime urban street, seen from a dashboard camera",
        }))
    for source in testing:
        write_image(source, testing_dir / f"{source.stem}.png")
    (training_dir / "metadata.jsonl").write_text("\n".join(records) + "\n", encoding="utf-8")
    (staging / "manifest.json").write_text(json.dumps({
        "dataset": "BDD100K", "mirror_revision": revision, "seed": args.seed, "size": [768, 432],
        "train_rainy": [p.name for p in training], "val_clear": [p.name for p in testing],
    }, indent=2) + "\n", encoding="utf-8")
    staging.rename(data)
    print(f"Prepared {len(training)} rainy training images and {len(testing)} clear test images.")


if __name__ == "__main__":
    main()
