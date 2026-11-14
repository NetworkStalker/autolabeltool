"""Profile-driven YOLO export with reproducible splits."""
from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import tempfile
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from PIL import Image
import yaml

from label_project import open_image


PROFILES = json.loads(Path(__file__).with_name("yolo_profiles.json").read_text(encoding="utf-8"))
SPLITS = ("train", "val", "test")


def split_images(images, ratios, seed):
    if len(ratios) != 3 or not all(math.isfinite(r) and r >= 0 for r in ratios) or not sum(ratios):
        raise ValueError("Enter three non-negative split weights with a positive total.")
    groups = {key: [] for key in SPLITS}
    unassigned = []
    for image in images:
        if image["split"] in groups:
            groups[image["split"]].append(image)
        else:
            unassigned.append(image)
    random.Random(seed).shuffle(unassigned)
    total = len(images)
    ideal = {key: total * (ratio / sum(ratios)) for key, ratio in zip(SPLITS, ratios)}
    positive = [key for key in SPLITS if ideal[key] > 0]
    for key in positive:
        if unassigned and not groups[key]:
            groups[key].append(unassigned.pop())
    for image in unassigned:
        key = max(positive, key=lambda name: ideal[name] - len(groups[name]))
        groups[key].append(image)
    return groups


def box_row(bounds, width, height):
    x1, y1, x2, y2 = bounds
    return ((x1 + x2) / (2 * width), (y1 + y2) / (2 * height),
            (x2 - x1) / width, (y2 - y1) / height)


def point_box(x, y, side, width, height):
    side_x, side_y = min(side, width), min(side, height)
    x1 = min(max(0, x - side_x / 2), width - side_x)
    y1 = min(max(0, y - side_y / 2), height - side_y)
    return x1, y1, x1 + side_x, y1 + side_y


def as_row(item, width, height, task, point_mode, side, joints):
    if item.kind == "point":
        if task != "detect" or point_mode != "box":
            return None
        return box_row(point_box(*item.points, side, width, height), width, height)
    if task == "seg":
        points = item.points
        if item.kind == "box":
            x1, y1, x2, y2 = item.bounds()
            points = (x1, y1, x2, y1, x2, y2, x1, y2)
        return tuple(v / (height if i % 2 else width) for i, v in enumerate(points))
    bounds = box_row(item.bounds(), width, height)
    if task != "pose":
        return bounds
    if not item.keypoints:
        return None
    keypoints = item.keypoints + ((0, 0, 0),) * (joints - len(item.keypoints))
    return bounds + tuple(v for x, y, visible in keypoints for v in (x / width, y / height, visible))


def _copy_image(source, target, width, height):
    with Image.open(source) as image:
        orientation = image.getexif().get(274, 1)
        actual_size = image.size[::-1] if orientation in (5, 6, 7, 8) else image.size
        if actual_size != (width, height):
            raise ValueError(f"Image dimensions changed after import: {source}")
    if orientation in range(2, 9):
        open_image(source).save(target.with_suffix(".png"))
    else:
        shutil.copy2(source, target.with_suffix(source.suffix.lower()))


def _train_script(profile, task):
    if profile["layout"] == "split":
        return """import argparse
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("repository", type=Path, help="Path to a local YOLOv5 checkout")
args, extra = parser.parse_known_args()
root = Path(__file__).resolve().parent
entry = args.repository.resolve() / %r
subprocess.run([sys.executable, str(entry), "--data", str(root / "data.yaml"),
                "--weights", %r, *extra], cwd=entry.parent, check=True)
""" % ("segment/train.py" if task == "seg" else "train.py",
       profile.get(f"{task}_weight", profile["weights"]))
    return """from pathlib import Path
from tempfile import TemporaryDirectory
import yaml
from ultralytics import YOLO

root = Path(__file__).resolve().parent
data = yaml.safe_load((root / "data.yaml").read_text(encoding="utf-8"))
data["path"] = root.as_posix()
with TemporaryDirectory() as temp:
    config = Path(temp) / "data.yaml"
    config.write_text(yaml.safe_dump(data, allow_unicode=True), encoding="utf-8")
    YOLO(%r).train(data=str(config), epochs=100, imgsz=640)
""" % profile.get(f"{task}_weight", profile["weights"])


def export_yolo(project, destination, version="v11", task="detect", ratios=(70, 20, 10),
                seed=42, include_null=True, reviewed_only=False, point_mode="skip",
                point_size=16, zip_output=False, progress=None):
    profile = PROFILES[version]
    if task not in profile["tasks"]:
        raise ValueError(f"{version} does not support {task} in this exporter.")
    if point_mode not in ("skip", "box", "csv") or not math.isfinite(point_size) or point_size <= 0:
        raise ValueError("Select a valid point policy and positive point size.")
    dest = Path(destination).resolve()
    zip_path = dest.with_name(dest.name + ".zip")
    if dest.exists() or (zip_output and zip_path.exists()):
        raise FileExistsError("Select a new export folder or archive name.")
    classes = project.classes()
    if not classes:
        raise ValueError("Create at least one class before exporting.")
    mapping = {row["id"]: i for i, row in enumerate(classes)}
    eligible = {"Reviewed", "Null"} if reviewed_only else {"Reviewed", "Labeled", "Null"}
    if not include_null:
        eligible.discard("Null")
    selected = []
    skipped_points = 0
    skipped_objects = 0
    joint_count = 0
    for image in project.images():
        if image["status"] not in eligible:
            continue
        objects = project.objects(image["id"])
        joint_count = max(joint_count, max((len(obj.keypoints) for obj in objects), default=0))
        exportable = [obj for obj in objects if (
            (obj.kind == "point" and task == "detect" and point_mode == "box")
            or (obj.kind != "point" and (task != "pose" or obj.keypoints))
        )]
        skipped_points += sum(obj.kind == "point" and obj not in exportable for obj in objects)
        skipped_objects += sum(obj.kind != "point" and obj not in exportable for obj in objects)
        if exportable or image["status"] == "Null" or (point_mode == "csv" and objects):
            selected.append(dict(image, objects=objects))
    if not selected:
        raise ValueError("No eligible images have annotations for this task. Review predictions before exporting.")
    if task == "pose" and not joint_count:
        raise ValueError("Add at least one keypoint to a box for pose export.")
    # CSV-only images are reported separately, so their objects are not treated as background.
    training = [row for row in selected if row["status"] == "Null" or any(
        as_row(obj, row["width"], row["height"], task, point_mode, point_size, joint_count)
        is not None for obj in row["objects"]
    )]
    groups = split_images(training, ratios, seed)
    counts = {key: len(value) for key, value in groups.items()}
    names = [row["name"] for row in classes]
    stats = dict(counts, skipped_points=skipped_points if point_mode != "csv" else 0,
                 skipped_objects=skipped_objects)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".label-export-", dir=dest.parent) as stage:
        root = Path(stage) / "dataset"
        root.mkdir()
        data = {"path": dest.as_posix()}
        used = set()
        processed = 0
        for split, rows in groups.items():
            image_dir = f"{split}/images" if profile["layout"] == "split" else f"images/{split}"
            label_dir = f"{split}/labels" if profile["layout"] == "split" else f"labels/{split}"
            (root / image_dir).mkdir(parents=True)
            (root / label_dir).mkdir(parents=True)
            if split != "test" or rows:
                data[split] = image_dir
            for image in rows:
                path = Path(image["path"])
                stem = path.stem
                suffix = 1
                while stem.casefold() in used:
                    stem = f"{path.stem}_{suffix}"
                    suffix += 1
                used.add(stem.casefold())
                _copy_image(path, root / image_dir / stem, image["width"], image["height"])
                lines = []
                for obj in image["objects"]:
                    row = as_row(obj, image["width"], image["height"], task, point_mode, point_size, joint_count)
                    if row is not None:
                        lines.append(str(mapping[obj.class_id]) + " " + " ".join(f"{v:.8f}" for v in row))
                (root / label_dir / f"{stem}.txt").write_text(
                    "\n".join(lines) + ("\n" if lines else ""), encoding="utf-8"
                )
                processed += 1
                if progress:
                    progress(processed, len(training))
        if profile["layout"] == "split":
            data["nc"] = len(names)
            data["names"] = names
        else:
            data["names"] = dict(enumerate(names))
        if task == "pose":
            data["kpt_shape"] = [joint_count, 3]
        (root / "data.yaml").write_text(yaml.safe_dump(data, allow_unicode=True, sort_keys=False), encoding="utf-8")
        (root / "train.py").write_text(_train_script(profile, task), encoding="utf-8")
        if point_mode == "csv":
            with (root / "points.csv").open("w", encoding="utf-8-sig", newline="") as file:
                writer = csv.writer(file)
                writer.writerow(("image", "class", "x", "y"))
                for image in selected:
                    for obj in image["objects"]:
                        if obj.kind == "point":
                            writer.writerow((image["path"], names[mapping[obj.class_id]], *obj.points))
        note = "" if counts["train"] and counts["val"] else "\nTraining requires non-empty train and val splits.\n"
        (root / "README.dataset.txt").write_text(
            f"YOLO {version} / {task}\nSplit weights: {ratios}, seed: {seed}\n"
            f"Images: {counts}\nPoint policy: {point_mode}\n"
            f"Skipped points: {stats['skipped_points']}\nSkipped objects: {skipped_objects}\n"
            "Run the example train.py or set data.yaml path to the dataset location after moving it.\n"
            + note, encoding="utf-8",
        )
        archive = Path(stage) / "dataset.zip"
        if zip_output:
            with ZipFile(archive, "w", ZIP_DEFLATED) as out:
                for path in root.rglob("*"):
                    out.write(path, path.relative_to(root).as_posix())
        if dest.exists() or (zip_output and zip_path.exists()):
            raise FileExistsError("The export destination appeared while exporting.")
        root.rename(dest)
        if zip_output:
            archive.rename(zip_path)
    return stats

