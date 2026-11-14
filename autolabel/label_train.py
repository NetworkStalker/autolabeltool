"""Run a local YOLO detection training job in a separate process."""
from __future__ import annotations

import os
from pathlib import Path
import subprocess
import sys
import tempfile

import yaml

from label_project import IMAGE_EXTENSIONS, class_names, sibling_label


def training_config(dataset):
    dataset = Path(dataset).resolve()
    data = yaml.safe_load(dataset.read_text(encoding="utf-8-sig"))
    if not isinstance(data, dict) or "kpt_shape" in data:
        raise ValueError("Choose a YOLO Detect data.yaml.")
    names = class_names(dataset)
    config = {"path": dataset.parent.as_posix(), "names": names}
    for split in ("train", "val"):
        value = data.get(split)
        if not isinstance(value, str) or not value.strip() or "://" in value:
            raise ValueError(f"data.yaml must specify a local {split} image folder.")
        folder = (dataset.parent / value).resolve()
        images = [path for path in folder.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS]
        if not images:
            raise ValueError(f"The {split} split has no images: {folder}")
        boxes = 0
        for image in images:
            label = sibling_label(image)
            if label is None:
                continue
            for line in label.read_text(encoding="utf-8-sig").splitlines():
                fields = line.split()
                if not fields:
                    continue
                if len(fields) != 5:
                    raise ValueError(f"Expected box labels for YOLO Detect: {label}")
                try:
                    class_id = int(fields[0])
                    coordinates = [float(value) for value in fields[1:]]
                    if not 0 <= class_id < len(names) or not all(0 <= v <= 1 for v in coordinates):
                        raise ValueError
                    if min(coordinates[2:]) <= 0:
                        raise ValueError
                except ValueError as error:
                    raise ValueError(f"Invalid YOLO box label: {label}") from error
                boxes += 1
        if split == "train" and not boxes:
            raise ValueError("The train split has no labeled boxes.")
        config[split] = folder.as_posix()
    return config


def train_dataset(dataset, output, model, epochs, imgsz, device, cancelled, progress):
    config = training_config(dataset)
    if cancelled.is_set():
        return None
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    run = Path(tempfile.mkdtemp(prefix="detect-", dir=output))
    config_file = run / "data.yaml"
    config_file.write_text(yaml.safe_dump(config, allow_unicode=True), encoding="utf-8")
    log = run / "training.log"
    with tempfile.TemporaryDirectory(prefix=".settings-", dir=run) as settings_dir:
        env = dict(os.environ, YOLO_OFFLINE="true", YOLO_AUTOINSTALL="false",
                   YOLO_CONFIG_DIR=settings_dir, PYTHONIOENCODING="utf-8")
        args = [sys.executable, "-u", str(Path(__file__).resolve()), str(config_file),
                model, str(epochs), str(imgsz), device]
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        with log.open("w", encoding="utf-8") as stream:
            process = subprocess.Popen(args, stdout=stream, stderr=subprocess.STDOUT, env=env,
                                       creationflags=flags)
            try:
                last_epoch = 0
                while process.poll() is None and not cancelled.wait(0.4):
                    results = run / "results.csv"
                    if results.is_file():
                        completed = max(0, len(results.read_text(encoding="utf-8").splitlines()) - 1)
                        if completed > last_epoch:
                            progress(min(completed, epochs), epochs)
                            last_epoch = completed
            finally:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=5)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
    best = run / "weights" / "best.pt"
    if not cancelled.is_set() and (process.returncode != 0 or not best.is_file()):
        with log.open("rb") as stream:
            stream.seek(max(0, log.stat().st_size - 4096))
            tail = stream.read().decode("utf-8", errors="replace").splitlines()[-8:]
        raise RuntimeError(f"Training did not produce best.pt. Log: {log}\n" + "\n".join(tail))
    return {"cancelled": cancelled.is_set(), "log": log, "best": best if best.is_file() else None}


def _train(config_file, model_path, epochs, imgsz, device):
    try:
        from ultralytics import YOLO, settings
    except ImportError as error:
        raise RuntimeError("Install requirements-ml.txt to train a model.") from error
    disabled = ("sync", "clearml", "comet", "dvc", "mlflow", "neptune", "raytune", "tensorboard", "wandb")
    settings.update({key: False for key in disabled if key in settings})
    model = YOLO(model_path)
    if model.task != "detect":
        raise ValueError("Select a YOLO detection checkpoint.")
    run = Path(config_file).parent
    model.train(data=config_file, epochs=int(epochs), imgsz=int(imgsz), device=device,
                workers=0, project=str(run.parent), name=run.name, exist_ok=True, amp=False)


if __name__ == "__main__":
    _train(*sys.argv[1:])
