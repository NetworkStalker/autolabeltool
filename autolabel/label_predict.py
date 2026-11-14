"""Local Ultralytics inference and prediction merge policy."""
import os
from dataclasses import replace
from pathlib import Path

from label_project import Annotation, MODEL_EXTENSIONS, Project, open_image

OFFICIAL = {
    "YOLOv5u": ("yolov5{}u.pt", "nsmlx"),
    "YOLOv8": ("yolov8{}.pt", "nsmlx"),
    "YOLOv9": ("yolov9{}.pt", "tsmce"),
    "YOLOv10": ("yolov10{}.pt", "nsmblx"),
    "YOLO11": ("yolo11{}.pt", "nsmlx"),
    "YOLO26": ("yolo26{}.pt", "nsmlx"),
}
TEXT_MODEL = "yolov8s-worldv2.pt"
_MODEL = None
_MODEL_KEY = None


def load_model(source, prompts=()):
    global _MODEL, _MODEL_KEY
    os.environ.setdefault("YOLO_OFFLINE", "true")
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    try:
        from ultralytics import YOLO
    except ImportError as error:
        raise RuntimeError("Install the optional inference dependencies from requirements-ml.txt.") from error
    path = Path(source)
    if path.is_absolute():
        if not path.is_file() or path.suffix.lower() not in MODEL_EXTENSIONS:
            raise ValueError("Select an existing .pt, .onnx, or .engine file.")
        stat = path.stat()
        key = (source, stat.st_mtime_ns, stat.st_size)
    else:
        if source != TEXT_MODEL and source not in {
            fmt.format(size) for fmt, sizes in OFFICIAL.values() for size in sizes
        }:
            raise ValueError("Select an official weight or browse for a local model.")
        key = (source,)
    key += (tuple(prompts),)
    if _MODEL_KEY != key:
        if source == TEXT_MODEL:
            if not prompts:
                raise ValueError("Enter at least one text prompt.")
            try:
                import clip

                if not (hasattr(clip, "load") and hasattr(clip, "tokenize")):
                    raise ImportError("The installed clip module is missing the CLIP API.")
            except ImportError as error:
                raise RuntimeError("Install requirements-text.txt to use Auto-label from text.") from error
        model = YOLO(source, task="detect")
        if model.task != "detect":
            raise ValueError(f"Select a detection model. This model's task is {model.task}.")
        if source == TEXT_MODEL:
            model.set_classes(list(prompts))
        _MODEL, _MODEL_KEY = model, key
    return _MODEL


def inspect_model(source, image_path, device, imgsz):
    model = load_model(source)
    if Path(source).suffix.lower() == ".pt":
        return dict(model.names)
    if not image_path:
        raise ValueError("Import and open a sample image to inspect the model.")
    result = model.predict(open_image(image_path), device=device or None, imgsz=imgsz, verbose=False)[0]
    return dict(result.names)


def iou(first, second):
    ax, ay, ar, ab = first.bounds()
    bx, by, br, bb = second.bounds()
    overlap = max(0, min(ar, br) - max(ax, bx)) * max(0, min(ab, bb) - max(ay, by))
    union = (ar - ax) * (ab - ay) + (br - bx) * (bb - by) - overlap
    return overlap / union if union else 0


def integrate(objects, predictions, policy, threshold):
    if policy == "Replace":
        objects = [obj for obj in objects if not obj.pending]
    combined = list(objects)
    for candidate in predictions:
        if policy in ("Replace", "Merge") and any(
            obj.kind != "point" and obj.class_id == candidate.class_id and iou(obj, candidate) >= threshold
            for obj in combined
        ):
            continue
        combined.append(candidate)
    return combined


def detect(model, image, source, config):
    mapping = config["mapping"]
    original = open_image(image["path"])
    w, h = original.size
    if (w, h) != (image["width"], image["height"]):
        raise ValueError("Image dimensions changed after import.")
    result = model.predict(original, conf=config["conf"], iou=config["iou"],
                           imgsz=config["imgsz"], device=config["device"] or None,
                           classes=sorted(mapping), verbose=False)[0]
    if result.boxes is None:
        raise ValueError("The selected model did not return detection boxes.")
    predictions = []
    for coords, cid, score in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist(), result.boxes.conf.tolist()):
        if int(cid) not in mapping:
            continue
        x1, y1, x2, y2 = coords
        bounds = (max(0, min(w, x1)), max(0, min(h, y1)), max(0, min(w, x2)), max(0, min(h, y2)))
        if min(bounds[2] - bounds[0], bounds[3] - bounds[1]) < config["min_box"]:
            continue
        predictions.append(Annotation(mapping[int(cid)], "box", bounds, score=score,
                                      pending=True, source=source).copy())
    return predictions


def batch(project_path, ids, source, config, preview, cancelled, progress):
    project = Project(project_path)
    errors = []
    previews = {}
    changed = skipped = total_boxes = 0
    try:
        model = load_model(source, config["prompts"]) if source == TEXT_MODEL else load_model(source)
        for index, image_id in enumerate(ids, 1):
            if cancelled.is_set():
                break
            image = project.image(image_id)
            existing = project.objects(image_id)
            try:
                if image["status"] == "Null" or (existing and config["policy"] == "Skip"):
                    skipped += 1
                    continue
                predictions = detect(model, image, source, config)
                updated = integrate(existing, predictions, config["policy"], config["merge_iou"])
                added = sum(obj in predictions for obj in updated)
                if cancelled.is_set():
                    break
                if preview:
                    previews[image_id] = updated
                elif updated != existing:
                    project.save(image_id, updated)
                changed += updated != existing
                total_boxes += added
            except Exception as error:
                errors.append(f"{Path(image['path']).name}: {error}")
            finally:
                progress(index, len(ids))
        return dict(previews=previews, changed=changed, skipped=skipped, boxes=total_boxes,
                    errors=errors, cancelled=cancelled.is_set())
    finally:
        project.close()


def approve(objects):
    return [replace(item, pending=False) if item.pending else item for item in objects]

