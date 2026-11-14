"""SQLite project storage and image/label import for Label Tool."""
from __future__ import annotations

import json
import math
import sqlite3
from dataclasses import dataclass, replace
from pathlib import Path
from uuid import uuid4

from PIL import Image, ImageOps
import yaml


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
STATUSES = ("Unlabeled", "Predicted", "Labeled", "Reviewed", "Null")
COLORS = ("#ab91ff", "#2dd4bf", "#fbbf24", "#fb7185", "#60a5fa", "#a3e635")
MODEL_EXTENSIONS = {".pt", ".onnx", ".engine"}

SCHEMA = """
CREATE TABLE classes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL COLLATE NOCASE UNIQUE,
    color TEXT NOT NULL,
    position INTEGER NOT NULL,
    hotkey INTEGER UNIQUE CHECK(hotkey BETWEEN 1 AND 9)
);
CREATE TABLE images (
    id INTEGER PRIMARY KEY,
    path TEXT NOT NULL UNIQUE,
    width INTEGER NOT NULL,
    height INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'Unlabeled',
    split TEXT
);
CREATE TABLE annotations (
    image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
    object_id TEXT NOT NULL,
    class_id INTEGER NOT NULL REFERENCES classes(id),
    payload TEXT NOT NULL,
    PRIMARY KEY (image_id, object_id)
);
CREATE INDEX annotations_class ON annotations(class_id);
CREATE TABLE models (source TEXT PRIMARY KEY, settings TEXT NOT NULL);
PRAGMA user_version = 1;
"""


@dataclass(frozen=True)
class Annotation:
    class_id: int
    kind: str
    points: tuple[float, ...]
    id: str = ""
    score: float | None = None
    pending: bool = False
    source: str = "manual"
    keypoints: tuple[tuple[float, float, int], ...] = ()

    def copy(self):
        return replace(self, id=uuid4().hex)

    def bounds(self):
        xs, ys = self.points[::2], self.points[1::2]
        return min(xs), min(ys), max(xs), max(ys)


def status_for(objects):
    if any(item.pending for item in objects):
        return "Predicted"
    return "Labeled" if objects else "Unlabeled"


def open_image(path):
    with Image.open(path) as image:
        return ImageOps.exif_transpose(image).convert("RGB")


def image_size(path):
    with Image.open(path) as image:
        w, h = image.size
        return (h, w) if image.getexif().get(274) in (5, 6, 7, 8) else (w, h)


def class_names(path):
    path = Path(path)
    if path.suffix.lower() == ".txt":
        names = path.read_text(encoding="utf-8-sig").splitlines()
    else:
        data = yaml.safe_load(path.read_text(encoding="utf-8-sig"))
        names = data.get("names") if isinstance(data, dict) else None
        if isinstance(names, dict):
            ordered = {int(key): value for key, value in names.items()}
            if set(ordered) != set(range(len(ordered))):
                raise ValueError("YOLO class IDs must start at 0 and be consecutive.")
            names = [ordered[index] for index in range(len(ordered))]
    if not isinstance(names, list) or not names or not all(isinstance(n, str) and n.strip() for n in names):
        raise ValueError("Provide a class name on each line or a YOLO YAML with names.")
    names = [name.strip() for name in names]
    if len({name.casefold() for name in names}) != len(names):
        raise ValueError("Class names must be unique.")
    return names


def parse_yolo(path, ids, width, height, task="detect", keypoint_shape=None):
    objects = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            fields = line.split()
            class_index = int(fields[0])
            if not 0 <= class_index < len(ids):
                raise ValueError("unknown class ID")
            values = tuple(float(v) for v in fields[1:])
            if not all(math.isfinite(v) for v in values):
                raise ValueError("non-finite coordinate")
            keypoints = ()
            if task == "seg" or (task == "detect" and len(values) > 4 and len(values) % 2 == 0):
                if len(values) < 6 or len(values) % 2 or not all(0 <= v <= 1 for v in values):
                    raise ValueError("expected normalized polygon vertices")
                kind = "polygon"
                points = tuple(v * (width if i % 2 == 0 else height) for i, v in enumerate(values))
            else:
                if len(values) < 4 or (task == "detect" and len(values) != 4):
                    raise ValueError("expected cx cy width height")
                cx, cy, w, h = values[:4]
                if not all(0 <= v <= 1 for v in (cx, cy, w, h)) or not w or not h:
                    raise ValueError("invalid normalized box")
                points = (max(0, cx - w / 2) * width, max(0, cy - h / 2) * height,
                          min(1, cx + w / 2) * width, min(1, cy + h / 2) * height)
                kind = "box"
                if task == "pose":
                    count, dimensions = keypoint_shape
                    if len(values) != 4 + count * dimensions:
                        raise ValueError("keypoints do not match kpt_shape")
                    joints = []
                    for i in range(4, len(values), dimensions):
                        x, y = values[i:i + 2]
                        visible = values[i + 2] if dimensions == 3 else 2
                        if visible not in (0, 1, 2) or (visible and not (0 <= x <= 1 and 0 <= y <= 1)):
                            raise ValueError("invalid keypoint")
                        joints.append((x * width, y * height, int(visible)) if visible else (0, 0, 0))
                    keypoints = tuple(joints)
            objects.append(Annotation(ids[class_index], kind, points, uuid4().hex, keypoints=keypoints))
        except (IndexError, ValueError) as error:
            raise ValueError(f"{path.name}:{line_no}: {error}") from error
    return objects


def sibling_label(path):
    for parent in path.parents:
        if parent.name.casefold() == "images":
            candidate = (parent.parent / "labels" / path.relative_to(parent)).with_suffix(".txt")
            if candidate.is_file():
                return candidate
    candidate = path.with_suffix(".txt")
    return candidate if candidate.is_file() else None


class Project:
    def __init__(self, path, create=False):
        self.path = Path(path).resolve()
        if create and self.path.exists():
            raise FileExistsError("Select a new project filename.")
        if not create and not self.path.is_file():
            raise FileNotFoundError(self.path)
        self.db = sqlite3.connect(self.path)
        self.db.row_factory = sqlite3.Row
        try:
            self.db.execute("PRAGMA foreign_keys = ON")
            if create:
                self.db.executescript(SCHEMA)
            if self.db.execute("PRAGMA user_version").fetchone()[0] != 1:
                raise ValueError("This file is not a supported Label Tool project.")
            self.db.execute("PRAGMA journal_mode = WAL")
        except Exception:
            self.db.close()
            raise

    def close(self):
        self.db.close()

    def classes(self):
        return self.db.execute("""
            SELECT c.*, (SELECT COUNT(*) FROM annotations a WHERE a.class_id = c.id) AS count
            FROM classes c ORDER BY position, id
        """).fetchall()

    def add_class(self, name, color=None):
        with self.db:
            return self._add_class(name, color)

    def _add_class(self, name, color=None):
        name = name.strip()
        if not name:
            raise ValueError("Enter a class name.")
        rows = self.db.execute("SELECT position, hotkey FROM classes").fetchall()
        order = max((row["position"] for row in rows), default=-1) + 1
        used = {row["hotkey"] for row in rows}
        key = next((number for number in range(1, 10) if number not in used), None)
        return self.db.execute(
            "INSERT INTO classes(name, color, position, hotkey) VALUES (?, ?, ?, ?)",
            (name, color or COLORS[order % len(COLORS)], order, key),
        ).lastrowid

    def import_classes(self, names):
        with self.db:
            ids = {row["name"].casefold(): row["id"] for row in self.classes()}
            imported = []
            for name in names:
                key = name.strip().casefold()
                if key not in ids:
                    ids[key] = self._add_class(name)
                imported.append(ids[key])
            return imported

    def edit_class(self, class_id, name, color, hotkey):
        if not name.strip():
            raise ValueError("Enter a class name.")
        with self.db:
            if hotkey is not None:
                self.db.execute("UPDATE classes SET hotkey = NULL WHERE hotkey = ?", (hotkey,))
            self.db.execute(
                "UPDATE classes SET name = ?, color = ?, hotkey = ? WHERE id = ?",
                (name.strip(), color, hotkey, class_id),
            )

    def move_class(self, class_id, step):
        rows = self.classes()
        index = next(i for i, row in enumerate(rows) if row["id"] == class_id)
        other = index + step
        if 0 <= other < len(rows):
            with self.db:
                self.db.executemany("UPDATE classes SET position = ? WHERE id = ?", [
                    (rows[other]["position"], class_id),
                    (rows[index]["position"], rows[other]["id"]),
                ])

    def remove_class(self, class_id, replacement=None):
        if class_id == replacement:
            raise ValueError("Select a different destination class.")
        with self.db:
            if replacement is None:
                affected = self.db.execute(
                    "SELECT DISTINCT image_id FROM annotations WHERE class_id = ?", (class_id,)
                ).fetchall()
                self.db.execute("DELETE FROM annotations WHERE class_id = ?", (class_id,))
                for (image_id,) in affected:
                    self.db.execute("UPDATE images SET status = ? WHERE id = ?",
                                    (status_for(self.objects(image_id)), image_id))
            else:
                self.db.execute("UPDATE annotations SET class_id = ? WHERE class_id = ?", (replacement, class_id))
            for source, settings in self.models().items():
                changed = False
                for field in ("mapping", "class_targets"):
                    mapping = settings.get(field, {})
                    if class_id in mapping.values():
                        settings[field] = {
                            key: replacement if value == class_id else value
                            for key, value in mapping.items()
                            if replacement is not None or value != class_id
                        }
                        changed = True
                if changed:
                    self.db.execute("UPDATE models SET settings = ? WHERE source = ?", (json.dumps(settings), source))
            self.db.execute("DELETE FROM classes WHERE id = ?", (class_id,))

    def image(self, image_id):
        return self.db.execute("SELECT * FROM images WHERE id = ?", (image_id,)).fetchone()

    def images(self, status=None, search="", class_id=None):
        rows = self.db.execute("""
            SELECT * FROM images i
            WHERE (? IS NULL OR i.status = ?) AND instr(lower(i.path), lower(?)) > 0
            AND (? IS NULL OR EXISTS (
                SELECT 1 FROM annotations a WHERE a.image_id = i.id AND a.class_id = ?
            )) ORDER BY id
        """, (status, status, search, class_id, class_id)).fetchall()
        available = []
        for row in rows:
            try:
                if not Path(row["path"]).is_file():
                    continue
            except OSError:
                # Keep inaccessible files visible so a temporary access problem
                # does not silently remove them from the user's view.
                pass
            available.append(row)
        return available

    def objects(self, image_id):
        objects = []
        for row in self.db.execute("SELECT * FROM annotations WHERE image_id = ? ORDER BY rowid", (image_id,)):
            data = json.loads(row["payload"])
            data["points"] = tuple(data["points"])
            data["keypoints"] = tuple(tuple(p) for p in data.get("keypoints", ()))
            objects.append(Annotation(row["class_id"], id=row["object_id"], **data))
        return objects

    def save(self, image_id, objects, status=None):
        with self.db:
            self._save(image_id, objects, status or status_for(objects))

    def _save(self, image_id, objects, status):
        self.db.execute("DELETE FROM annotations WHERE image_id = ?", (image_id,))
        self.db.executemany("INSERT INTO annotations VALUES (?, ?, ?, ?)", [
            (image_id, item.id, item.class_id, json.dumps(dict(
                kind=item.kind, points=item.points, score=item.score, pending=item.pending,
                source=item.source, keypoints=item.keypoints,
            ))) for item in objects
        ])
        self.db.execute("UPDATE images SET status = ? WHERE id = ?", (status, image_id))

    def set_split(self, image_ids, split):
        with self.db:
            self.db.executemany("UPDATE images SET split = ? WHERE id = ?", [(split, i) for i in image_ids])

    def import_images(self, paths, yolo_names=None, task="detect", keypoint_shape=None, mark_split=False):
        errors = []
        count = 0
        with self.db:
            ids = None
            if yolo_names is not None:
                known = {row["name"].casefold(): row["id"] for row in self.classes()}
                ids = [known[name.casefold()] if name.casefold() in known else self._add_class(name)
                       for name in yolo_names]
            for path in dict.fromkeys(Path(p).resolve() for p in paths):
                if path.suffix.lower() not in IMAGE_EXTENSIONS or not path.is_file():
                    continue
                existing = self.db.execute("SELECT * FROM images WHERE path = ?", (str(path),)).fetchone()
                if existing and (ids is None or existing["status"] != "Unlabeled"):
                    continue
                try:
                    label = sibling_label(path) if ids is not None else None
                    if existing and not label:
                        continue
                    width, height = image_size(path)
                    if existing and (width, height) != (existing["width"], existing["height"]):
                        raise ValueError("Image dimensions changed after import.")
                    objects = parse_yolo(label, ids, width, height, task, keypoint_shape) if label else []
                except (OSError, ValueError) as error:
                    errors.append(f"{path.name}: {error}")
                    continue
                split = None
                if mark_split:
                    for parent in path.parents:
                        if parent.name in ("train", "valid", "val", "test"):
                            split = "val" if parent.name == "valid" else parent.name
                            break
                if existing:
                    image_id = existing["id"]
                    self.db.execute("UPDATE images SET split = COALESCE(split, ?) WHERE id = ?", (split, image_id))
                else:
                    image_id = self.db.execute(
                        "INSERT INTO images(path, width, height, split) VALUES (?, ?, ?, ?)",
                        (str(path), width, height, split),
                    ).lastrowid
                if label:
                    self._save(image_id, objects, "Labeled" if objects else "Null")
                count += 1
        return count, errors

    def register_model(self, source, settings):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO models VALUES (?, ?)", (source, json.dumps(settings)))

    def models(self):
        return {row["source"]: json.loads(row["settings"])
                for row in self.db.execute("SELECT * FROM models ORDER BY rowid")}

