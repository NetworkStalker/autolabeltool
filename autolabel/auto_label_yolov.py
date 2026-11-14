"""
Auto label รูปภาพ แล้วจัดโครงสร้างแบบ Roboflow "YOLOv8" export

ผลลัพธ์:
  my_dataset.yolov8/
  ├── train/
  │   ├── images/
  │   └── labels/
  ├── valid/
  │   ├── images/
  │   └── labels/
  ├── test/
  │   ├── images/
  │   └── labels/
  └── data.yaml

ตัวอย่าง:
  python auto_label_yolov8.py --source raw_images --output cat.yolov8 --classes cat

  # เอาหลาย class + กำหนดสัดส่วน + สร้าง zip ไว้อัปโหลด Roboflow
  python auto_label_yolov8.py --source raw_images --output pets.yolov8 \
      --classes cat dog --split 0.7 0.2 0.1 --zip

ติดตั้ง: pip install ultralytics opencv-python
"""
import argparse
import random
import shutil
from pathlib import Path

import cv2
from ultralytics import YOLO

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = ("train", "valid", "test")


def parse_args():
    p = argparse.ArgumentParser(description="Auto label -> YOLOv8 (Roboflow-style) dataset")
    p.add_argument("--source", required=True, help="โฟลเดอร์รูปต้นฉบับ (รูปรวมกันในโฟลเดอร์เดียว)")
    p.add_argument("--output", required=True, help="ชื่อโฟลเดอร์ dataset ที่จะสร้าง เช่น cat.yolov8")
    p.add_argument("--model", default="yolo11n.pt", help="ไฟล์โมเดล (.pt)")
    p.add_argument("--classes", nargs="*", help="class ที่ต้องการ (ชื่อหรือ id) ไม่ใส่ = ทั้งหมด")
    p.add_argument("--conf", type=float, default=0.4, help="confidence ขั้นต่ำ")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--split", nargs=3, type=float, default=[0.7, 0.2, 0.1],
                   metavar=("TRAIN", "VALID", "TEST"), help="สัดส่วนแบ่งข้อมูล (default 0.7 0.2 0.1)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-empty", action="store_true",
                   help="รวมภาพที่ไม่พบวัตถุ (label ว่าง) ไว้ใน dataset ด้วย")
    p.add_argument("--zip", action="store_true", help="บีบอัดเป็น .zip เพื่ออัปโหลด")
    return p.parse_args()


def build_class_map(model, wanted):
    names = model.names
    name_to_id = {v.lower(): k for k, v in names.items()}

    if not wanted:
        return {i: i for i in names}, [names[i] for i in sorted(names)]

    selected = []
    for c in wanted:
        cid = int(c) if c.isdigit() else name_to_id.get(c.lower())
        if cid is None or cid not in names:
            raise SystemExit(f"ไม่รู้จัก class: {c}")
        selected.append(cid)

    return {cid: new for new, cid in enumerate(selected)}, [names[c] for c in selected]


def split_files(items, ratios):
    total = sum(ratios)
    n = len(items)
    n_train = round(n * ratios[0] / total)
    n_valid = round(n * ratios[1] / total)
    return (
        items[:n_train],
        items[n_train:n_train + n_valid],
        items[n_train + n_valid:],
    )


def main():
    args = parse_args()
    src = Path(args.source)
    out = Path(args.output)

    files = sorted(p for p in src.iterdir() if p.suffix.lower() in IMG_EXT)
    if not files:
        raise SystemExit(f"ไม่พบรูปภาพใน {src}")

    model = YOLO(args.model)
    remap, class_names = build_class_map(model, args.classes)

    # --- 1) ตรวจจับ + เก็บผลไว้ในหน่วยความจำก่อน ---
    results = []  # (path, [label lines])
    no_det, unreadable = [], []

    for i, path in enumerate(files, 1):
        img = cv2.imread(str(path))
        if img is None:
            unreadable.append(path.name)
            continue

        res = model.predict(img, conf=args.conf, imgsz=args.imgsz, verbose=False)[0]
        lines = []
        for box in res.boxes:
            cid = int(box.cls)
            if cid not in remap:
                continue
            x, y, w, h = box.xywhn[0].tolist()
            lines.append(f"{remap[cid]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")

        if not lines:
            no_det.append(path.name)
            if not args.save_empty:
                continue

        results.append((path, lines))
        if i % 50 == 0 or i == len(files):
            print(f"[{i}/{len(files)}]")

    if not results:
        raise SystemExit("ไม่มีภาพที่ตรวจพบวัตถุเลย ลองลด --conf หรือเปลี่ยนโมเดล")

    # --- 2) สุ่มแบ่ง train / valid / test ---
    random.Random(args.seed).shuffle(results)
    parts = dict(zip(SPLITS, split_files(results, args.split)))

    # --- 3) สร้างโครงสร้างโฟลเดอร์ + คัดลอกไฟล์ ---
    if out.exists():
        shutil.rmtree(out)
    for split, items in parts.items():
        (out / split / "images").mkdir(parents=True)
        (out / split / "labels").mkdir(parents=True)
        for path, lines in items:
            shutil.copy2(path, out / split / "images" / path.name)
            (out / split / "labels" / f"{path.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else "")
            )

    # --- 4) data.yaml ---
    names_str = ", ".join(repr(n) for n in class_names)
    (out / "data.yaml").write_text(
        f"path: .\n"
        f"train: train/images\n"
        f"val: valid/images\n"
        f"test: test/images\n"
        f"\n"
        f"nc: {len(class_names)}\n"
        f"names: [{names_str}]\n"
    )

    # --- 5) zip (ถ้าต้องการ) ---
    if args.zip:
        zip_path = shutil.make_archive(str(out), "zip", root_dir=out)
        print(f"สร้าง zip แล้ว: {zip_path}")

    # --- 6) สรุป ---
    print(f"\nเสร็จแล้ว -> {out}/")
    for split, items in parts.items():
        print(f"  {split:5s}: {len(items)} ภาพ")
    print(f"ไม่พบวัตถุ: {len(no_det)} | อ่านไม่ได้: {len(unreadable)}")
    if no_det and not args.save_empty:
        Path("no_detection.txt").write_text("\n".join(no_det) + "\n")
        print("รายชื่อภาพที่ไม่พบวัตถุ -> no_detection.txt (ควรตรวจ/label เอง)")


if __name__ == "__main__":
    main()
