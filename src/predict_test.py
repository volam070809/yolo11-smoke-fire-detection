"""
predict_test.py – Predict ảnh / video / webcam
Kết quả lưu tại: runs/predict/

Cách dùng:
  Chỉnh SOURCE trong phần CONFIG bên dưới
  Sau đó chạy:
    python src/predict_test.py
"""

import sys
import time
from pathlib import Path
from datetime import datetime
from collections import Counter

try:
    from ultralytics import YOLO
except ImportError as e:
    sys.exit(f"[ERROR] {e} – pip install ultralytics")


# ============================================================
# CONFIG - CHỈNH THỦ CÔNG Ở ĐÂY
# ============================================================

# 1. Predict thư mục ảnh test mặc định
SOURCE = "datasets/fire_data/test/images"

# 2. Predict 1 ảnh cụ thể
# SOURCE = "test.jpg"

# 3. Predict video
# SOURCE = "video.mp4"

# 4. Predict webcam
# SOURCE = "0"

CONF = 0.4
IMGSZ = 640
IOU = 0.45

SHOW = False        # True nếu muốn hiện cửa sổ predict
SAVE_TXT = True    # True nếu muốn lưu file label .txt
SAVE_CONF = True   # True nếu muốn lưu confidence vào file .txt


# ============================================================
# PATH
# ============================================================

ROOT = Path(__file__).resolve().parent.parent
TRAIN_DIR = ROOT / "runs" / "train"
PREDICT_DIR = ROOT / "runs" / "predict"


def get_model():
    if TRAIN_DIR.exists():
        best_pt, best_score = None, -1

        for pt in TRAIN_DIR.glob("*/weights/best.pt"):
            csv_path = pt.parent.parent / "results.csv"
            score = -1

            if csv_path.exists():
                try:
                    import csv as _csv

                    with open(csv_path, encoding="utf-8") as f:
                        for row in _csv.DictReader(f):
                            row = {k.strip(): v for k, v in row.items()}

                            for k in (
                                "metrics/mAP50-95(B)",
                                "metrics/mAP50-95",
                                "mAP50-95",
                            ):
                                if k in row and row[k] != "":
                                    try:
                                        score = max(score, float(row[k]))
                                    except ValueError:
                                        pass

                except Exception:
                    pass

            if score == -1:
                score = pt.stat().st_mtime * 1e-12

            if score > best_score:
                best_score, best_pt = score, pt

        if best_pt:
            run_name = best_pt.parent.parent.name
            print(f"[INFO] Model run : {run_name}")
            print(f"[INFO] Weight    : {best_pt}")
            print(f"[INFO] mAP50-95 : {best_score:.4f}")
            return str(best_pt)

    default_model = ROOT / "yolo11n.pt"

    if default_model.exists():
        print(f"[INFO] Dùng model mặc định: {default_model}")
        return str(default_model)

    if Path("yolo11n.pt").exists():
        print("[INFO] Dùng model mặc định: yolo11n.pt")
        return "yolo11n.pt"

    sys.exit("[ERROR] Không tìm thấy model. Chạy train.py trước!")


def format_counter(counter):
    if not counter:
        return "Không có object"

    return ", ".join(
        f"{name}: {count}"
        for name, count in counter.items()
    )


def predict_source(model):
    start_time = time.time()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"predict_{ts}"
    save_dir = PREDICT_DIR / run_name

    source = int(SOURCE) if SOURCE == "0" else SOURCE

    print("=" * 60)
    print("YOLO PREDICTION")
    print("=" * 60)
    print(f"[INFO] Source   : {SOURCE}")
    print(f"[INFO] Conf     : {CONF}")
    print(f"[INFO] Img size : {IMGSZ}")
    print(f"[INFO] IOU      : {IOU}")
    print(f"[INFO] Save txt : {SAVE_TXT}")
    print(f"[INFO] Save conf: {SAVE_CONF}")
    print(f"[INFO] Show     : {SHOW}")

    results = model.predict(
        source=source,
        conf=CONF,
        imgsz=IMGSZ,
        iou=IOU,
        save=True,
        show=SHOW,
        save_txt=SAVE_TXT,
        save_conf=SAVE_CONF,
        show_labels=True,
        show_conf=True,
        show_boxes=True,
        project=str(PREDICT_DIR),
        name=run_name,
        verbose=False,
        stream=True
    )

    total_items = 0
    total_boxes = 0
    class_counter = Counter()

    for r in results:
        total_items += 1

        if r.boxes is not None and len(r.boxes) > 0:
            total_boxes += len(r.boxes)

            for cls_id in r.boxes.cls:
                class_name = r.names[int(cls_id)]
                class_counter[class_name] += 1

    elapsed = time.time() - start_time

    print("=" * 60)
    print("PREDICTION SUMMARY")
    print("=" * 60)
    print(f"[DONE] Số ảnh/frame xử lý : {total_items}")
    print(f"[DONE] Tổng object detect : {total_boxes}")
    print(f"[DONE] Theo từng class    : {format_counter(class_counter)}")
    print(f"[DONE] Thời gian xử lý    : {elapsed:.2f}s")

    if elapsed > 0:
        print(f"[DONE] Tốc độ trung bình  : {total_items / elapsed:.2f} item/s")
    else:
        print("[DONE] Tốc độ trung bình  : N/A")

    print(f"[DONE] Kết quả lưu tại    : {save_dir}")


def main():
    model = YOLO(get_model())
    predict_source(model)


if __name__ == "__main__":
    main()
