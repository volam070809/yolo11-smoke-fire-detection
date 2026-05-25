"""
predict_test.py – Predict ảnh / video / webcam
Kết quả → runs/predict/

Cách dùng:
  python src/predict_test.py                        # ảnh test mặc định
  python src/predict_test.py --source video.mp4
  python src/predict_test.py --source 0             # webcam
"""
import sys, os, argparse
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
    import cv2
except ImportError as e:
    sys.exit(f"[ERROR] {e} – pip install ultralytics opencv-python")

ROOT        = Path(__file__).resolve().parent.parent
TRAIN_DIR   = ROOT / "runs" / "train"
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
                            for k in ("metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95"):
                                if k in row and row[k] != "":
                                    try: score = max(score, float(row[k]))
                                    except ValueError: pass
                except Exception:
                    pass
            # fallback: dùng mtime nếu không đọc được CSV
            if score == -1:
                score = pt.stat().st_mtime * 1e-12
            if score > best_score:
                best_score, best_pt = score, pt
        if best_pt:
            run_name = best_pt.parent.parent.name
            print(f"[INFO] Dùng model tốt nhất: {run_name}  (mAP50-95={best_score:.4f})")
            return str(best_pt)
    if Path("yolo11n.pt").exists(): return "yolo11n.pt"
    sys.exit("[ERROR] Không tìm thấy model. Chạy train.py trước!")


def alert(names):
    if any("fire" in n.lower() or "smoke" in n.lower() for n in names):
        print(f"  🔥 Phát hiện: {', '.join(set(names))}")
        try:
            print("\a", end="", flush=True) if os.name != "nt" else None
        except Exception: pass


def predict_images(model, source, conf):
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PREDICT_DIR / f"images_{ts}"
    results = model.predict(source=source, save=True, imgsz=640, conf=conf,
                             iou=0.45, project=str(out_dir.parent),
                             name=out_dir.name, verbose=False)
    total = sum(len(r.boxes) if r.boxes else 0 for r in results)
    for r in results:
        if r.boxes: alert([r.names[int(c)] for c in r.boxes.cls])
    print(f"[DONE] {total} detections / {len(results)} ảnh → {out_dir}")


def predict_video(model, source, conf, show=True):
    is_cam = str(source) == "0"
    cap    = cv2.VideoCapture(int(source) if is_cam else source)
    if not cap.isOpened(): sys.exit(f"[ERROR] Không mở được: {source}")

    fps, w, h = (cap.get(cv2.CAP_PROP_FPS) or 30,
                 int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                 int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = PREDICT_DIR / f"video_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer  = cv2.VideoWriter(str(out_dir / "output.avi"),
                               cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))
    frames, detects, last_alert = 0, 0, -999

    try:
        while True:
            ret, frame = cap.read()
            if not ret: break
            frames += 1
            res  = model.predict(source=frame, imgsz=640, conf=conf,
                                  iou=0.45, verbose=False)
            ann  = res[0].plot()
            if res[0].boxes and len(res[0].boxes):
                detects += 1
                names = [res[0].names[int(c)] for c in res[0].boxes.cls]
                if frames - last_alert > 30:
                    alert(names); last_alert = frames
                cv2.putText(ann, "FIRE/SMOKE DETECTED", (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
            writer.write(ann)
            if show:
                cv2.imshow("Detector", ann)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"): break
                if k == ord("s"):
                    snap = out_dir / f"snap_{frames}.jpg"
                    cv2.imwrite(str(snap), ann)
                    print(f"  Snap → {snap}")
    finally:
        cap.release(); writer.release()
        if show: cv2.destroyAllWindows()
    print(f"[DONE] {frames} frames, {detects} detections → {out_dir}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source", default="datasets/fire_data/test/images")
    p.add_argument("--conf",   type=float, default=0.4)
    p.add_argument("--no-show", action="store_true")
    args = p.parse_args()

    model  = YOLO(get_model())
    src    = args.source
    src_p  = Path(src)
    is_vid = src_p.exists() and src_p.suffix.lower() in {".mp4",".avi",".mov",".mkv",".webm"}
    is_cam = src == "0"
    is_img = src_p.exists() and (src_p.is_dir() or src_p.suffix.lower() in {".jpg",".jpeg",".png",".bmp"})

    if is_img:        predict_images(model, src, args.conf)
    elif is_vid or is_cam: predict_video(model, src, args.conf, show=not args.no_show)
    else:             predict_images(model, src, args.conf)


if __name__ == "__main__":
    main()