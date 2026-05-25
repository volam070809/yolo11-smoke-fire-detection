"""
detect_realtime.py – Real-time fire/smoke detection
Kết quả → runs/realtime/<timestamp>/

Cách dùng:
  python src/detect_realtime.py
  python src/detect_realtime.py --source video.mp4
  python src/detect_realtime.py --source rtsp://...
  Phím: q=thoát  s=chụp thủ công  c=tăng conf
"""
import sys, os, time, argparse, threading
from pathlib import Path
from datetime import datetime
from collections import deque

try:
    from ultralytics import YOLO
    import cv2
except ImportError as e:
    sys.exit(f"[ERROR] {e} – pip install ultralytics opencv-python")

ROOT         = Path(__file__).resolve().parent.parent
TRAIN_DIR    = ROOT / "runs" / "train"
REALTIME_DIR = ROOT / "runs" / "realtime"

CONF_DEFAULT     = 0.45
IOU_DEFAULT      = 0.45
ALERT_COOLDOWN_S = 3


def get_model():
    if TRAIN_DIR.exists():
        pts = sorted(TRAIN_DIR.glob("*/weights/best.pt"),
                     key=lambda p: p.stat().st_mtime, reverse=True)
        if pts: return str(pts[0])
    if Path("yolo11n.pt").exists(): return "yolo11n.pt"
    sys.exit("[ERROR] Không tìm thấy model. Chạy train.py trước!")


def beep():
    def _b():
        try:
            if os.name == "nt":
                import winsound
                [winsound.Beep(1000, 200) or time.sleep(0.1) for _ in range(3)]
            else:
                [print("\a", end="", flush=True) or time.sleep(0.1) for _ in range(3)]
        except Exception: pass
    threading.Thread(target=_b, daemon=True).start()


class Detector:
    def __init__(self, model_path, conf, iou, source, save_snaps):
        self.model   = YOLO(model_path)
        self.conf    = conf
        self.iou     = iou
        self.source  = source
        self.snaps   = save_snaps
        ts           = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.out     = REALTIME_DIR / ts
        self.out.mkdir(parents=True, exist_ok=True)
        self.log_f   = open(self.out / "events.log", "a", encoding="utf-8")
        self.fps_buf = deque(maxlen=20)
        self.last_alr = 0
        self.nframes  = 0
        self.ndetect  = 0
        self.nsnaps   = 0

    def log(self, msg):
        line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
        print(line)
        self.log_f.write(line + "\n")
        self.log_f.flush()

    def process(self, frame):
        t0  = time.perf_counter()
        res = self.model.predict(source=frame, imgsz=640, conf=self.conf,
                                  iou=self.iou, verbose=False)
        self.fps_buf.append(1.0 / max(time.perf_counter() - t0, 1e-6))
        fps = sum(self.fps_buf) / len(self.fps_buf)

        self.nframes += 1
        r      = res[0]
        alert  = False
        names  = []
        do_snap = False

        if r.boxes and len(r.boxes):
            self.ndetect += 1
            names = [r.names[int(c)] for c in r.boxes.cls]
            confs = [float(c) for c in r.boxes.conf]
            if any("fire" in n.lower() or "smoke" in n.lower() for n in names):
                alert = True
                now   = time.time()
                # Beep + log: cooldown 3s
                if now - self.last_alr >= ALERT_COOLDOWN_S:
                    self.last_alr = now
                    self.log(f"DETECTED {', '.join(f'{n}({c:.2f})' for n,c in zip(names,confs))}")
                    beep()
                # Đánh dấu cần chụp (sẽ chụp sau khi ann được vẽ xong)
                if self.snaps:
                    do_snap = True

        # Vẽ bounding box
        ann = r.plot(line_width=2, labels=True, conf=True)
        h, w = ann.shape[:2]

        # Vẽ FPS + thời gian
        cv2.putText(ann, f"FPS:{fps:.1f}", (10, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(ann, datetime.now().strftime("%H:%M:%S"), (10, 54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1)

        # Vẽ overlay cảnh báo
        if alert:
            ov = ann.copy()
            cv2.rectangle(ov, (0, 0), (w, 70), (0, 0, 180), -1)
            cv2.addWeighted(ov, 0.3, ann, 0.7, 0, ann)
            cv2.putText(ann, "CANH BAO CHAY!", (10, h - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)

        # Chụp SAU khi đã vẽ đầy đủ overlay + cảnh báo
        if do_snap:
            cv2.imwrite(str(self.out / f"snap_{self.nsnaps:04d}.jpg"), ann)
            self.nsnaps += 1

        return ann

    def run(self):
        is_cam = str(self.source) == "0"
        label  = "Webcam" if is_cam else str(self.source)
        cap    = cv2.VideoCapture(int(self.source) if is_cam else self.source)
        if not cap.isOpened():
            sys.exit(f"[ERROR] Không mở được: {self.source}")

        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        out_vid = self.out / "output.avi"
        writer  = cv2.VideoWriter(str(out_vid), cv2.VideoWriter_fourcc(*"XVID"), fps, (w, h))

        self.log(f"START source={label} conf={self.conf} iou={self.iou}")
        print(f"[INFO] Output → {self.out}  |  q=thoát  s=chụp  c=tăng conf")

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    self.log("END")
                    break
                ann = self.process(frame)
                writer.write(ann)
                cv2.imshow(f"Detector – {label}", ann)
                k = cv2.waitKey(1) & 0xFF
                if k == ord("q"):
                    self.log("QUIT")
                    break
                if k == ord("s"):
                    p = self.out / f"manual_{self.nsnaps:04d}.jpg"
                    cv2.imwrite(str(p), ann)
                    self.nsnaps += 1
                    print(f"  Snap → {p}")
                if k == ord("c"):
                    self.conf = min(0.95, self.conf + 0.05)
                    print(f"  conf → {self.conf:.2f}")
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()
            self.log_f.close()
            rate = self.ndetect / max(self.nframes, 1) * 100
            print(f"[DONE] {self.nframes} frames | {self.ndetect} detected ({rate:.1f}%) | {self.nsnaps} snaps")
            print(f"       video → {out_vid}")
            print(f"       log   → {self.out / 'events.log'}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source",  default="0")
    p.add_argument("--conf",    type=float, default=CONF_DEFAULT)
    p.add_argument("--iou",     type=float, default=IOU_DEFAULT)
    p.add_argument("--model",   default=None)
    p.add_argument("--no-snap", action="store_true")
    args = p.parse_args()
    Detector(args.model or get_model(), args.conf, args.iou,
             args.source, not args.no_snap).run()


if __name__ == "__main__":
    main()