import os, re, sys, torch
from pathlib import Path
from ultralytics import YOLO

ROOT         = Path(__file__).resolve().parent.parent
TRAIN_DIR    = ROOT / "runs" / "train"
TRAIN_PREFIX = "train_fire"
EPOCHS, IMGSZ, BATCH, MODEL = 10, 640, 8, "yolo11n.pt"


def get_device():
    if torch.cuda.is_available():   return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    print("[WARNING] Không có GPU – dùng CPU sẽ rất chậm!")
    return "cpu"


def next_run_name():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    pat = re.compile(rf"^{TRAIN_PREFIX}_(\d+)$")
    nums = [int(m.group(1)) for f in TRAIN_DIR.iterdir()
            if f.is_dir() and (m := pat.match(f.name))]
    return f"{TRAIN_PREFIX}_{(max(nums, default=0) + 1):03d}"


def main():
    data_yaml = ROOT / "data.yaml"
    if not data_yaml.exists():
        sys.exit(f"[ERROR] Không tìm thấy {data_yaml}")

    device    = get_device()
    run_name  = next_run_name()
    model     = YOLO(MODEL)

    print(f"[TRAIN] {run_name} | epochs={EPOCHS} batch={BATCH} imgsz={IMGSZ} device={device}")

    model.train(
        data=str(data_yaml), epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH,
        device=device, workers=0 if os.name == "nt" else 4,
        project=str(TRAIN_DIR), name=run_name, exist_ok=False,
        patience=20, cos_lr=True,
        hsv_h=0.015, hsv_s=0.7, hsv_v=0.4, mosaic=1.0, mixup=0.1,
        save=True, save_period=-1, plots=True,
    )

    best = TRAIN_DIR / run_name / "weights" / "best.pt"
    print(f"[DONE] {run_name} → {best}")


if __name__ == "__main__":
    main()
