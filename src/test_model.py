"""
test_model.py – Đánh giá model trên tập test/val
Cách dùng:
  python src/test_model.py              # menu chọn
  python src/test_model.py --best       # tự chọn run tốt nhất
  python src/test_model.py --latest     # run mới nhất
  python src/test_model.py --run train_fire_003
  python src/test_model.py --model path/to/best.pt
"""
import os, sys, csv, argparse, torch, yaml
from pathlib import Path
from datetime import datetime

try:
    from ultralytics import YOLO
except ImportError:
    sys.exit("[ERROR] pip install ultralytics")

ROOT       = Path(__file__).resolve().parent.parent
DATA_YAML  = ROOT / "data.yaml"
TRAIN_DIR  = ROOT / "runs" / "train"
VAL_DIR    = ROOT / "runs" / "val"


def get_device():
    if torch.cuda.is_available():   return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available(): return "mps"
    return "cpu"


# ── Đọc metrics tốt nhất từ results.csv ──────────────────────────
def read_score(folder):
    csv_path = Path(folder) / "results.csv"
    best = {"score": -1, "map50": None, "map5095": None,
            "precision": None, "recall": None, "epoch": None}
    if not csv_path.exists():
        return best

    def fget(row, *keys):
        for k in keys:
            if k in row and row[k] != "":
                try: return float(row[k])
                except ValueError: pass
        return None

    try:
        with open(csv_path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                row = {k.strip(): v for k, v in row.items()}
                m50   = fget(row, "metrics/mAP50(B)", "metrics/mAP50", "mAP50")
                m5095 = fget(row, "metrics/mAP50-95(B)", "metrics/mAP50-95", "mAP50-95")
                score = m5095 if m5095 is not None else m50
                if score and score > best["score"]:
                    best = {"score": score, "map50": m50, "map5095": m5095,
                            "precision": fget(row, "metrics/precision(B)", "metrics/precision"),
                            "recall":    fget(row, "metrics/recall(B)",    "metrics/recall"),
                            "epoch":     int(fget(row, "epoch") or 0)}
    except Exception:
        pass
    return best


# ── Đọc args.yaml ─────────────────────────────────────────────────
def read_args(folder):
    p = Path(folder) / "args.yaml"
    if not p.exists(): return {}
    try:
        with open(p, encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


# ── Tìm tất cả run ───────────────────────────────────────────────
def find_runs():
    if not TRAIN_DIR.exists(): return []
    runs = []
    for folder in TRAIN_DIR.iterdir():
        bp = folder / "weights" / "best.pt"
        if not folder.is_dir() or not bp.exists(): continue
        sc = read_score(folder)
        runs.append({"name": folder.name, "folder": folder, "best_pt": bp,
                     "mtime": bp.stat().st_mtime, **sc, "best_epoch": sc["epoch"]})
    return runs


def best_run(runs):
    valid = [r for r in runs if r["score"] >= 0]
    return sorted(valid, key=lambda x: x["score"], reverse=True)[0] if valid \
           else sorted(runs, key=lambda x: x["mtime"], reverse=True)[0]


def latest_run(runs):
    return sorted(runs, key=lambda x: x["mtime"], reverse=True)[0]


def menu(runs):
    if not runs:
        sys.exit(f"[ERROR] Không tìm thấy model trong {TRAIN_DIR}")
    runs_s = sorted(runs, key=lambda x: x["name"])
    print("\nCác lần train:")
    for i, r in enumerate(runs_s, 1):
        m50 = f"{r['map50']:.3f}" if r["map50"] else "N/A"
        print(f"  {i}. {r['name']}  mAP50={m50}  epoch={r['best_epoch'] or 'N/A'}")
    print("\n1=best  2=chọn  3=latest  Enter=1")
    c = input(">>> ").strip()
    if c in ("", "1"): return best_run(runs)
    if c == "3":       return latest_run(runs)
    if c == "2":
        while True:
            idx = input(f"Số thứ tự [1-{len(runs_s)}]: ").strip()
            if idx.isdigit() and 1 <= int(idx) <= len(runs_s):
                return runs_s[int(idx) - 1]
    return best_run(runs)


def get_run_by_name(name):
    bp = TRAIN_DIR / name / "weights" / "best.pt"
    if not bp.exists(): sys.exit(f"[ERROR] Không tìm thấy {bp}")
    folder = TRAIN_DIR / name
    sc = read_score(folder)
    return {"name": name, "folder": folder, "best_pt": bp,
            "mtime": bp.stat().st_mtime, **sc, "best_epoch": sc["epoch"]}


def get_run_by_path(path):
    p = Path(path)
    if not p.is_absolute(): p = ROOT / p
    if not p.exists(): sys.exit(f"[ERROR] Không tìm thấy {p}")
    folder = p.parent.parent
    sc = read_score(folder)
    return {"name": folder.name, "folder": folder, "best_pt": p,
            "mtime": p.stat().st_mtime, **sc, "best_epoch": sc["epoch"]}


# ── In thông số run ───────────────────────────────────────────────
def fmt(val): return f"{val:.4f}" if val else "N/A"

def print_run_info(run):
    print(f"\n{'─'*60}")
    print(f"Run      : {run['name']}")
    print(f"Model    : {run['best_pt']}")
    print(f"Epoch    : {run['best_epoch'] or 'N/A'}")
    print(f"mAP50    : {fmt(run['map50'])}")
    print(f"mAP50-95 : {fmt(run['map5095'])}")
    print(f"Precision: {fmt(run['precision'])}")
    print(f"Recall   : {fmt(run['recall'])}")

    args = read_args(run["folder"])
    if args:
        SHOW = ["model","epochs","batch","imgsz","device","optimizer",
                "lr0","lrf","momentum","weight_decay","patience","cos_lr",
                "hsv_h","hsv_s","hsv_v","mosaic","mixup","augment"]
        print("─── args.yaml ───")
        for k in SHOW:
            if k in args: print(f"  {k:<16}: {args[k]}")
    print(f"{'─'*60}")


# ── Main ──────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--best",   action="store_true")
    p.add_argument("--latest", action="store_true")
    p.add_argument("--run",    type=str, default=None)
    p.add_argument("--model",  type=str, default=None)
    p.add_argument("--split",  default="test", choices=["train","val","test"])
    p.add_argument("--conf",   type=float, default=0.25)
    p.add_argument("--iou",    type=float, default=0.5)
    args = p.parse_args()

    if not DATA_YAML.exists():
        sys.exit(f"[ERROR] Không tìm thấy {DATA_YAML}")

    runs = find_runs()
    if args.model:       run = get_run_by_path(args.model)
    elif args.run:       run = get_run_by_name(args.run)
    elif args.best:      run = best_run(runs)
    elif args.latest:    run = latest_run(runs)
    else:                run = menu(runs)

    print_run_info(run)

    device = get_device()
    model  = YOLO(str(run["best_pt"]))
    ts     = datetime.now().strftime("%Y%m%d_%H%M%S")
    name   = f"{run['name']}_{args.split}_{ts}"

    print(f"[INFO] Đánh giá split={args.split} conf={args.conf} iou={args.iou} device={device}")
    metrics = model.val(
        data=str(DATA_YAML), split=args.split, imgsz=640,
        conf=args.conf, iou=args.iou, device=device,
        workers=0 if os.name == "nt" else 4,
        plots=True, save_json=False, verbose=False,
        project=str(VAL_DIR), name=name, exist_ok=False,
    )

    map50 = metrics.box.map50
    map95 = metrics.box.map
    prec  = metrics.box.mp
    rec   = metrics.box.mr
    f1    = 2 * prec * rec / (prec + rec + 1e-9)

    print(f"\n{'─'*60}")
    print(f"mAP@0.5      : {map50:.4f}  ({map50*100:.1f}%)")
    print(f"mAP@0.5:0.95 : {map95:.4f}  ({map95*100:.1f}%)")
    print(f"Precision    : {prec:.4f}  ({prec*100:.1f}%)")
    print(f"Recall       : {rec:.4f}  ({rec*100:.1f}%)")
    print(f"F1           : {f1:.4f}  ({f1*100:.1f}%)")

    grade = ("✅ Tốt" if map50>=0.8 else "🟡 Khá" if map50>=0.6
             else "🟠 Trung bình" if map50>=0.4 else "🔴 Yếu")
    print(f"Đánh giá     : {grade}")

    for idx, (_, cname) in enumerate(model.names.items()):
        if idx < len(metrics.box.maps):
            print(f"  [{cname}] mAP50-95={metrics.box.maps[idx]:.4f}")

    # Lưu CSV
    VAL_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = VAL_DIR / f"summary_{name}.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows([
            ["run_name", run["name"]], ["model", str(run["best_pt"])],
            ["split", args.split], ["timestamp", ts], ["device", device],
            ["conf", args.conf], ["iou", args.iou],
            ["train_best_epoch", run["best_epoch"]],
            ["train_map50", run["map50"]], ["train_map50_95", run["map5095"]],
            ["test_mAP50", f"{map50:.6f}"], ["test_mAP50-95", f"{map95:.6f}"],
            ["test_precision", f"{prec:.6f}"], ["test_recall", f"{rec:.6f}"],
            ["test_f1", f"{f1:.6f}"],
        ])

    print(f"\n[DONE] {metrics.save_dir}")
    print(f"[CSV]  {csv_path}")


if __name__ == "__main__":
    main()