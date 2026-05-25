"""
check_dataset.py  –  Kiểm tra và thống kê dataset
====================================================
BUGS ĐÃ SỬA so với bản gốc:
  1. Không đọc data.yaml → thêm đọc path từ data.yaml tự động
  2. Không kiểm tra label hợp lệ (class_id âm, tọa độ ngoài [0,1])
  3. Không đếm số box theo class
  4. Không phát hiện ảnh không có nhãn nào (background)
  5. Không check split 'valid' vs 'val' (lỗi hay gặp)
"""

import sys
from pathlib import Path
from collections import Counter

try:
    import yaml
except ImportError:
    print("[ERROR] Cài: pip install pyyaml")
    sys.exit(1)


def load_data_yaml(yaml_path="data.yaml"):
    """Đọc data.yaml và trả về config."""
    yaml_path = Path(yaml_path)
    if not yaml_path.exists():
        print(f"[ERROR] Không tìm thấy {yaml_path}")
        sys.exit(1)
    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def check_label_file(lbl_path: Path, nc: int):
    """Kiểm tra một file label, trả về (list_errors, n_boxes, counter_per_class)."""
    errors = []
    boxes  = []
    cls_counter = Counter()

    if not lbl_path.exists():
        return ["MISSING"], 0, cls_counter

    with open(lbl_path, "r") as f:
        lines = [l.strip() for l in f if l.strip()]

    for i, line in enumerate(lines):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"  Dòng {i+1}: sai định dạng '{line[:50]}'")
            continue
        try:
            cls_id = int(parts[0])
            cx, cy, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"  Dòng {i+1}: giá trị không phải số '{line[:50]}'")
            continue

        # Kiểm tra class_id hợp lệ
        if cls_id < 0 or cls_id >= nc:
            errors.append(f"  Dòng {i+1}: class_id={cls_id} ngoài phạm vi [0, {nc-1}]")

        # Kiểm tra tọa độ YOLO format trong [0, 1]
        for val, name in zip([cx, cy, w, h], ["cx", "cy", "w", "h"]):
            if not (0.0 <= val <= 1.0):
                errors.append(f"  Dòng {i+1}: {name}={val:.4f} ngoài [0, 1]")

        cls_counter[cls_id] += 1
        boxes.append((cls_id, cx, cy, w, h))

    return errors, len(boxes), cls_counter


def check_split(split_name: str, img_dir: Path, lbl_dir: Path, nc: int, class_names: dict):
    """Kiểm tra một split (train/valid/test)."""
    print(f"\n{'─'*55}")
    print(f"  SPLIT: {split_name.upper()}")
    print(f"{'─'*55}")

    if not img_dir.exists():
        print(f"  [ERROR] Không tìm thấy thư mục ảnh: {img_dir}")
        return

    images = sorted(
        list(img_dir.glob("*.jpg"))
        + list(img_dir.glob("*.jpeg"))
        + list(img_dir.glob("*.png"))
        + list(img_dir.glob("*.bmp"))
    )
    labels = sorted(list(lbl_dir.glob("*.txt"))) if lbl_dir.exists() else []

    print(f"  Số ảnh      : {len(images)}")
    print(f"  Số label    : {len(labels)}")

    # Ảnh thiếu label
    missing_labels = []
    label_errors   = []
    total_boxes    = 0
    total_cls      = Counter()
    background_imgs = []   # ảnh không có box nào (có thể là negative samples)

    for img in images:
        lbl_path = lbl_dir / f"{img.stem}.txt"
        errs, n_boxes, cls_cnt = check_label_file(lbl_path, nc)

        if "MISSING" in errs:
            missing_labels.append(img.name)
        else:
            if errs:
                label_errors.append((img.name, errs))
            if n_boxes == 0:
                background_imgs.append(img.name)
            total_boxes += n_boxes
            total_cls   += cls_cnt

    print(f"  Tổng số box : {total_boxes}")

    # Thống kê per-class
    print(f"\n  Phân bố box theo class:")
    for cls_id, count in sorted(total_cls.items()):
        name = class_names.get(cls_id, f"class_{cls_id}")
        pct  = count / max(total_boxes, 1) * 100
        print(f"    [{cls_id}] {name}: {count} box ({pct:.1f}%)")

    # Ảnh thiếu label
    if missing_labels:
        print(f"\n  ⚠  Ảnh thiếu file label: {len(missing_labels)}")
        for name in missing_labels[:10]:
            print(f"    - {name}")
        if len(missing_labels) > 10:
            print(f"    ... và {len(missing_labels)-10} ảnh khác")
    else:
        print(f"\n  ✅ Tất cả ảnh đều có file label")

    # Ảnh không có box (background)
    if background_imgs:
        print(f"\n  ℹ  Ảnh background (label trống): {len(background_imgs)}")
        print(f"     (Đây là negative samples – hợp lệ nếu chủ động thêm)")

    # Label bị lỗi
    if label_errors:
        print(f"\n  ❌ File label có lỗi: {len(label_errors)}")
        for fname, errs in label_errors[:5]:
            print(f"    [{fname}]")
            for e in errs[:3]:
                print(f"      {e}")
    else:
        print(f"  ✅ Tất cả label đều hợp lệ")


def main():
    cfg = load_data_yaml("data.yaml")

    base_dir    = Path(cfg.get("path", "."))
    nc          = cfg.get("nc", 1)
    class_names = cfg.get("names", {0: "fire"})

    print("═" * 55)
    print("  KIỂM TRA DATASET")
    print("═" * 55)
    print(f"  Dataset path : {base_dir.resolve()}")
    print(f"  Số class     : {nc}")
    print(f"  Classes      : {class_names}")

    # BUG FIX: kiểm tra tên split "valid" vs "val"
    val_split = cfg.get("val", "")
    if "val/" in val_split and not (base_dir / val_split).exists():
        alt = val_split.replace("val/", "valid/")
        if (base_dir / alt).exists():
            print(f"\n  ⚠  data.yaml dùng '{val_split}' nhưng thư mục thực tế là '{alt}'")
            print(f"     → Sửa data.yaml: val: {alt}")

    # Kiểm tra từng split
    for split_key, split_name in [("train", "train"), ("val", "valid"), ("test", "test")]:
        split_rel = cfg.get(split_key, f"{split_name}/images")
        img_dir   = base_dir / split_rel
        lbl_dir   = img_dir.parent.parent / split_rel.replace("images", "labels")
        # Thử cả "valid" nếu "val" không tồn tại
        if not img_dir.exists() and split_key == "val":
            img_dir = base_dir / "valid" / "images"
            lbl_dir = base_dir / "valid" / "labels"

        check_split(split_name, img_dir, lbl_dir, nc, class_names)

    print(f"\n{'═'*55}")
    print("  KIỂM TRA HOÀN TẤT")
    print("═" * 55)
    print("\n[NEXT] Nếu mọi thứ OK, chạy train.py để bắt đầu train.")


if __name__ == "__main__":
    main()
