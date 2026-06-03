import sys
from pathlib import Path
from collections import Counter

try:
    import yaml
except ImportError:
    sys.exit("[ERROR] Cài thư viện: pip install pyyaml")


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp"}
SPLITS = [("train", "train"), ("val", "valid"), ("test", "test")]


def load_yaml(file="data.yaml"):
    file = Path(file)

    if not file.exists():
        sys.exit(f"[ERROR] Không tìm thấy {file}")

    with open(file, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    base = Path(cfg.get("path", "."))
    if not base.is_absolute():
        base = file.parent / base

    names = cfg.get("names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))

    nc = int(cfg.get("nc", len(names)))
    return cfg, base, nc, names


def split_dirs(cfg, base, key):
    default = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images"
    }

    img_dir = base / cfg.get(key, default[key])

    if key == "val" and not img_dir.exists():
        img_dir = base / "val" / "images"

    lbl_dir = img_dir.parent / "labels"
    return img_dir, lbl_dir


def get_images(img_dir):
    if not img_dir.exists():
        return []

    return sorted(
        file for file in img_dir.iterdir()
        if file.suffix.lower() in IMG_EXTS
    )


def check_label(label_path, nc):
    if not label_path.exists():
        return "missing", [], 0, Counter()

    errors = []
    box_count = 0
    class_count = Counter()
    has_content = False

    with open(label_path, "r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()

            if not line:
                continue

            has_content = True
            parts = line.split()

            if len(parts) != 5:
                errors.append(f"Dòng {line_no}: sai định dạng -> {line}")
                continue

            try:
                cls_id = int(parts[0])
                cx, cy, w, h = map(float, parts[1:])
            except ValueError:
                errors.append(f"Dòng {line_no}: giá trị không phải số -> {line}")
                continue

            if not 0 <= cls_id < nc:
                errors.append(f"Dòng {line_no}: class_id={cls_id} ngoài [0, {nc - 1}]")
            else:
                class_count[cls_id] += 1

            for name, value in zip(("cx", "cy", "w", "h"), (cx, cy, w, h)):
                if not 0 <= value <= 1:
                    errors.append(f"Dòng {line_no}: {name}={value:.4f} ngoài [0, 1]")

            box_count += 1

    status = "ok" if has_content else "empty"
    return status, errors, box_count, class_count


def print_items(title, items):
    print(f"\n  {title}: {len(items)}")

    if not items:
        print("    Không có")
        return

    for item in items:
        print(f"    - {item}")


def print_errors(error_files):
    print(f"\n  File label có lỗi: {len(error_files)}")

    if not error_files:
        print("    Không có")
        return

    for file_name, errors in error_files:
        print(f"\n    [{file_name}]")
        for error in errors:
            print(f"      - {error}")


def print_class_count(class_count, total_boxes, names):
    print("\n  Phân bố box theo class:")

    if total_boxes == 0:
        print("    Không có box")
        return

    for cls_id, count in sorted(class_count.items()):
        name = names.get(cls_id, f"class_{cls_id}")
        percent = count / total_boxes * 100
        print(f"    [{cls_id}] {name}: {count} box ({percent:.1f}%)")


def check_split(split_name, img_dir, lbl_dir, nc, names):
    print(f"\n{'─' * 55}")
    print(f"  SPLIT: {split_name.upper()}")
    print(f"{'─' * 55}")

    if not img_dir.exists():
        print(f"  [SKIP] Không tìm thấy thư mục ảnh: {img_dir}")
        return

    images = get_images(img_dir)
    label_count = sum(1 for _ in lbl_dir.glob("*.txt")) if lbl_dir.exists() else 0

    missing = []
    empty = []
    error_files = []

    total_boxes = 0
    total_classes = Counter()

    for img in images:
        label_path = lbl_dir / f"{img.stem}.txt"
        status, errors, box_count, class_count = check_label(label_path, nc)

        if status == "missing":
            missing.append(img.name)
            continue

        if status == "empty":
            empty.append(img.name)

        if errors:
            error_files.append((img.name, errors))

        total_boxes += box_count
        total_classes += class_count

    print(f"  Số ảnh       : {len(images)}")
    print(f"  Số file label: {label_count}")
    print(f"  Tổng số box  : {total_boxes}")

    print_class_count(total_classes, total_boxes, names)
    print_items("Ảnh thiếu label", missing)
    print_items("Ảnh label rỗng / background", empty)
    print_errors(error_files)


def main():
    cfg, base, nc, names = load_yaml()

    print("═" * 55)
    print("  KIỂM TRA DATASET YOLO")
    print("═" * 55)
    print(f"  Dataset path : {base.resolve()}")
    print(f"  Số class     : {nc}")
    print(f"  Classes      : {names}")

    for key, name in SPLITS:
        img_dir, lbl_dir = split_dirs(cfg, base, key)
        check_split(name, img_dir, lbl_dir, nc, names)

    print(f"\n{'═' * 55}")
    print("  KIỂM TRA HOÀN TẤT")
    print("═" * 55)


if __name__ == "__main__":
    main()
