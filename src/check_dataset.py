import sys
from pathlib import Path
from collections import Counter, defaultdict
from statistics import mean

try:
    import yaml
except ImportError:
    sys.exit("[ERROR] Cài thư viện: pip install pyyaml")

try:
    from PIL import Image
except ImportError:
    Image = None


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
SPLITS = [("train", "train"), ("val", "valid"), ("test", "test")]


def load_yaml(file_name="data.yaml"):
    yaml_path = Path(file_name)

    if not yaml_path.exists():
        sys.exit("[ERROR] Không tìm thấy data.yaml. Hãy chạy file ở thư mục gốc project.")

    with open(yaml_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    root = Path(cfg.get("path", "."))
    if not root.is_absolute():
        root = yaml_path.parent / root

    names = cfg.get("names", {})
    if isinstance(names, list):
        names = dict(enumerate(names))

    names = {int(k): str(v) for k, v in names.items()}
    nc = int(cfg.get("nc", len(names)))

    return cfg, root, nc, names


def get_split_dirs(cfg, root, key):
    defaults = {
        "train": "train/images",
        "val": "valid/images",
        "test": "test/images",
    }

    raw_path = cfg.get(key)

    if key == "val" and raw_path is None:
        raw_path = cfg.get("valid")

    if raw_path is None:
        raw_path = defaults[key]

    img_dir = root / raw_path

    if key == "val" and not img_dir.exists():
        alt_dir = root / "val" / "images"
        if alt_dir.exists():
            img_dir = alt_dir

    label_dir = img_dir.parent / "labels"
    return img_dir, label_dir


def get_files(folder, exts):
    if not folder.exists():
        return []

    return sorted(
        file for file in folder.rglob("*")
        if file.is_file() and file.suffix.lower() in exts
    )


def get_label_path(image_path, image_dir, label_dir):
    relative_path = image_path.relative_to(image_dir)
    return label_dir / relative_path.with_suffix(".txt")


def read_image_size(image_path):
    if Image is None:
        return None, None

    try:
        with Image.open(image_path) as img:
            return img.size, None
    except Exception as e:
        return None, str(e)


def bbox_size_type(w, h):
    area = w * h

    if area < 0.02:
        return "small"
    if area < 0.15:
        return "medium"
    return "large"


def check_label(label_path, nc):
    if not label_path.exists():
        return "missing", [], Counter(), set(), Counter()

    lines = label_path.read_text(encoding="utf-8").splitlines()

    if not any(line.strip() for line in lines):
        return "empty", [], Counter(), set(), Counter()

    errors = []
    class_count = Counter()
    classes_in_image = set()
    bbox_sizes = Counter()

    for line_no, line in enumerate(lines, 1):
        line = line.strip()

        if not line:
            continue

        parts = line.split()

        if len(parts) != 5:
            errors.append(f"Dòng {line_no}: sai 5 cột -> {line}")
            continue

        try:
            cls_id = int(parts[0])
            x, y, w, h = map(float, parts[1:])
        except ValueError:
            errors.append(f"Dòng {line_no}: giá trị không phải số -> {line}")
            continue

        if not 0 <= cls_id < nc:
            errors.append(f"Dòng {line_no}: class_id={cls_id} không hợp lệ")
            continue

        if not 0 <= x <= 1 or not 0 <= y <= 1:
            errors.append(f"Dòng {line_no}: x/y ngoài khoảng 0..1")

        if not 0 < w <= 1 or not 0 < h <= 1:
            errors.append(f"Dòng {line_no}: w/h phải > 0 và <= 1")

        if x - w / 2 < 0 or x + w / 2 > 1 or y - h / 2 < 0 or y + h / 2 > 1:
            errors.append(f"Dòng {line_no}: bbox vượt biên ảnh")

        class_count[cls_id] += 1
        classes_in_image.add(cls_id)
        bbox_sizes[(cls_id, bbox_size_type(w, h))] += 1

    return "ok", errors, class_count, classes_in_image, bbox_sizes


def find_class_ids(names, keyword):
    return {
        cls_id for cls_id, name in names.items()
        if keyword.lower() in name.lower()
    }


def print_class_count(class_count, names):
    total = sum(class_count.values())

    print("\n  Số box theo class:")

    if total == 0:
        print("    Không có box")
        return

    for cls_id in sorted(names):
        count = class_count.get(cls_id, 0)
        percent = count / total * 100
        print(f"    [{cls_id}] {names[cls_id]}: {count} box ({percent:.1f}%)")


def print_image_size_stats(sizes):
    print("\n  Kích thước ảnh:")

    if not sizes:
        print("    Không đọc được kích thước ảnh")
        return

    min_size = min(sizes, key=lambda s: s[0] * s[1])
    max_size = max(sizes, key=lambda s: s[0] * s[1])
    avg_w = mean(w for w, h in sizes)
    avg_h = mean(h for w, h in sizes)
    common = Counter(sizes).most_common(3)

    print(f"    Nhỏ nhất          : {min_size[0]}x{min_size[1]}")
    print(f"    Lớn nhất          : {max_size[0]}x{max_size[1]}")
    print(f"    Trung bình        : {avg_w:.0f}x{avg_h:.0f}")
    print("    Phổ biến          : " + ", ".join(
        f"{w}x{h} ({count})" for (w, h), count in common
    ))


def print_bbox_size_stats(bbox_sizes, names):
    print("\n  Kích thước object:")

    for cls_id in sorted(names):
        small = bbox_sizes.get((cls_id, "small"), 0)
        medium = bbox_sizes.get((cls_id, "medium"), 0)
        large = bbox_sizes.get((cls_id, "large"), 0)

        print(
            f"    [{cls_id}] {names[cls_id]}: "
            f"small={small}, medium={medium}, large={large}"
        )


def check_split(split_name, image_dir, label_dir, nc, names, fire_ids, smoke_ids):
    print(f"\n{'─' * 60}")
    print(f"  SPLIT: {split_name.upper()}")
    print(f"{'─' * 60}")
    print(f"  Images: {image_dir}")
    print(f"  Labels: {label_dir}")

    images = get_files(image_dir, IMG_EXTS)
    labels = get_files(label_dir, {".txt"})

    missing_labels = 0
    background_images = 0
    corrupted_images = 0
    label_error_files = 0

    total_class_count = Counter()
    total_bbox_sizes = Counter()
    image_sizes = []

    images_with_fire = 0
    images_with_smoke = 0
    images_with_both = 0
    images_only_fire = 0
    images_only_smoke = 0

    image_stems = {img.relative_to(image_dir).with_suffix("") for img in images}
    label_stems = {lbl.relative_to(label_dir).with_suffix("") for lbl in labels}
    labels_without_images = len(label_stems - image_stems)

    for image in images:
        size, image_error = read_image_size(image)

        if image_error:
            corrupted_images += 1
        elif size:
            image_sizes.append(size)

        label_path = get_label_path(image, image_dir, label_dir)
        status, errors, class_count, classes_in_image, bbox_sizes = check_label(label_path, nc)

        if status == "missing":
            missing_labels += 1
            continue

        if status == "empty":
            background_images += 1

        if errors:
            label_error_files += 1

        total_class_count.update(class_count)
        total_bbox_sizes.update(bbox_sizes)

        has_fire = bool(classes_in_image & fire_ids)
        has_smoke = bool(classes_in_image & smoke_ids)

        if has_fire:
            images_with_fire += 1
        if has_smoke:
            images_with_smoke += 1
        if has_fire and has_smoke:
            images_with_both += 1
        if has_fire and not has_smoke:
            images_only_fire += 1
        if has_smoke and not has_fire:
            images_only_smoke += 1

    total_boxes = sum(total_class_count.values())

    print(f"  Tổng số ảnh              : {len(images)}")
    print(f"  Số file label             : {len(labels)}")
    print(f"  Thiếu file label          : {missing_labels}")
    print(f"  Label không có ảnh        : {labels_without_images}")
    print(f"  Ảnh background            : {background_images}")
    print(f"  Ảnh lỗi / không đọc được  : {corrupted_images}")
    print(f"  File label có lỗi         : {label_error_files}")
    print(f"  Tổng số bounding box      : {total_boxes}")

    print_class_count(total_class_count, names)

    print("\n  Số ảnh theo nội dung:")
    print(f"    Ảnh có fire             : {images_with_fire}")
    print(f"    Ảnh có smoke            : {images_with_smoke}")
    print(f"    Ảnh có cả fire và smoke : {images_with_both}")
    print(f"    Ảnh chỉ có fire         : {images_only_fire}")
    print(f"    Ảnh chỉ có smoke        : {images_only_smoke}")
    print(f"    Ảnh background          : {background_images}")

    print_image_size_stats(image_sizes)
    print_bbox_size_stats(total_bbox_sizes, names)

    return {
        "images": len(images),
        "boxes": total_boxes,
        "background": background_images,
        "errors": label_error_files,
        "class_count": total_class_count,
    }


def main():
    cfg, root, nc, names = load_yaml()

    fire_ids = find_class_ids(names, "fire")
    smoke_ids = find_class_ids(names, "smoke")

    print("═" * 60)
    print("  KIỂM TRA DATASET YOLO FIRE / SMOKE")
    print("═" * 60)
    print(f"  Dataset path : {root.resolve()}")
    print(f"  Số class     : {nc}")
    print(f"  Classes      : {names}")

    if not fire_ids:
        print("  [WARN] Không tìm thấy class tên fire")
    if not smoke_ids:
        print("  [WARN] Không tìm thấy class tên smoke")

    total_images = 0
    total_boxes = 0
    total_background = 0
    total_errors = 0
    total_class_count = Counter()

    for key, split_name in SPLITS:
        image_dir, label_dir = get_split_dirs(cfg, root, key)
        result = check_split(split_name, image_dir, label_dir, nc, names, fire_ids, smoke_ids)

        total_images += result["images"]
        total_boxes += result["boxes"]
        total_background += result["background"]
        total_errors += result["errors"]
        total_class_count.update(result["class_count"])

    print(f"\n{'═' * 60}")
    print("  TỔNG KẾT DATASET")
    print("═" * 60)
    print(f"  Tổng số ảnh        : {total_images}")
    print(f"  Tổng số box        : {total_boxes}")
    print(f"  Tổng ảnh background: {total_background}")
    print(f"  Tổng file label lỗi: {total_errors}")

    print_class_count(total_class_count, names)

    print(f"\n{'═' * 60}")
    print("  KIỂM TRA HOÀN TẤT")
    print("═" * 60)


if __name__ == "__main__":
    main()
