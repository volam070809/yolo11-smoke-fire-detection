from pathlib import Path
import shutil
import os

DATASET_DIR = Path(r"D:\0_DeepLearning\DL_Git\yolo11-smoke-fire-detection\datasets\fire_data")
OUT_DIR = Path(r"D:\0_DeepLearning\DL_Git\yolo11-smoke-fire-detection\background_images_check")

splits = ["train", "valid", "test"]
image_exts = [".jpg", ".jpeg", ".png", ".bmp", ".webp"]

OUT_DIR.mkdir(parents=True, exist_ok=True)

total = 0

for split in splits:
    image_dir = DATASET_DIR / split / "images"
    label_dir = DATASET_DIR / split / "labels"

    if not image_dir.exists() or not label_dir.exists():
        print(f"[SKIP] Không tìm thấy: {split}")
        continue

    out_split_dir = OUT_DIR / split
    out_split_dir.mkdir(parents=True, exist_ok=True)

    count = 0

    for label_path in label_dir.glob("*.txt"):
        content = label_path.read_text(encoding="utf-8", errors="ignore").strip()

        # label trống = background
        if content == "":
            image_path = None

            for ext in image_exts:
                candidate = image_dir / f"{label_path.stem}{ext}"
                if candidate.exists():
                    image_path = candidate
                    break

            if image_path:
                shutil.copy2(image_path, out_split_dir / image_path.name)
                count += 1
                total += 1
                print(f"[{split}] {image_path.name}")
            else:
                print(f"[WARNING] Không tìm thấy ảnh cho label: {label_path.name}")

    print(f"{split}: copy {count} ảnh background")

print("\nHoàn tất!")
print(f"Tổng ảnh background: {total}")
print(f"Thư mục xem ảnh: {OUT_DIR}")

# Tự mở thư mục trên Windows
os.startfile(OUT_DIR)