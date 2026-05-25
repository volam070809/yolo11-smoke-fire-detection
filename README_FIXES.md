# YOLO11 Smoke & Fire Detection – Phân tích lỗi & Hướng dẫn sửa

## Tóm tắt các lỗi được phát hiện

---

## 🔴 BUG NGHIÊM TRỌNG

### Bug 1 – `data.yaml`: Thiếu khai báo `nc:`
**File:** `data.yaml`

```yaml
# ❌ Bản gốc (thiếu nc)
path: datasets/fire_data
train: train/images
val: valid/images
names:
  0: fire

# ✅ Đã sửa (thêm nc:)
nc: 1
names:
  0: fire
```
**Hậu quả:** Một số phiên bản ultralytics báo lỗi hoặc load sai số class.

---

### Bug 2 – `train.py`: Epochs quá ít → mAP thấp
**File:** `src/train.py`

```python
# ❌ Bản gốc – chỉ 10 epoch
model.train(data="data.yaml", epochs=10, imgsz=640)

# ✅ Đã sửa – 100 epoch + patience
model.train(data="data.yaml", epochs=100, imgsz=640,
            patience=20, cos_lr=True, ...)
```
**Hậu quả:** Kết quả training thực tế cho thấy sau 1 epoch mAP50 chỉ đạt 0.03–0.32, sau 3 epoch đạt 0.43. Cần ít nhất 50–100 epochs để hội tụ.

**Số liệu từ training log của bạn:**
| Epoch | mAP50 | Precision | Recall |
|-------|-------|-----------|--------|
| 1     | 0.03–0.32 | 0.004–0.62 | 0.97 |
| 3     | 0.43  | 0.62      | 0.41  |

---

### Bug 3 – Training trên CPU không khai báo rõ device
**File:** `src/train.py`

```python
# ❌ Bản gốc – không khai báo device, mặc định CPU
model.train(data="data.yaml", epochs=10, imgsz=640)
# Kết quả: 1 epoch mất ~164-200 giây → 50 epoch = ~2.5 giờ!

# ✅ Đã sửa – tự động chọn GPU nếu có
device = "cuda" if torch.cuda.is_available() else "cpu"
model.train(..., device=device)
```
**Hậu quả:** Cực kỳ chậm. Dùng Google Colab GPU để tăng tốc ~10-50x.

---

### Bug 4 – `predict_test.py`: conf=0.25 quá thấp → nhiều false positive
**File:** `src/predict_test.py`

```python
# ❌ Bản gốc
model.predict(..., conf=0.25, ...)

# ✅ Đã sửa
model.predict(..., conf=0.40, ...)
```
**Hậu quả:** Với conf=0.25, model báo lửa/khói sai rất nhiều, đặc biệt khi precision thấp.

---

### Bug 5 – Không xử lý trường hợp best.pt chưa tồn tại
**File:** `src/predict_test.py`, `src/test_model.py`

```python
# ❌ Bản gốc – raise exception thô
if not candidates:
    raise FileNotFoundError("Chua co best.pt nao de predict.")

# ✅ Đã sửa – fallback + thông báo rõ ràng
if not candidates:
    if Path("yolo11n.pt").exists():
        print("[WARNING] Chưa có best.pt, dùng yolo11n.pt")
        return "yolo11n.pt"
    print("[ERROR] Không tìm thấy model. Chạy train.py trước!")
    sys.exit(1)
```

---

## 🟡 BUG VỪA

### Bug 6 – `check_dataset.py`: Không đọc data.yaml, không validate label
**File:** `src/check_dataset.py`

- Bản gốc hardcode đường dẫn `datasets/fire_data`
- Không kiểm tra label hợp lệ (tọa độ ngoài [0,1], class_id âm)
- Không đếm box theo class
- Không phát hiện split "valid" vs "val" mismatch

---

### Bug 7 – `train.py`: Không có `workers=0` trên Windows
**File:** `src/train.py`

```python
# ❌ Bản gốc – lỗi DataLoader trên Windows
model.train(data="data.yaml", ...)

# ✅ Đã sửa
workers = 0 if os.name == "nt" else 4
model.train(..., workers=workers)
```
**Hậu quả:** Crash với lỗi "RuntimeError: An attempt has been made to start a new process before the current process has finished its bootstrapping phase" trên Windows.

---

### Bug 8 – Không có `project` và `name` → kết quả lưu lẫn lộn
**File:** `src/train.py`

```python
# ❌ Bản gốc – lưu vào "runs/detect/train", "runs/detect/train2", ...
model.train(data="data.yaml", epochs=10, imgsz=640)

# ✅ Đã sửa – tên có nghĩa
model.train(..., project="runs/detect", name="train_fire", exist_ok=True)
```

---

## 🟢 THIẾU TÍNH NĂNG

### Thiếu 1 – Không có real-time detection
→ Thêm `src/detect_realtime.py` hỗ trợ webcam, video file, camera IP (RTSP).

### Thiếu 2 – Không có hệ thống cảnh báo
→ Thêm beep âm thanh và log sự kiện vào file khi phát hiện lửa/khói.

### Thiếu 3 – Không có hỗ trợ video trong predict_test.py
→ Bản gốc chỉ xử lý ảnh tĩnh. Đã thêm hỗ trợ video MP4/AVI.

### Thiếu 4 – Class "smoke" không được annotate
Hiện tại dataset chỉ có 1 class: **fire** (lửa). Không có nhãn **smoke** (khói).  
Đây là giới hạn quan trọng vì tên project là "smoke-fire-detection".

**Để thêm class smoke:**
1. Dùng tool annotate (Roboflow, LabelImg, CVAT) để vẽ thêm bounding box cho khói
2. Đặt class_id = 1 cho smoke
3. Sửa `data.yaml`:
   ```yaml
   nc: 2
   names:
     0: fire
     1: smoke
   ```

---

## Cấu trúc thư mục sau khi sửa

```
yolo11-smoke-fire-detection/
├── data.yaml              ← ĐÃ SỬA (thêm nc:1)
├── requirements.txt       ← ĐÃ CẬP NHẬT
├── datasets/
│   └── fire_data/
│       ├── train/images/ & labels/
│       ├── valid/images/ & labels/   ← tên là "valid" không phải "val"
│       └── test/images/  & labels/
└── src/
    ├── check_dataset.py   ← ĐÃ SỬA (validate label, đọc yaml)
    ├── train.py           ← ĐÃ SỬA (100 epochs, GPU, patience, workers)
    ├── test_model.py      ← ĐÃ SỬA (F1, per-class, lưu CSV)
    ├── predict_test.py    ← ĐÃ SỬA (video, webcam, conf=0.4, alert)
    └── detect_realtime.py ← MỚI (real-time webcam/video + cảnh báo)
```

---

## Quy trình chạy

```bash
# 1. Cài thư viện
pip install -r requirements.txt

# 2. Kiểm tra dataset
python src/check_dataset.py

# 3. Train (khuyến nghị dùng GPU)
python src/train.py

# 4. Đánh giá model
python src/test_model.py

# 5a. Predict trên ảnh test
python src/predict_test.py --source datasets/fire_data/test/images

# 5b. Predict trên video
python src/predict_test.py --source path/to/video.mp4

# 6. Real-time webcam
python src/detect_realtime.py --source 0 --conf 0.5
```

---

## Khuyến nghị nâng cao

| Vấn đề | Giải pháp |
|--------|-----------|
| Train chậm (CPU) | Dùng Google Colab + GPU T4 miễn phí |
| mAP50 thấp < 0.5 | Tăng epochs lên 150-200, thêm data |
| Nhiều false positive | Tăng conf lên 0.5-0.6 |
| Muốn detect khói | Annotate lại dataset với class smoke |
| Deploy nhúng/mobile | Export: `model.export(format="onnx")` |
| Detect video real-time | Dùng `detect_realtime.py` |
