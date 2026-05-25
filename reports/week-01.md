# Week 01 - Khởi tạo dự án YOLO11

## Mục tiêu
- Tạo repo GitHub
- Tạo cấu trúc thư mục dự án
- Chuẩn bị môi trường và dataset
- Train thử mô hình YOLO11

## Công việc đã làm
- Tạo repo `yolo11-smoke-fire-detection`
- Tạo các thư mục `src`, `datasets`, `reports`, `assets`, `results`
- Tạo các file `README.md`, `requirements.txt`, `data.yaml`
- Tổ chức dataset vào `datasets/fire_data`
- Viết `check_dataset.py` để kiểm tra dữ liệu
- Viết `train.py` để train thử mô hình
- Chạy train thử thành công với `epochs=1`

## Kết quả
- Môi trường chạy YOLO11 đã hoạt động
- Dataset đã được tổ chức đúng hơn
- Mô hình đã train thử thành công
- Kết quả được lưu trong `runs/detect/train13`

## Khó khăn
- Sai Python interpreter
- Thiếu thư viện `ultralytics`
- Sai đường dẫn trong `data.yaml`
- Ảnh và nhãn có lúc không khớp tên

## Kế hoạch tiếp theo
- Đưa mã nguồn lên GitHub
- Tăng số epoch
- Bổ sung dữ liệu smoke và background