# ===== Dockerfile for TTS Application (with ffmpeg) =====

# Bước 1: Chọn một hệ điều hành cơ bản (Python 3.10)
FROM python:3.10-slim

# === PHẦN THÊM VÀO: Cài đặt ffmpeg ===
# Chuyển sang user root để cài đặt, sau đó có thể chuyển lại nếu cần
USER root
RUN apt-get update && apt-get install -y ffmpeg && rm -rf /var/lib/apt/lists/*
# ======================================

# Bước 2: Thiết lập thư mục làm việc bên trong "hộp"
WORKDIR /app

# Bước 3: Sao chép file requirements vào trước để tận dụng cache
COPY requirements.txt .

# Bước 4: Cài đặt các thư viện Python
RUN pip install --no-cache-dir -r requirements.txt

# Bước 5: Sao chép toàn bộ source code của bạn vào "hộp"
COPY . .

# Bước 6: Mở cổng để bên ngoài có thể truy cập
EXPOSE 7860

# Bước 7: Lệnh để chạy ứng dụng khi "hộp" được khởi động
CMD ["gunicorn", "--worker-class", "gevent", "--workers", "1", "--bind", "0.0.0.0:7860", "--timeout", "600", "app:app"]