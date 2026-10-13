# Pakai OS Linux versi ringan biar RAM 512MB cukup
FROM python:3.10-slim

# Bikin folder kerja di dalam server
WORKDIR /app

# Install FFmpeg untuk potong video dan library pendukung OpenCV
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy file requirements dan install library Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy semua kode lu (main.py, services, dll)
COPY . .

# Perintah untuk menyalakan server saat mesin hidup
CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}"]
