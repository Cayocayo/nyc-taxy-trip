# Menggunakan base image Python 3.11 yang stabil untuk TensorFlow & Scikit-Learn
FROM python:3.11-slim

# Set direktori kerja di dalam container
WORKDIR /app

# Install system dependencies dasar
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt terlebih dahulu agar caching Docker efisien
COPY requirements.txt .

# Install seluruh library Python
RUN pip install --no-cache-dir -r requirements.txt

# Copy seluruh kode aplikasi dan model ke dalam container
COPY app.py .
COPY api.py .
COPY start.sh .
COPY models/ ./models/

# Beri izin eksekusi pada script start.sh
RUN chmod +x start.sh

# Expose port untuk Streamlit (8501) dan FastAPI (8000)
EXPOSE 8501
EXPOSE 8000

# Jalankan kedua service saat container menyala
CMD ["./start.sh"]