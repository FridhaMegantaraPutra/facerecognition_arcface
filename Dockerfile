# Gunakan base image Python yang ringan
FROM python:3.10-slim

# Install dependency sistem agar OpenCV, Psycopg2, dan Numpy bisa bekerja
RUN apt-get update && apt-get install -y \
    build-essential \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    libjpeg-dev \
    libpng-dev \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

# Set direktori kerja
WORKDIR /app

# Salin file requirements.txt terlebih dahulu
COPY requirements.txt .

# Install dependensi Python
RUN pip install --upgrade pip setuptools wheel
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua source code ke dalam container
COPY . .

# Expose port Flask
EXPOSE 5000

# Jalankan inisialisasi DB, lalu start Gunicorn
CMD bash -c "python init_db.py && gunicorn --bind 0.0.0.0:5000 app:app"
