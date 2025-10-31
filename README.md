# Sistem Pengenalan Wajah

Aplikasi web untuk pendaftaran dan pengenalan wajah menggunakan teknologi deep learning dengan model ArcFace.
## 🧠 Model Download

<p align="center">
  <a href="https://huggingface.co/garavv/arcface-onnx" target="_blank">
    <img src="https://img.shields.io/badge/Download%20Model-ArcFace%20ONNX-blue?style=for-the-badge&logo=huggingface" alt="Download ArcFace ONNX">
  </a>
</p>

Model ini digunakan untuk melakukan ekstraksi fitur wajah (face embedding) yang nantinya dibandingkan menggunakan cosine similarity untuk proses face recognition.


## 🎥 Demo Video

[![Watch the demo](https://img.youtube.com/vi/QwmApFc_Cho/0.jpg)](https://youtu.be/QwmApFc_Cho)

Klik gambar di atas untuk menonton video demonstrasi Face Recognition ArcFace di YouTube.


## Deskripsi

Sistem ini memungkinkan pengguna untuk mendaftarkan wajah baru ke dalam database dan melakukan pengenalan wajah dari gambar yang diunggah. Aplikasi menggunakan Flask sebagai backend, PostgreSQL untuk penyimpanan data, dan model ArcFace ONNX untuk ekstraksi fitur wajah.

## Fitur Utama

- Pendaftaran wajah baru dengan nama dan foto
- Pengenalan wajah dari gambar yang diunggah
- Visualisasi proses deteksi dan preprocessing wajah
- Manajemen database wajah (lihat dan hapus data)
- Antarmuka web yang responsif dan mudah digunakan
- Containerized dengan Docker untuk deployment yang mudah

## Arsitektur Sistem

### Diagram Alur Penyimpanan Wajah

```
┌─────────────────┐
│  Upload Gambar  │
│   + Nama User   │
└────────┬────────┘
         │
         ▼
┌─────────────────────────────┐
│  Deteksi Wajah (OpenCV)     │
│  - Haarcascade Classifier   │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Preprocessing              │
│  - Crop wajah (x,y,w,h)     │
│  - Resize 112x112           │
│  - Convert RGB              │
│  - Normalisasi (0-1)        │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Ekstraksi Fitur (ArcFace)  │
│  - Input: 112x112x3         │
│  - Output: Embedding 512-D  │
│  - Normalisasi L2           │
└────────┬────────────────────┘
         │
         ▼
┌─────────────────────────────┐
│  Simpan ke Database         │
│  - ID (auto increment)      │
│  - Nama (string)            │
│  - Embedding (binary)       │
└─────────────────────────────┘

```
### visualisasi lewat gambar messi
![messi digaram](goat_messi.png)


### Diagram Alur Pengenalan Wajah
```mermaid
flowchart TD
    A["Upload Gambar\nuntuk Diuji"] --> B["Deteksi Wajah (OpenCV)\n- Haarcascade Classifier"]
    B --> C["Preprocessing\n- Crop wajah\n- Resize 112x112\n- Convert RGB\n- Normalisasi"]
    C --> D["Ekstraksi Fitur (ArcFace)\n- Output: Embedding 512-D\n- Normalisasi L2"]
    D --> E["Ambil Semua Embedding\n dari Database"]
    E --> F["Hitung Cosine Similarity\n dengan Setiap Embedding\n di Database"]
    F --> G["Cek Threshold\n- Similarity > 0.2 ?"]
    
    G -->|Ya| H["Wajah Dikenali\nReturn: Nama + Confidence"]
    G -->|Tidak| I["Wajah Tidak Dikenali"]




## Teknologi yang Digunakan

### Backend
- Python 3.10
- Flask 2.3.3 - Web framework
- Flask-SQLAlchemy 3.1.1 - ORM database
- Gunicorn 20.1.0 - WSGI server

### Machine Learning & Computer Vision
- OpenCV 4.9.0.80 - Deteksi wajah
- ONNX Runtime 1.17.3 - Inferensi model ArcFace
- NumPy 1.26.4 - Operasi array
- Scikit-learn 1.3.2 - Cosine similarity

### Database
- PostgreSQL 16
- Psycopg2-binary 2.9.9 - PostgreSQL adapter

### Frontend
- Bootstrap 5.1.3
- Vanilla JavaScript

## Struktur Proyek

```
face-recognitin/
│
├── app.py                  # Entry point aplikasi Flask
├── database.py             # Model database dan konfigurasi
├── face_utils.py           # Logika pengenalan wajah
├── init_db.py              # Script inisialisasi database
│
├── models/
│   └── arc.onnx            # Model ArcFace (tidak termasuk di repo)
│
├── templates/
│   └── index.html          # Template HTML utama
│
├── static/
│   └── style.css           # Stylesheet
│
├── requirements.txt        # Dependensi Python
├── Dockerfile              # Docker image configuration
├── docker-compose.yml      # Docker compose configuration
└── .env                    # Environment variables
```

## Persyaratan Sistem

- Docker dan Docker Compose
- Minimal 2GB RAM
- Model ArcFace ONNX (arc.onnx) - unduh terlebih dahulu

## Instalasi dan Menjalankan

### Menggunakan Docker (Rekomendasi)

1. Clone repository ini

2. Download model ArcFace ONNX dan letakkan di root folder dengan nama `arc.onnx`

3. Pastikan file `.env` sudah dikonfigurasi dengan benar:
```env
DATABASE_URL='postgresql://postgres:150802@localhost:5432/face_db'
```

4. Jalankan dengan Docker Compose:
```bash
docker-compose up --build
```

5. Akses aplikasi di browser: `http://localhost:5000`

### Instalasi Manual

1. Install PostgreSQL 16

2. Buat database:
```sql
CREATE DATABASE face_db;
```

3. Install dependensi Python:
```bash
pip install -r requirements.txt
```

4. Pastikan model `arc.onnx` ada di folder `models/`

5. Inisialisasi database:
```bash
python init_db.py
```

6. Jalankan aplikasi:
```bash
python app.py
```

## Cara Penggunaan

### Mendaftarkan Wajah Baru

1. Buka halaman utama aplikasi
2. Di bagian "Pendaftaran Wajah", masukkan nama
3. Upload foto wajah (pastikan wajah terlihat jelas)
4. Klik tombol "Daftar"
5. Sistem akan menampilkan proses deteksi dan embedding wajah
6. Wajah akan tersimpan di database

### Mengenali Wajah

1. Di bagian "Pengenalan Wajah", upload foto yang ingin dikenali
2. Klik tombol "Kenali"
3. Sistem akan menampilkan hasil:
   - Jika wajah dikenali: nama dan tingkat keyakinan
   - Jika tidak dikenali: notifikasi wajah tidak ditemukan
4. Proses deteksi akan ditampilkan di modal

### Mengelola Database

- Lihat daftar wajah yang terdaftar di tabel "Daftar Wajah"
- Klik tombol "Hapus" untuk menghapus wajah dari database
- Klik tombol "Refresh" untuk memperbarui daftar

## API Endpoints

### GET /api/face
Mendapatkan semua wajah yang terdaftar

**Response:**
```json
{
  "status": "success",
  "data": [
    {
      "id": 1,
      "name": "John Doe",
      "embedding_preview": ["0.1234", "0.5678", "..."]
    }
  ]
}
```

### POST /api/face/register
Mendaftarkan wajah baru

**Request:** 
- Form-data dengan field `name` dan `image`

**Response:**
```json
{
  "status": "success",
  "message": "Wajah berhasil didaftarkan",
  "data": {
    "id": 1,
    "name": "John Doe",
    "embedding_preview": ["0.1234", "..."],
    "process_images": {
      "original": "base64...",
      "detection": "base64...",
      "cropped": "base64...",
      "preprocessed": "base64..."
    }
  }
}
```

### POST /api/face/recognize
Mengenali wajah dari gambar

**Request:**
- Form-data dengan field `image`

**Response:**
```json
{
  "status": "success",
  "message": "Wajah dikenali",
  "data": {
    "id": 1,
    "name": "John Doe",
    "confidence": 0.85,
    "embedding_preview": ["0.1234", "..."],
    "process_images": {...}
  }
}
```

### DELETE /api/face/{id}
Menghapus wajah berdasarkan ID

**Response:**
```json
{
  "status": "success",
  "message": "Wajah dengan ID 1 berhasil dihapus"
}
```

## Detail Teknis

### Model ArcFace

Model ArcFace yang digunakan menghasilkan embedding 512 dimensi untuk setiap wajah. Embedding ini merepresentasikan fitur unik dari wajah yang dapat digunakan untuk perbandingan.

**Spesifikasi Model:**
- Input: 112x112x3 (RGB image)
- Output: 512-dimensional embedding vector
- Format: ONNX

### Proses Deteksi Wajah

1. **Deteksi**: Menggunakan Haar Cascade Classifier dari OpenCV
2. **Cropping**: Memotong area wajah berdasarkan koordinat deteksi
3. **Resize**: Mengubah ukuran menjadi 112x112 pixel
4. **Normalisasi**: Nilai pixel dinormalisasi ke range 0-1
5. **Ekstraksi**: Embedding dihasilkan oleh model ArcFace

### Pencocokan Wajah

Sistem menggunakan Cosine Similarity untuk membandingkan embedding:

```
similarity = (embedding1 · embedding2) / (||embedding1|| × ||embedding2||)
```

Threshold yang digunakan: **0.2**
- Similarity >= 0.2: Wajah dikenali
- Similarity < 0.2: Wajah tidak dikenali

### Penyimpanan Database

Embedding disimpan sebagai binary data menggunakan pickle serialization di kolom PostgreSQL dengan tipe `LargeBinary`.

## Troubleshooting

### Tidak ada wajah terdeteksi
- Pastikan foto memiliki pencahayaan yang baik
- Wajah harus menghadap kamera
- Ukuran wajah dalam gambar tidak terlalu kecil

### Wajah tidak dikenali padahal sudah terdaftar
- Coba dengan foto yang berbeda
- Pastikan kondisi foto saat pendaftaran dan pengenalan serupa
- Periksa nilai confidence yang ditampilkan

### Error koneksi database
- Pastikan PostgreSQL sudah berjalan
- Periksa konfigurasi `DATABASE_URL` di `.env`
- Untuk Docker, pastikan service `db` sudah ready

### Model tidak ditemukan
- Download model ArcFace ONNX
- Letakkan file dengan nama `arc.onnx` di root atau folder `models/`


