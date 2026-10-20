# 🚕 NYC Taxi Fare Prediction — End-to-End MLOps System

[![Python](https://img.shields.io/badge/Python-3.11%20%7C%203.13-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-Microservice%20Backend-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-Interactive%20Frontend-FF4B4B.svg)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-FF6F00.svg)](https://www.tensorflow.org/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED.svg)](https://www.docker.com/)

Sistem estimasi tarif taksi instan (*Upfront Fare Prediction*) berbasis **Deep Learning** berskala produksi untuk New York City. Proyek ini dibangun secara *end-to-end* dengan arsitektur **Microservices terpisah**: antarmuka visual cerah menggunakan **Streamlit**, backend model serving berbasis **FastAPI**, serta terintegrasi penuh dalam *container* **Docker**.

---

## 📌 Daftar Isi
- [Latar Belakang & Masalah Bisnis](#-latar-belakang--masalah-bisnis)
- [Arsitektur Sistem & Tech Stack](#-arsitektur-sistem--tech-stack)
- [Data Engineering & SQL Pipeline](#-data-engineering--sql-pipeline)
- [Pengembangan Model & Evaluasi](#-pengembangan-model--evaluasi)
- [Struktur Direktori Repositori](#-struktur-direktori-repositori)
- [Cara Menjalankan Secara Lokal](#-cara-menjalankan-secara-lokal)
- [Deployment dengan Docker](#-deployment-dengan-docker)
- [Rekomendasi Bisnis](#-rekomendasi-bisnis)

---

## 🎯 Latar Belakang & Masalah Bisnis
* **Permasalahan:** Tarif taksi konvensional dengan argo sering kali memicu ketidakpastian harga bagi penumpang akibat kepadatan lalu lintas dan rute tak terduga, yang berujung pada tingginya angka pembatalan pemesanan (*ride cancellation*).
* **Solusi AI:** Membangun mesin prediksi tarif instan di muka (*Upfront Pricing Engine*) sebelum penumpang memulai perjalanan, dengan target deviasi kesalahan rata-rata (**Mean Absolute Error / MAE**) di bawah **$1.50**.

---

## 🏗️ Arsitektur Sistem & Tech Stack

[ Pengguna / Stakeholder ]
│
▼
┌─────────────────────────┐
│   Streamlit Frontend    │  (Port 8501: Dashboard EDA & Input Kalkulator)
└───────────┬─────────────┘
│  HTTP POST JSON (/predict)
▼
┌─────────────────────────┐
│    FastAPI Backend      │  (Port 8000: Pydantic Validation & Swagger UI)
└───────────┬─────────────┘
│
▼
┌─────────────────────────┐
│ Deep Learning Engine    │  (Pipeline Preprocessor + TensorFlow Keras .h5)
└─────────────────────────┘


* **Core Language:** Python 3.11 / 3.13
* **Data Processing & SQL:** DuckDB, Pandas, NumPy
* **Machine Learning & Pipeline:** Scikit-Learn (`ColumnTransformer`, `StandardScaler`, `OneHotEncoder`), Joblib
* **Deep Learning:** TensorFlow / Keras (Multi-Layer Perceptron dengan `BatchNormalization` & `Dropout`)
* **API & Serving:** FastAPI, Uvicorn, Pydantic
* **User Interface:** Streamlit (Clean Bright Theme)
* **DevOps / MLOps:** Docker (`python:3.11-slim`), Bash Scripting (`start.sh`)

---

## 📊 Data Engineering & SQL Pipeline
* **Sumber Data:** *Cleaned NYC Taxi Trip Data (2025 Sample)* — mencakup **3.9+ Juta baris data** (~614 MB).
* **Ekstraksi SQL Cepat (DuckDB):** Menggunakan DuckDB untuk analisis *in-memory* dan pengambilan sampel representatif sebanyak **150.000 baris** dengan filter validitas bisnis (`fare_amount > $2.50` dan `trip_distance > 0`).
* **Feature Preprocessing Pipeline:**
  * **Fitur Numerik** (`trip_distance`, `duration_min`, `passenger_count`): Dinormalisasi menggunakan `StandardScaler`.
  * **Fitur Kategorikal & Temporal** (`PULocationID`, `DOLocationID`, `payment_type`, `pickup_hour`, `pickup_dayofweek`): Ditransformasi dengan `OneHotEncoder(handle_unknown='ignore')`.

---

## 🤖 Pengembangan Model & Evaluasi
Eksperimen membandingkan model ensemble pohon keputusan dengan arsitektur *Deep Learning*:

1. **Baseline Model:** `RandomForestRegressor`
2. **Advanced Deep Learning (TensorFlow/Keras):**
   * Arsitektur: Dense(128, ReLU) $\rightarrow$ Dense(64, ReLU) $\rightarrow$ Dense(32, ReLU) $\rightarrow$ Output(1, Linear)
   * Regularisasi: `BatchNormalization()`, `Dropout(0.2)`, `Adam Optimizer`, dan `EarlyStopping(patience=5)`

### 🏆 Hasil Evaluasi Model
| Metrik Evaluasi | Target Bisnis | Hasil Model Deep Learning |
| :--- | :---: | :---: |
| **R-Squared ($R^2$)** | > 0.85 | **0.9515 (95.15% Varians Dijelaskan)** |
| **Mean Absolute Error (MAE)** | < $1.50 | **~$1.21** (Melampaui Target) |
| **Generalisasi** | Bebas Overfitting | **Sangat Stabil (Train vs Val Loss Selaras)** |

---

## 📁 Struktur Direktori Repositori
```text
nyc-taxy-trip/
├── data/
│   └── NYC_Taxi_Cleaned_Analysis_Ready.csv   # Data sampel untuk analisis EDA
├── models/
│   ├── taxi_dl_model.h5                     # Model Deep Learning terlatih
│   └── taxi_preprocessor.pkl                # Scikit-learn Pipeline artifact
├── api.py                                   # Backend REST API Microservice (FastAPI)
├── app.py                                   # Frontend Dashboard & Prediksi (Streamlit)
├── start.sh                                 # Runner script multi-service
├── Dockerfile                               # Cetak biru Docker container
├── requirements.txt                         # Dependensi pustaka Python
├── .gitignore                               # File pengecualian version control Git
└── README.md                                # Dokumentasi resmi proyek
💻 Cara Menjalankan Secara Lokal
1. Clone Repositori & Masuk Direktori
Bash
git clone [https://github.com/Cayocayo/nyc-taxy-trip.git](https://github.com/Cayocayo/nyc-taxy-trip.git)
cd nyc-taxy-trip
2. Buat dan Aktifkan Virtual Environment
Bash
python3 -m venv venv
source venv/bin/activate
3. Install Seluruh Dependensi
Bash
pip install -r requirements.txt
4. Jalankan Backend FastAPI (Terminal 1)
Bash
uvicorn api:app --reload --port 8000
Dokumentasi Swagger UI interaktif dapat diakses pada: http://localhost:8000/docs

5. Jalankan Frontend Streamlit (Terminal 2)
Bash
streamlit run app.py
Aplikasi web akan terbuka otomatis di: http://localhost:8501

🐳 Deployment dengan Docker
Untuk menjalankan seluruh sistem (FastAPI + Streamlit) di dalam satu isolated container:

Bash
# 1. Build image Docker
docker build -t nyc-taxi-mlops .

# 2. Jalankan container
docker run -p 8501:8501 -p 8000:8000 nyc-taxi-mlops
Akses Frontend Streamlit: http://localhost:8501

Akses Backend API: http://localhost:8000/docs

💡 Rekomendasi Bisnis
Upfront Fixed-Price Guarantee: Terapkan tarif pasti di muka menggunakan endpoint API dengan menambahkan safety buffer dinamis sebesar $1.25 (mengacu pada batas MAE model) guna menjaga profitabilitas pengemudi.

Proactive Fleet Dispatching: Relokasi armada taksi ke zona penjemputan dengan frekuensi tinggi (PULocationID 161, 236) 30 menit sebelum jam sibuk (rush hour).

Siklus Retraining Otomatis: Jadwalkan pembaruan model secara berkala menggunakan pipeline DuckDB dan Docker agar adaptif terhadap dinamika lalu lintas musiman di NYC.