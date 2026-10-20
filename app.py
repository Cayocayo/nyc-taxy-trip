import streamlit as st
import pandas as pd
import requests

# 1. Konfigurasi Halaman & Tema Cerah (CSS)
st.set_page_config(page_title="NYC Taxi Fare Predictor", page_icon="🚕", layout="wide")

st.markdown("""
    <style>
    /* Background utama cerah */
    .stApp {
        background-color: #f4f9f9 !important;
        background-image: linear-gradient(135deg, #fdfbfb 0%, #e2ebf0 100%) !important;
        color: #1A365D !important;
    }
    /* Paksa semua judul, paragraf, dan label input menjadi gelap & jelas */
    h1, h2, h3, h4, h5, h6, p, span, label, .stMarkdown {
        color: #1A365D !important;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    }
    section[data-testid="stSidebar"] {
        background-color: #ffffff !important;
    }
    .stButton>button {
        background-color: #ff9f43 !important;
        color: #ffffff !important;
        font-weight: bold;
        border-radius: 10px;
        border: none;
        padding: 10px 24px;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #ee5253 !important;
        color: #ffffff !important;
        transform: scale(1.05);
    }
    div[data-testid="stForm"] {
        background-color: rgba(255, 255, 255, 0.95) !important;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        padding: 24px;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Fungsi Load Data EDA
@st.cache_data
def load_data():
    df = pd.read_csv("data/NYC_Taxi_Cleaned_Analysis_Ready.csv", nrows=10000)
    return df

# URL Microservice Backend FastAPI
API_URL = "http://127.0.0.1:8000/predict"

# 3. Sidebar Navigasi
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3204/3204395.png", width=100)
    st.title("Menu Navigasi")
    menu = st.radio("Pilih Halaman:", ["📊 Dashboard EDA", "🔮 Kalkulator Prediksi"])
    st.markdown("---")
    st.info("Aplikasi ini terhubung ke **FastAPI Backend Microservice** yang mengeksekusi model **Deep Learning** secara real-time.")

# 4. Halaman Dashboard EDA
if menu == "📊 Dashboard EDA":
    st.title("🚕 Dashboard Analisis Taksi NYC")
    st.markdown("Eksplorasi data sampel perjalanan taksi untuk mendapatkan **Insight Bisnis** operasional.")
    
    with st.spinner("Memuat data..."):
        df = load_data()
    
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Distribusi Tarif Perjalanan ($)")
        st.bar_chart(df['fare_amount'].value_counts(bins=20).sort_index(), color="#ff9f43")
        
    with col2:
        st.subheader("Distribusi Jarak Tempuh (Miles)")
        st.bar_chart(df['trip_distance'].value_counts(bins=20).sort_index(), color="#0abde3")
        
    st.subheader("Data Sampel Mentah (Tabel Interaktif)")
    st.dataframe(df.head(100), use_container_width=True)

# 5. Halaman Kalkulator Prediksi (Memanggil FastAPI)
elif menu == "🔮 Kalkulator Prediksi":
    st.title("🔮 Kalkulator Prediksi Tarif (AI Microservice)")
    st.markdown("Masukkan detail rencana perjalanan Anda untuk mendapatkan estimasi tarif dari model Deep Learning.")
    
    with st.form("prediction_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            passenger_count = st.number_input("Jumlah Penumpang 🧑‍🤝‍🧑", min_value=1, max_value=6, value=1)
            trip_distance = st.number_input("Jarak Tempuh (Miles) 📏", min_value=0.1, max_value=50.0, value=2.0, step=0.1)
            duration_min = st.number_input("Estimasi Durasi (Menit) ⏳", min_value=1.0, max_value=120.0, value=15.0, step=1.0)
            
        with col2:
            PULocationID = st.number_input("ID Zona Penjemputan 📍", min_value=1, max_value=265, value=161)
            DOLocationID = st.number_input("ID Zona Penurunan 🏁", min_value=1, max_value=265, value=236)
            pickup_hour = st.slider("Jam Keberangkatan 🕒", 0, 23, 12)
            pickup_dayofweek = st.selectbox("Hari (0=Senin, 6=Minggu) 📅", options=[0, 1, 2, 3, 4, 5, 6], index=0)
            payment_type = st.selectbox("Tipe Pembayaran 💳", options=[1, 2, 3, 4], index=0, format_func=lambda x: {1: "Credit Card", 2: "Cash", 3: "No Charge", 4: "Dispute"}.get(x))

        st.markdown("<br>", unsafe_allow_html=True)
        submit_button = st.form_submit_button("Hitung Estimasi Tarif 🚀")

    if submit_button:
        payload = {
            "passenger_count": int(passenger_count),
            "trip_distance": float(trip_distance),
            "duration_min": float(duration_min),
            "PULocationID": int(PULocationID),
            "DOLocationID": int(DOLocationID),
            "payment_type": int(payment_type),
            "pickup_hour": int(pickup_hour),
            "pickup_dayofweek": int(pickup_dayofweek)
        }
        
        try:
            with st.spinner("Mengirim data ke AI Microservice..."):
                response = requests.post(API_URL, json=payload, timeout=10)
                
            if response.status_code == 200:
                result = response.json()
                predicted_fare = result["estimated_fare_usd"]
                st.success(f"### Estimasi Tarif Terbaik: **${predicted_fare:.2f}**")
                st.balloons()
            else:
                st.error(f"Error dari API: {response.text}")
        except requests.exceptions.ConnectionError:
            st.error("⚠️ Gagal terhubung ke FastAPI! Pastikan backend sudah menyala dengan perintah: `uvicorn api:app --port 8000` di terminal.")
        except Exception as e:
            st.error(f"Terjadi kesalahan: {e}")