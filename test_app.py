import os
import numpy as np
import pandas as pd
import streamlit as st

@st.cache_data
def load_data():
    csv_path = "data/NYC_Taxi_Cleaned_Analysis_Ready.csv"
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path, nrows=10000)
        except Exception:
            pass
    # Fallback data otomatis saat dijalankan di GitHub Actions CI/CD (tanpa file CSV lokal)
    np.random.seed(42)
    n = 500
    return pd.DataFrame({
        "passenger_count": np.random.choice([1, 2, 3, 4], size=n),
        "trip_distance": np.round(np.random.uniform(0.5, 20.0, size=n), 2),
        "duration_min": np.round(np.random.uniform(3.0, 60.0, size=n), 1),
        "PULocationID": np.random.choice([161, 236, 237, 132], size=n),
        "DOLocationID": np.random.choice([236, 161, 138, 230], size=n),
        "payment_type": np.random.choice([1, 2], size=n),
        "fare_amount": np.round(np.random.uniform(5.0, 65.0, size=n), 2)
    })