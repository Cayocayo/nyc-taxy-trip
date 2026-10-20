from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import pandas as pd
import tensorflow as tf
import joblib
import sklearn.compose._column_transformer

# Perbaikan kompatibilitas versi scikit-learn antara Colab dan lokal
if not hasattr(sklearn.compose._column_transformer, '_RemainderColsList'):
    sklearn.compose._column_transformer._RemainderColsList = list

# 1. Inisialisasi Aplikasi FastAPI
app = FastAPI(
    title="NYC Taxi Fare Prediction API",
    description="REST API Microservice menggunakan Deep Learning untuk memprediksi tarif taksi NYC secara real-time.",
    version="1.0.0"
)

# 2. Load Model dan Preprocessor saat startup
model = tf.keras.models.load_model("models/taxi_dl_model.h5", compile=False)
preprocessor = joblib.load("models/taxi_preprocessor.pkl")

# 3. Skema Validasi Input (Pydantic)
class TaxiTripInput(BaseModel):
    passenger_count: int = Field(1, ge=1, le=6, description="Jumlah penumpang (1-6)")
    trip_distance: float = Field(2.5, gt=0.0, le=100.0, description="Jarak tempuh dalam satuan Miles")
    duration_min: float = Field(15.0, gt=0.0, le=300.0, description="Estimasi durasi perjalanan (Menit)")
    PULocationID: int = Field(161, ge=1, le=265, description="ID Zona Penjemputan NYC (1-265)")
    DOLocationID: int = Field(236, ge=1, le=265, description="ID Zona Penurunan NYC (1-265)")
    payment_type: int = Field(1, ge=1, le=4, description="1=Credit Card, 2=Cash, 3=No Charge, 4=Dispute")
    pickup_hour: int = Field(12, ge=0, le=23, description="Jam keberangkatan (0-23)")
    pickup_dayofweek: int = Field(0, ge=0, le=6, description="Hari keberangkatan (0=Senin, 6=Minggu)")

# 4. Endpoint Health Check
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "service": "NYC Taxi Fare Prediction API (Deep Learning)",
        "model_loaded": True
    }

# 5. Endpoint Prediksi Tarif
@app.post("/predict")
def predict_fare(trip: TaxiTripInput):
    try:
        # Ubah request JSON menjadi DataFrame
        input_df = pd.DataFrame([trip.model_dump()])
        
        # Transformasi fitur menggunakan pipeline preprocessor
        processed_data = preprocessor.transform(input_df)
        
        # Prediksi menggunakan model Deep Learning
        prediction = float(model.predict(processed_data, verbose=0)[0][0])
        
        return {
            "status": "success",
            "estimated_fare_usd": round(prediction, 2),
            "currency": "USD",
            "input_summary": trip.model_dump()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))