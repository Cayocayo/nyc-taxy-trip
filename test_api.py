from fastapi.testclient import TestClient
from api import app

client = TestClient(app)

def test_health_check():
    """Menguji apakah endpoint utama (GET /) menyala dan mengembalikan status online."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "online"
    assert "r2_score" in data
    assert "mae_usd" in data

def test_predict_fare_valid_input():
    """Menguji prediksi tarif standar pada endpoint (POST /predict)."""
    payload = {
        "passenger_count": 2,
        "trip_distance": 3.5,
        "duration_min": 18.0,
        "PULocationID": 161,
        "DOLocationID": 236,
        "payment_type": 1,
        "pickup_hour": 17,
        "pickup_dayofweek": 1
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert "estimated_fare_usd" in data
    assert isinstance(data["estimated_fare_usd"], float)
    # Memastikan tarif lebih besar dari tarif minimum flag drop NYC ($2.50)
    assert data["estimated_fare_usd"] >= 2.50

def test_predict_fare_minimum_bound():
    """Menguji perjalanan jarak sangat pendek agar tidak di bawah batas minimum $2.50."""
    payload = {
        "passenger_count": 1,
        "trip_distance": 0.1,
        "duration_min": 1.0,
        "PULocationID": 100,
        "DOLocationID": 100,
        "payment_type": 2,
        "pickup_hour": 11,
        "pickup_dayofweek": 2
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert response.json()["estimated_fare_usd"] >= 2.50

def test_predict_fare_invalid_type():
    """Menguji validasi Pydantic jika dikirim tipe data yang salah (string pada angka)."""
    invalid_payload = {
        "passenger_count": "dua_orang",  # Harusnya integer
        "trip_distance": 3.5,
        "duration_min": 15.0,
        "PULocationID": 161,
        "DOLocationID": 236,
        "payment_type": 1,
        "pickup_hour": 14,
        "pickup_dayofweek": 2
    }
    response = client.post("/predict", json=invalid_payload)
    # Pydantic otomatis menolak dengan status 422 Unprocessable Entity
    assert response.status_code == 422