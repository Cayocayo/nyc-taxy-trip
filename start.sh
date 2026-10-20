#!/bin/bash
# Menjalankan FastAPI di background pada port 8000
uvicorn api:app --host 0.0.0.0 --port 8000 &

# Menjalankan Streamlit di foreground pada port 8501
streamlit run app.py --server.port=8501 --server.address=0.0.0.0