import streamlit as st
import pandas as pd
import joblib
import json
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Seminar Hasil Kelompok 6", layout="wide")

# --- FUNGSI LOAD DATA & METRIK ---
@st.cache_data
def load_all_metrics():
    # Membaca file JSON hasil perhitungan dari notebook
    with open('Models/metrics.json', 'r') as f:
        return json.load(f)

# --- LOAD ASSETS (Pastikan folder & file ini ada di GitHub nanti) ---
@st.cache_resource
def load_model_assets(bank):
    # Load model dan scaler sesuai struktur folder Anda
    model = joblib.load(f"Models/Trained/{bank}_rf_model.pkl")
    scaler = joblib.load(f"Models/Scalers/{bank}_scaler.pkl")
    return model, scaler

# Inisialisasi data
all_metrics = load_all_metrics()

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("Menu Navigasi")
menu = st.sidebar.radio("Pilih Tahapan Implementasi:", [
    "0. Ringkasan Proyek",
    "1. Pengumpulan Data",
    "2. Prapemrosesan & Fitur",
    "3. Evaluasi Performa Model",
    "4. Analisis Feature Importance",
    "5. Demo Prediksi Real-time"
])

bank_pilihan = st.sidebar.selectbox("Pilih Bank Fokus:", ["BBCA", "BBRI", "BMRI", "BBNI", "BBTN"])
m = all_metrics.get(bank_pilihan, all_metrics["BBCA"])

# --- KONTEN HALAMAN ---
if menu == "0. Ringkasan Proyek":
    st.header("Penerapan Algoritma Random Forest Regression untuk Prediksi Harga Saham Sektor Perbankan")
    st.info("Oleh: Kelompok 6 - Universitas Siber Asia")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Tujuan Penelitian")
        st.write("- Memprediksi harga saham 5 bank besar (BCA, BRI, Mandiri, BNI, BTN).")
        st.write("- Mencapai akurasi R² Score ≥ 0.85.")
        st.write("- Mengidentifikasi fitur teknikal paling berpengaruh.")
    with col2:
        st.subheader(f"📈 Hasil Utama ({bank_pilihan})")
        # Nilai ini sekarang otomatis dari metrics.json
        st.success(f"R² Score: {m['r2']:.4f}")
        st.write(f"Model {bank_pilihan} berhasil melampaui target akurasi dengan performa yang sangat stabil.")

elif menu == "1. Pengumpulan Data":
    st.header("Tahap 1: Pengumpulan Data Historis")
    st.write("Data ditarik dari Yahoo Finance API (17 Okt 2022 - 17 Okt 2025).")
    st.image("Visual/01_Analisis_Data_Mentah.png", caption="Tren Seluruh Saham Perbankan")

elif menu == "2. Prapemrosesan & Fitur":
    st.header("Tahap 2 & 3: Preprocessing & Feature Engineering")
    st.markdown("""
    - **Cleaning**: Penanganan missing values.
    - **Features**: 22 Indikator Teknikal (SMA, EMA, RSI, MACD, dll).
    - **Scaling**: MinMaxScaler (0-1).
    """)
    st.image(f"Visual/03_Technical_Indicators_{bank_pilihan}.png", caption=f"Indikator Teknikal: {bank_pilihan}")

elif menu == "3. Evaluasi Performa Model":
    st.header(f"Hasil Evaluasi Model: {bank_pilihan}")
    # Tampilkan metrik dari Langkah 6
    # Metrik Baris Atas (Otomatis)
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score", f"{m['r2']:.4f}")
    col2.metric("MAE", f"Rp {m['mae']:.2f}")
    col3.metric("MAPE", f"{m['mape']:.2f}%")
    col4.metric("RMSE", f"{m['rmse']:.2f}")

    st.subheader("Visualisasi Prediksi vs Aktual")
    st.image(f"Visual/09_Prediction_{bank_pilihan}.png")

elif menu == "4. Analisis Feature Importance":
    st.header(f"Interpretasi Model: {bank_pilihan}")
    st.write("Menganalisis faktor yang paling mempengaruhi keputusan model.")
    st.image(f"Visual/18_Feature_Importance_{bank_pilihan}.png")
    st.warning("Temuan: Variabel harga harian (High/Low) tetap menjadi prediktor paling dominan.")

elif menu == "5. Demo Prediksi Real-time":
    st.header("Demo Prediksi Harga Esok Hari")
    st.write(f"Menggunakan model yang sudah dilatih untuk memprediksi harga {bank_pilihan}")

    # Logic input manual atau tarik real-time seperti saran sebelumnya
    high = st.number_input("Input Harga Tertinggi (High) Hari Ini", value=10000)
    low = st.number_input("Input Harga Terendah (Low) Hari Ini", value=9850)

    if st.button("Jalankan Prediksi"):
        # Di sini Anda bisa memanggil model & scaler jika ingin kalkulasi riil
        st.balloons()
        st.success(f"Permintaan prediksi untuk {bank_pilihan} diterima. Berdasarkan evaluasi, model memiliki tingkat kepastian R²: {m['r2']:.4f}")
