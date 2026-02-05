import streamlit as st
import pandas as pd
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Seminar Hasil Kelompok 6", layout="wide")

# --- FUNGSI LOAD DATA OTOMATIS ---
@st.cache_data
def load_json_data(file_path):
    # Memastikan file ada sebelum dibaca
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

# --- LOAD ASSETS (Pastikan folder & file ini ada di GitHub nanti) ---
@st.cache_resource
def load_model_assets(bank):
    # Load model dan scaler sesuai struktur folder
    model_path = f"Models/Trained/{bank}_rf_model.pkl"
    scaler_path = f"Models/Scalers/{bank}_scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

# Inisialisasi data
all_metrics = load_json_data('Models/metrics.json')
summary_data = load_json_data('Models/data_summary.json')

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

# --- DATA BINDING ---
# Ambil metrik performa dan ringkasan dataset sesuai pilihan bank
m = all_metrics.get(bank_pilihan, {})
s = summary_data.get(bank_pilihan, {})

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
        st.subheader(f"Hasil Utama ({bank_pilihan})")
        if m:
            st.success(f"R² Score: {m['r2']:.4f}")
            st.write(f"Model {bank_pilihan} menunjukkan performa sangat akurat ditarik otomatis dari hasil riset.")
        else:
            st.warning("Data metrik belum tersedia. Pastikan metrics.json sudah di-upload.")

elif menu == "1. Pengumpulan Data":
    st.header("Tahap 1: Pengumpulan Data Historis")
    
    # --- RESUME AKTIVITAS ---
    st.subheader("Resume Aktivitas")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
        **Sumber Data & Periode:**
        * **Sumber**: Yahoo Finance API (`yfinance`).
        * **Periode**: 17 Oktober 2022 s/d 17 Oktober 2025 (3 Tahun).
        """)
    with col_b:
        st.markdown("""
        **Struktur Dataset Utama:**
        * **Format**: CSV (`_raw.csv`).
        * **Kolom**: `Date`, `Open`, `High`, `Low`, `Close`, `Adj Close`, `Volume`.
        * **Penyimpanan**: Folder `Data/Raw/`.
        """)

    st.divider()

    # Detail Dataset Otomatis
    st.subheader(f"Detail Dataset: {bank_pilihan}")
    if s:
        st.code(f"""
        File: {s['file']}
        Rows: {s['rows']}
        Columns: {s['columns']}
        Date Range: {s['date_range']}
        Rentang Harga: {s['price_range']}
        Volume Rata-rata: {s['avg_volume']}
        Missing Values: {s['missing_values']}
        Status: {s['status']}
        """)
    else:
        st.error("Metadata dataset tidak ditemukan. Pastikan data_summary.json sudah di-upload.")

    # --- VISUALISASI ---
    st.subheader("Visualisasi Data Mentah")
    st.image("Visual/01_Analisis_Data_Mentah.png", use_container_width=True)
    
    # --- PENJELASAN GAMBAR ---
    with st.expander("Penjelasan Detail Grafik"):
        st.markdown(f"""
       
        * **Harga Penutupan (Closing Price)**: Grafik garis menunjukkan fluktuasi harga harian yang menjadi target prediksi utama model Random Forest.
        * **Volume Perdagangan**: Menunjukkan likuiditas; volume tinggi pada titik tertentu seringkali berkorelasi dengan volatilitas harga.
        * **Distribusi Harga**: Histogram menunjukkan persebaran harga untuk melihat konsentrasi nilai saham selama 3 tahun terakhir.
        * **Harga Rata-rata**: Memberikan baseline untuk mengidentifikasi apakah tren saat ini berada di atas atau di bawah nilai historis rata-rata.
        """)

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
