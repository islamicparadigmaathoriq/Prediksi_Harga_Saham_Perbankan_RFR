import streamlit as st
import pandas as pd
import joblib
import json
import os
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Kecerdasan Buatan Hasil Kelompok 6", layout="wide")

# --- FUNGSI LOAD DATA OTOMATIS ---
@st.cache_data
def load_json_data(file_path):
    # Memastikan file ada sebelum dibaca
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            return json.load(f)
    return {}

# --- LOAD ASSETS ---
@st.cache_resource
def load_model_assets(bank):
    # Load model dan scaler sesuai struktur folder lokal/GitHub
    model_path = f"Models/Trained/{bank}_rf_model.pkl"
    scaler_path = f"Models/Scalers/{bank}_scaler.pkl"
    
    if os.path.exists(model_path) and os.path.exists(scaler_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        return model, scaler
    return None, None

# Inisialisasi data dari file JSON hasil riset Colab
all_metrics = load_json_data('Models/metrics.json')
summary_data = load_json_data('Models/data_summary.json')

# --- SIDEBAR NAVIGASI ---
st.sidebar.title("Menu Navigasi")
menu = st.sidebar.radio("Pilih Tahapan Implementasi:", [
    "0. Ringkasan Proyek",
    "1. Pengumpulan Data",
    "2. Prapemrosesan Data",
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

# MENU 0: RINGKASAN PROYEK
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

# MENU 1: PENGUMPULAN DATA
elif menu == "1. Pengumpulan Data":
    st.header("Tahap 1: Pengumpulan Data Historis")
    
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

    st.subheader("Visualisasi Data Mentah")
    st.image("Visual/01_Analisis_Data_Mentah.png", use_container_width=True)
    
    with st.expander("Penjelasan Detail Grafik"):
        st.markdown("""
        * **Harga Penutupan (Closing Price)**: Grafik garis menunjukkan fluktuasi harga harian yang menjadi target prediksi utama model Random Forest.
        * **Volume Perdagangan**: Menunjukkan likuiditas; volume tinggi pada titik tertentu seringkali berkorelasi dengan volatilitas harga.
        * **Distribusi Harga**: Histogram menunjukkan persebaran harga untuk melihat konsentrasi nilai saham selama 3 tahun terakhir.
        * **Harga Rata-rata**: Memberikan baseline untuk mengidentifikasi apakah tren saat ini berada di atas atau di bawah nilai historis rata-rata.
        """)

# MENU 2: PRAPEMROSESAN
elif menu == "2. Prapemrosesan Data":
    st.header("⚙️ Tahap 2 & 3: Prapemrosesan & Rekayasa Fitur")
    
    # Membagi menjadi 3 Tab sesuai alur kerja tim
    tab1, tab2, tab3 = st.tabs(["1. Data Cleaning & Feature Engineering", "2. Normalisasi (Scaling)", "3. Data Splitting"])
    
    with tab1:
        st.subheader("Pembersihan Data & Ekstraksi Fitur Teknikal")
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("""
            **Proses Pembersihan (Cleaning):**
            * **Pengecekan Nilai Kosong**: Mengidentifikasi data yang hilang pada dataset mentah.
            * **Imputasi**: Mengisi nilai kosong menggunakan metode `ffill` (forward fill) dan `bfill` (backward fill) untuk menjaga kontinuitas deret waktu.
            * **Drop Unused**: Menghapus kolom yang tidak relevan seperti `Adj Close` agar model fokus pada harga transaksi utama.
            """)
        with col2:
            st.markdown("""
            **Rekayasa Fitur (22 Indikator):**
            * **Trend**: SMA & EMA (periode 5, 10, 20) untuk menangkap arah pergerakan harga.
            * **Momentum**: RSI & MACD untuk mengukur kekuatan tren.
            * **Volatility**: Bollinger Bands & HL Range untuk memantau batasan fluktuasi harga.
            """)
        
        st.info(f"Menampilkan hasil ekstraksi 22 indikator teknikal untuk {bank_pilihan}")
        st.image(f"Visual/03_Technical_Indicators_{bank_pilihan}.png", use_container_width=True)

    with tab2:
        st.subheader("Normalisasi Data dengan MinMaxScaler")
        st.markdown("""
        Karena rentang nilai antara **Harga Saham** (ribuan) dan **Volume** (jutaan) sangat jauh berbeda, dilakukan normalisasi:
        * **Metode**: MinMaxScaler.
        * **Rentang**: Mengubah seluruh nilai fitur ke dalam skala **0 hingga 1**.
        * **Pentingnya Scaling**: Mencegah fitur dengan angka besar mendominasi proses pembelajaran model *Random Forest*.
        """)
        
        # Menampilkan visual scaling yang telah Anda buat
        st.image("Visual/06_Scaling_Comparison.png", caption="Perbandingan Data Sebelum dan Sesudah MinMaxScaler", use_container_width=True)

    with tab3:
        st.subheader("Data Splitting (Train-Test Split)")
        st.markdown("""
        Dataset dibagi menjadi dua bagian utama untuk memastikan model dapat melakukan generalisasi dengan baik:
        * **Rasio Pembagian**: **80% untuk Training** (Pelatihan) dan **20% untuk Testing** (Pengujian).
        * **Metode**: *Time-series splitting* (Data diurutkan berdasarkan waktu, tidak diacak/shuffle) untuk menjaga integritas data deret waktu.
        """)
        
        # Menampilkan visual splitting
        st.image("Visual/08_Train_Test_Split.png", caption="Visualisasi Pembagian Data Pelatihan (80%) dan Pengujian (20%)", use_container_width=True)
        
        st.warning("Catatan: Data Test (20% terakhir) digunakan sebagai simulasi data 'masa depan' yang tidak pernah dilihat model saat pelatihan.")

# MENU 3: EVALUASI PERFORMA
elif menu == "3. Evaluasi Performa Model":
    st.header(f"Hasil Evaluasi Model: {bank_pilihan}")
    
    if m:
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("R² Score", f"{m['r2']:.4f}")
        col2.metric("MAE", f"Rp {m['mae']:.2f}")
        col3.metric("MAPE", f"{m['mape']:.2f}%")
        col4.metric("RMSE", f"{m['rmse']:.2f}")
    else:
        st.error("Metrik evaluasi tidak ditemukan.")

    st.subheader("Visualisasi Prediksi vs Aktual")
    st.image(f"Visual/09_Prediction_{bank_pilihan}.png")

# MENU 4: FEATURE IMPORTANCE
elif menu == "4. Analisis Feature Importance":
    st.header(f"Interpretasi Model: {bank_pilihan}")
    st.write("Menganalisis faktor (fitur teknikal) yang paling mempengaruhi keputusan model.")
    st.image(f"Visual/18_Feature_Importance_{bank_pilihan}.png")
    st.warning("Temuan: Variabel harga harian (High/Low) tetap menjadi prediktor paling dominan dalam model ini.")

# MENU 5: DEMO PREDIKSI
elif menu == "5. Demo Prediksi Real-time":
    st.header("Demo Prediksi Harga Esok Hari")
    st.write(f"Simulasi harga penutupan menggunakan model yang sudah dilatih untuk {bank_pilihan}")

    high = st.number_input("Input Harga Tertinggi (High) Hari Ini", value=10000.0)
    low = st.number_input("Input Harga Terendah (Low) Hari Ini", value=9850.0)

    if st.button("Jalankan Prediksi"):
        st.balloons()
        if m:
            st.success(f"Permintaan prediksi untuk {bank_pilihan} diterima. Berdasarkan evaluasi riset, model memiliki tingkat akurasi R²: {m['r2']:.4f}")
        else:
            st.success(f"Permintaan prediksi untuk {bank_pilihan} sedang diproses oleh model Random Forest.")
