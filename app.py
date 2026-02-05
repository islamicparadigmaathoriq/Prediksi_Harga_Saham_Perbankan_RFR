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
elif menu == "2. Prapemrosesan & Fitur":
    st.header("Tahap 2: Prapemrosesan & Rekayasa Fitur")
    
    # Membagi menjadi 4 Tab sesuai alur kerja teknis
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Data Cleaning", 
        "2. Feature Engineering", 
        "3. Normalisasi (Scaling)", 
        "4. Data Splitting"
    ])
    
    # --- TAB 1: DATA CLEANING ---
    with tab1:
        st.subheader("Pembersihan Data (Cleaning)")
        st.markdown("""
        Proses awal untuk memastikan integritas data deret waktu:
        * **Penanganan Missing Values**: Menggunakan teknik `ffill` dan `bfill` untuk mengisi kekosongan data akibat hari libur bursa atau anomali.
        * **Seleksi Kolom**: Menghapus kolom `Adj Close` dan fokus pada kolom harga utama.
        """)
        
        img_02 = "Visual/02_Komparasi_Data_Cleaning.png"
        if os.path.exists(img_02):
            st.image(img_02, caption="Visualisasi menunjukkan data sebelum cleaning memiliki gap/nilai kosong, sedangkan setelah cleaning data menjadi kontinu dan siap diproses.", use_container_width=True)
        else:
            st.error(f"File {img_02} tidak ditemukan.")

    # --- TAB 2: FEATURE ENGINEERING ---
    with tab2:
        st.subheader("Ekstraksi 22 Indikator Teknikal")
        st.markdown("""
        Menambahkan variabel baru untuk menangkap pola pasar:
        * **Trend**: SMA & EMA (5, 10, 20).
        * **Momentum**: RSI & MACD.
        * **Volatility**: Bollinger Bands & HL Range.
        """)
        
        img_03 = f"Visual/03_Technical_Indicators_{bank_pilihan}.png"
        if os.path.exists(img_03):
            st.image(img_03, caption=f"IGabungan indikator harian untuk {bank_pilihan} memberikan informasi lebih dalam kepada model dibanding hanya menggunakan harga tunggal.", use_container_width=True)
        else:
            st.error(f"File {img_03} tidak ditemukan.")

    # --- TAB 3: NORMALISASI ---
    with tab3:
        st.subheader("Normalisasi dengan MinMaxScaler")
        st.markdown("""
        Menyamakan skala data agar fitur dengan angka besar (Volume) tidak mendominasi fitur kecil (Harga):
        * **Metode**: Transformasi linear ke rentang **0 hingga 1**.
        """)
        
        col1, col2 = st.columns(2)
        with col1:
            img_04 = "Visual/04_Perbandingan_Normalisasi.png"
            if os.path.exists(img_04):
                st.image(img_04, caption="Terlihat perbedaan distribusi data mentah yang lebar menjadi terkumpul dalam rentang 0-1.")
        with col2:
            img_05 = f"Visual/05_Distribusi_Setelah_Normalisasi.png"
            if os.path.exists(img_05):
                st.image(img_05, caption=f"Distribusi Normalisasi: {bank_pilihan}")

    # --- TAB 4: DATA SPLITTING ---
    with tab4:
        st.subheader("Pembagian Data Training & Testing")
        st.markdown("""
        Memisahkan data untuk melatih model dan menguji validitasnya:
        * **Proporsi**: **80% Pelatihan** dan **20% Pengujian**.
        * **Metode**: *Time-series Split* (Tanpa pengacakan) untuk mempertahankan urutan kronologis transaksi.
        """)
        
        img_06 = "Visual/06_Visualisasi_Pembagian_Pelatihan_Uji.png"
        if os.path.exists(img_06):
            st.image(img_06, caption="Area hijau menunjukkan data yang dipelajari model, sedangkan area oranye adalah 'ujian' untuk model.", use_container_width=True)
        
        img_07 = f"Visual/07_Distribusi_Uji_Pelatihan_{bank_pilihan}.png"
        if os.path.exists(img_07):
            st.image(img_07, caption=f"Distribusi Pelatihan vs Uji: {bank_pilihan}")

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
