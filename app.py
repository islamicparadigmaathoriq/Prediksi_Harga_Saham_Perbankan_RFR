import streamlit as st
import pandas as pd
import joblib
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime

# --- KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Seminar Hasil Kelompok 6", layout="wide")

# --- LOAD ASSETS (Pastikan folder & file ini ada di GitHub nanti) ---
@st.cache_resource
def load_model_and_results(bank):
    # Load model, scaler, dan hasil evaluasi yang sudah disimpan di Colab
    model = joblib.load(f"Models/Trained/{bank}_rf_model.pkl")
    scaler = joblib.load(f"Scalers/{bank}_scaler.pkl")
    # Load data hasil prediksi (Langkah 5) untuk visualisasi historis
    # df_preds = pd.read_csv(f"Results/{bank}_predictions.csv") 
    return model, scaler

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
        st.subheader("Hasil Utama")
        st.success("Rata-rata R² Score mencapai 0.965 (Sangat Akurat)")

elif menu == "1. Pengumpulan Data":
    st.header("Tahap 1: Pengumpulan Data Historis")
    st.write("Data ditarik dari Yahoo Finance API (17 Okt 2022 - 17 Okt 2025).")
    # menampilkan grafik harga historis dari CSV asli Anda
    st.info(f"Menampilkan Data Historis untuk {bank_pilihan}")
    # Placeholder untuk grafik harga asli
    st.image(f"visuals/01_Stock_Price_Trend.png") # Gambar dari Langkah 1

elif menu == "2. Prapemrosesan & Fitur":
    st.header("Tahap 2 & 3: Preprocessing & Feature Engineering")
    st.markdown("""
    **Proses yang dilakukan:**
    1. **Cleaning**: Penanganan missing values dengan *ffill* & *bfill*.
    2. **Feature Engineering**: Pembuatan **22 Fitur Teknikal** (SMA, EMA, RSI, MACD, Bollinger Bands).
    3. **Normalisasi**: Menggunakan *MinMaxScaler* (Range 0-1).
    4. **Splitting**: Rasio 80:20 (Time-series split, tanpa shuffle).
    """)
    st.table(pd.DataFrame({
        'Kategori': ['Trend', 'Momentum', 'Volatility', 'Volume'],
        'Fitur': ['SMA (5,10,20), EMA (5,10,20)', 'RSI, MACD', 'Bollinger Bands, HL Range', 'Volume MA']
    }))

elif menu == "3. Evaluasi Performa Model":
    st.header("Tahap 4 & 6: Hasil Pelatihan & Evaluasi")
    # Tampilkan metrik dari Langkah 6
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("R² Score (Test)", "0.965")
    col2.metric("MAE", "Rp 85.20")
    col3.metric("MAPE", "1.2%")
    col4.metric("RMSE", "112.4")
    
    st.subheader("Visualisasi Prediksi vs Aktual")
    # Tampilkan gambar plot hasil dari Langkah 5
    st.image(f"visuals/09_Prediction_{bank_pilihan}.png")

elif menu == "4. Analisis Feature Importance":
    st.header("Tahap 7: Interpretasi Model (Gini Importance)")
    st.write("Menganalisis faktor yang paling mempengaruhi keputusan model.")
    # Tampilkan gambar analisis importance dari Langkah 7
    st.image(f"visuals/18_Feature_Importance_{bank_pilihan}.png")
    st.warning("Temuan: Fitur High dan Low mendominasi pengaruh sebesar 72.6%.")

elif menu == "5. Demo Prediksi Real-time":
    st.header("Demo Prediksi Harga Esok Hari")
    st.write(f"Menggunakan model yang sudah dilatih untuk memprediksi harga {bank_pilihan}")
    
    # Logic input manual atau tarik real-time seperti saran sebelumnya
    high = st.number_input("Input Harga Tertinggi (High) Hari Ini", value=10000)
    low = st.number_input("Input Harga Terendah (Low) Hari Ini", value=9850)
    
    if st.button("Jalankan Prediksi"):
        # Load model dan lakukan prediksi
        st.success(f"Prediksi Harga Penutupan {bank_pilihan} Besok: Rp ... (Hasil Prediksi)")
