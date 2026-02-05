# Prediksi Harga Saham Perbankan Indonesia - Random Forest Regression
**Proyek Tugas Kecerdasan Buatan - Kelompok 6 (Universitas Siber Asia)**

Aplikasi ini menggunakan algoritma **Random Forest Regression** untuk memprediksi harga saham harian dari lima (5) bank terbesar di Indonesia (BCA, BRI, BNI, Mandiri, dan BTN).

## Fitur Utama
- **Dashboard Interaktif**: Visualisasi alur penelitian dari data mentah hingga prediksi.
- **Prediksi Real-time**: Integrasi dengan Yahoo Finance API untuk mendapatkan harga saham terkini.
- **Analisis Feature Importance**: Penjelasan variabel apa yang paling memengaruhi harga saham.

## Struktur Folder
- `app.py`: Aplikasi utama berbasis Streamlit.
- `Models/Trained/`: Model Random Forest (.pkl) yang sudah dilatih dengan akurasi R² > 0.85.
- `Models/Scalers/`: Objek normalisasi MinMaxScaler untuk setiap bank.
- `Visual/`: Koleksi grafik hasil evaluasi dan analisis data.
- `Data/Raw/`: Dataset historis periode 2022-2025.

## Cara Menjalankan Secara Lokal
1. Clone repositori ini.
2. Instal dependensi: `pip install -r requirements.txt`.
3. Jalankan aplikasi: `streamlit run app.py`.

## Anggota Kelompok 6
- Fikri Amrullah Sya’bani
- Islamic Paradigma Athoriq
- Rifki Fajar Rudin
- Sopia Rita Meliala
- Taufik Hidayat
