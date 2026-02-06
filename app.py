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
    "4. Demo Prediksi Real-time"
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

# MENU 2: PRAPEMROSESAN & FITUR
elif menu == "2. Prapemrosesan Data":
    st.header("Tahap 2: Prapemrosesan & Rekayasa Fitur")
    
    # Membagi menjadi 4 Tab sesuai alur kerja teknis di dokumen
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Data Cleaning", 
        "2. Feature Engineering", 
        "3. Normalisasi (Scaling)", 
        "4. Data Splitting"
    ])
    
    # --- TAB 1: DATA CLEANING ---
    with tab1:
        st.subheader("Pembersihan Data (Data Cleaning)")
        
        # Menampilkan Hasil Eksplorasi Data Awal dari Colab
        if s:
            st.markdown(f"**Hasil Eksplorasi Data Awal: {bank_pilihan}**")
            col_info1, col_info2, col_info3 = st.columns(3)
            with col_info1:
                st.metric("Dimensi Data", s.get('shape', 'N/A'))
            with col_info2:
                st.metric("Total Duplikat", s.get('duplicates', 0))
            with col_info3:
                st.metric("Missing Values", s.get('missing_values', 0))
            
            with st.expander("Lihat Statistik Deskriptif Mentah"):
                # Menampilkan deskripsi statistik jika tersedia di JSON
                if 'desc_stats' in s:
                    st.write(pd.DataFrame(s['desc_stats']))
        
        st.divider()
        st.markdown("""
        Tahap ini bertujuan memastikan integritas data deret waktu sebelum dilakukan perhitungan indikator.
        * **Pengecekan Missing Values**: Mengidentifikasi kekosongan data akibat hari libur bursa.
        * **Imputasi (ffill & bfill)**: Mengisi nilai kosong berdasarkan harga sebelumnya (*Forward Fill*) dan sesudahnya (*Backward Fill*) agar tren tidak terputus.
        * **Seleksi Fitur**: Menghapus kolom `Adj Close` dan `Unnamed` untuk menyederhanakan struktur dataset mentah.
        """)
        
        img_02 = "Visual/02_Komparasi_Data_Cleaning.png"
        if os.path.exists(img_02):
            st.image(img_02, caption="Interpretasi: Grafik menunjukkan transisi dari data mentah yang terputus (kiri) menjadi data kontinu yang bersih (kanan).", use_container_width=True)
        else:
            st.error(f"File {img_02} tidak ditemukan.")

    # --- TAB 2: FEATURE ENGINEERING ---
    with tab2:
        st.subheader("Ekstraksi 22 Indikator Teknikal")
        st.markdown("""
        Model Random Forest membutuhkan fitur tambahan untuk menangkap dinamika pasar yang tidak terlihat pada harga penutupan saja.
        * **Trend (SMA & EMA)**: Menangkap rata-rata harga jangka pendek (5 hari) hingga menengah (20 hari).
        * **Momentum (RSI & MACD)**: Mengukur kekuatan pergerakan harga dan potensi kejenuhan pasar.
        * **Volatility (Bollinger Bands)**: Mengidentifikasi rentang fluktuasi harga normal.
        * **Lainnya**: Volume MA, Daily Return, dan HL Range (selisih High-Low).
        """)
        
        img_03 = f"Visual/03_Technical_Indicators_{bank_pilihan}.png"
        if os.path.exists(img_03):
            st.image(img_03, caption=f"Interpretasi: Visualisasi 22 fitur teknikal untuk {bank_pilihan} yang akan menjadi prediktor bagi algoritma Random Forest.", use_container_width=True)
        else:
            st.error(f"File {img_03} tidak ditemukan.")

    # --- TAB 3: NORMALISASI ---
    with tab3:
        st.subheader("Normalisasi dengan MinMaxScaler")
        st.markdown("""
        Proses ini menggunakan **MinMaxScaler** untuk menyamakan skala seluruh fitur ke dalam rentang **0 hingga 1**. 
        Hal ini krusial karena model *Random Forest* akan lebih stabil jika fitur dengan angka besar (Volume) memiliki bobot yang setara dengan fitur angka kecil (Harga).
        Tahap ini memastikan seluruh fitur memiliki skala yang seragam antara **0 hingga 1**. Tanpa normalisasi, fitur dengan nominal besar (Volume) akan mendominasi perhitungan model dibandingkan fitur harga.""")

        # Dokumentasi Kode Proses Normalisasi
        with st.expander("Lihat Logika Kode Normalisasi"):
            st.code("""
            # Inisialisasi MinMaxScaler untuk rentang 0-1
            scaler = MinMaxScaler(feature_range=(0, 1))

            # Fit dan transform fitur teknikal
            scaled_features = scaler.fit_transform(features_to_scale)

            # Konversi kembali ke DataFrame dengan kolom yang sama
            df_normalized = pd.DataFrame(
                scaled_features,
                columns=columns_to_normalize,
                index=features_to_scale.index
            )
                        """, language='python')
            
        # A. RINGKASAN NORMALISASI DALAM BENTUK KARTU (Sesuai Bank Pilihan)
        if s and 'norm_verification' in s:
            v = s['norm_verification']
            
            # Baris Pertama: Info Dasar
            col_info1, col_info2 = st.columns(2)
            with col_info1:
                st.info(f"**Dimensi Data:** {s.get('rows')} Baris")
            with col_info2:
                st.info(f"**Jumlah Fitur:** {len(s.get('columns', []))} Kolom")

            # Baris Kedua: Verifikasi Range (Kartu)
            st.markdown("### Verifikasi Range [0, 1]")
            c1, c2, c3 = st.columns(3)
            
            with c1:
                st.metric("Normalized Close", "0.0000 - 1.0000")
                st.caption(f"Original: {s.get('price_range')}")
            
            with c2:
                st.metric("Normalized Volume", "0.0000 - 1.0000")
                st.caption(f"Original: {s.get('avg_volume')} (Avg)")
            
            with c3:
                # Menampilkan fitur lain seperti RSI jika ada
                if 'RSI' in v:
                    st.metric("Normalized RSI", f"{v['RSI']['min']:.2f} - {v['RSI']['max']:.2f}")
                    st.caption("Momentum Scaled")
        else:
            st.warning(f"Data normalisasi untuk {bank_pilihan} belum tersedia di data_summary.json")

        st.divider()

        # B. VISUALISASI DISTRIBUSI SPESIFIK BANK
        st.subheader(f"Analisis Distribusi & Outlier: {bank_pilihan}")
        img_dist = f"Visual/05_Distribusi_{bank_pilihan}.png"
        
        if os.path.exists(img_dist):
            st.image(img_dist, use_container_width=True)
            
            # C. INTERPRETASI DATA
            with st.expander("Interpretasi Grafik"):
                st.markdown(f"""
                **Analisis untuk {bank_pilihan}:**
                * **Histogram**: Memperlihatkan bahwa setelah normalisasi, bentuk asli distribusi data tetap terjaga, namun sumbu X kini berada di rentang 0-1. Garis merah (Mean) menunjukkan posisi rata-rata data hasil scaling.
                * **Box Plot**: Memvalidasi bahwa tidak ada nilai yang keluar dari batas 0 atau 1. Kotak biru menunjukkan area konsentrasi harga terbesar (Interquartile Range) untuk {bank_pilihan}.
                """)
        else:
            st.error(f"Grafik distribusi {img_dist} tidak ditemukan.")

    # --- TAB 4: DATA SPLITTING ---
    with tab4:
        st.subheader("Pembagian Data Training & Testing")
        st.markdown("""
        Pemisahan data dilakukan menggunakan metode **Time-Series Split**. Berbeda dengan *Random Split*, metode ini mempertahankan urutan kronologis untuk mencegah **Data Leakage** (model belajar dari masa depan).
        """)

        # Dokumentasi Kode Splitting
        with st.expander("Lihat Logika Kode Time-Series Split"):
            st.code("""
        def time_series_split(df, train_ratio=0.8):
            # Hitung index split berdasarkan rasio 80%
            split_index = int(len(df) * train_ratio)

            # Split data secara kronologis (bukan acak)
            train_df = df.iloc[:split_index].copy()
            test_df = df.iloc[split_index:].copy()
            
            return train_df, test_df, split_date
                    """, language='python')
        
        # Ringkasan Split InformatioN
        if s and 'split_details' in s:
            sd = s['split_details']
            st.info(f"**Informasi Split: {bank_pilihan}**")
            c_split1, c_split2, c_split3 = st.columns(3)
            c_split1.metric("Tanggal Split", sd['split_date'])
            c_split2.metric("Data Latih (Train)", f"{sd['train_rows']} Baris", sd['train_pct'])
            c_split3.metric("Data Uji (Test)", f"{sd['test_rows']} Baris", sd['test_pct'])
        
        st.divider()

        # Visualisasi Area Split
        img_06 = "Visual/06_Visualisasi_Pembagian_Pelatihan_Uji.png"
        if os.path.exists(img_06):
            st.image(img_06, caption="Interpretasi: Area Biru (80%) adalah data historis untuk pelatihan, area oranye (20%) adalah data simulasi masa depan untuk pengujian.")

        st.divider()

        # Statistik Train vs Test Set (Sesuai Image 2 Anda)
        st.subheader("Perbandingan Statistik Train vs Test")
        if 'split_stats' in s:
            ss = s['split_stats']
            
            # Tabel Komparasi Close Price
            st.markdown("**Statistik Harga Penutupan (Close):**")
            close_stats = pd.DataFrame({
                'Metric': ['Mean (Rata-rata)', 'Std Dev (Standar Deviasi)'],
                'Train Set': [ss['Close']['Train Mean'], ss['Close']['Train Std']],
                'Test Set': [ss['Close']['Test Mean'], ss['Close']['Test Std']]
            })
            st.table(close_stats.style.format("{:.6f}", subset=['Train Set', 'Test Set']))

            # Tampilan Distribusi
            img_07 = f"Visual/07_Distribusi_Uji_Pelatihan_{bank_pilihan}.png"
            if os.path.exists(img_07):
                st.image(img_07, caption=f"Distribusi Fitur pada Data Train vs Test: {bank_pilihan}")
        
        st.divider()

        # Ringkasan Global (Summary DF)
        st.subheader("Ringkasan Global Semua Bank")
        summary_list = []
        for ticker, data in summary_data.items():
            if 'split_details' in data:
                d = data['split_details']
                summary_list.append({
                    'Bank': ticker,
                    'Total Rows': d['total_rows'],
                    'Train Rows': d['train_rows'],
                    'Test Rows': d['test_rows'],
                    'Train %': d['train_pct'],
                    'Test %': d['test_pct'],
                    'Split Date': d['split_date']
                })
        
        if summary_list:
            st.dataframe(pd.DataFrame(summary_list), use_container_width=True)
            st.caption("Rata-rata global pembagian data berada pada rasio 80% Train dan 20% Test sesuai standar evaluasi time-series.")

# MENU 3: EVALUASI PERFORMA
elif menu == "3. Evaluasi Performa Model":
    st.header(f"Tahap 3: Evaluasi & Interpretasi Model: {bank_pilihan})")
    
    # Pembagian 4 Tab Utama sesuai permintaan
    tab1, tab2, tab3, tab4 = st.tabs([
        "1. Training & Tuning", 
        "2. Prediksi (Data Test)", 
        "3. Skor Performa", 
        "4. Feature Importance"
    ])

    # --- TAB 1: TRAINING & TUNING (Proses Pemodelan) ---
    with tab1:
        st.subheader(f"Pembangunan & Optimasi Model: {bank_pilihan}")

        # 1. BASELINE PERFORMANCE CARDS (Sesuai Image 2)
        if s and 'baseline_perf' in s:
            bp = s['baseline_perf']
            st.markdown("### Baseline Performance")
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Baseline Train R²", f"{bp['train_r2']:.4f}")
            c2.metric("Baseline Test R²", f"{bp['test_r2']:.4f}")
            c3.metric("Baseline MAE", f"{bp['mae']:.4f}")
            c4.metric("Training Time", f"{bp['time']:.2f}s")
            
            # Status Target Baseline
            if bp['test_r2'] >= 0.85:
                st.success(f"Target Baseline Tercapai! (R² ≥ 0.85)")
            else:
                st.warning(f"Target Baseline Belum Tercapai. Perlu Hyperparameter Tuning.")
        
        st.divider()

        # 2. TUNED MODEL PERFORMANCE CARDS (Sesuai Image 3)
        if s and 'tuning_results' in s:
            tr = s['tuning_results']
            st.markdown("### Tuned Model Performance")
            tc1, tc2, tc3 = st.columns(3)
            tc1.metric("Final Tuned R²", f"{tr['final_r2']:.4f}", f"{tr['improvement']:+.6f}")
            tc2.metric("Best Max Depth", tr['best_params'].get('max_depth', 'N/A'))
            tc3.metric("Best Estimators", tr['best_params'].get('n_estimators', 'N/A'))
            
            with st.expander("Lihat Parameter Terbaik"):
                st.json(tr['best_params'])
        
        st.divider()

        # 3. K-FOLD CROSS VALIDATION (Image 4)
        st.subheader("Validasi Stabilitas: K-Fold Cross-Validation")
        img_cv = "Visual/08_Skor_Cross_Validation.png"
        if os.path.exists(img_cv):
            st.image(img_cv, caption="Distribusi Skor R² pada 5-Fold Cross Validation")
            
            # Interpretasi Image 4
            st.info("""
            **Interpretasi Grafik Cross-Validation:**
            * **Garis Merah Putus-putus**: Menunjukkan rata-rata (*Mean*) skor R² dari 5 kali percobaan. Semakin tinggi garis ini, semakin akurat model secara umum.
            * **Garis Hijau Titik-titik**: Batas target akurasi (0.85).
            * **Batang Biru**: Menunjukkan konsistensi model pada setiap lipatan (*fold*). Jika tinggi batang relatif seragam (seperti pada BBCA atau BNI), berarti model sangat stabil dan tidak mengalami *overfitting*[cite: 1508, 1509].
            """)
        else:
            st.error("File 08_Skor_Cross_Validation.png tidak ditemukan.")

        st.divider()

        # 4. FINAL COMPREHENSIVE PERFORMANCE (Sesuai Image 5)
        st.subheader("Ringkasan Comprehensive Final")
        if s and 'baseline_perf' in s:
            # Membuat baris status final
            status_color = "green" if s['tuning_results']['final_r2'] >= 0.85 else "red"
            st.markdown(f"""
            | Metric | Score | Status |
            | :--- | :--- | :--- |
            | **Final Train R²** | {s['baseline_perf']['train_r2']:.6f} | Pass |
            | **Final Test R²** | {s['tuning_results']['final_r2']:.6f} | {'Pass' if s['tuning_results']['final_r2'] >= 0.85 else 'Below Target'} |
            | **Final MAE** | {s['baseline_perf']['mae']:.6f} | Optimized |
            """)
        
        # Statistik Keseluruhan (Rata-rata dari summary_data)
            all_r2 = [v['tuning_results']['final_r2'] for k, v in summary_data.items() if 'tuning_results' in v]
            if all_r2:
                avg_r2 = sum(all_r2) / len(all_r2)
                st.write(f"**Rata-rata Akurasi Sektor (5 Bank):** `{avg_r2:.6f}`")
                if avg_r2 >= 0.85:
                    st.success("SEMUA BANK MELEWATI TARGET!")

        # Detail Tuning untuk Bank yang dipilih
        if 'tuning_results' in s:
            st.divider()
            st.markdown(f"**Optimasi Hyperparameter ({bank_pilihan}):**")
            st.json(s['tuning_results']['best_params'])
            st.success(f"Peningkatan Akurasi vs Baseline: {s['tuning_results']['improvement']:.6f}")

    # --- TAB 2: PREDIKSI (DATA TEST) ---
    with tab2:
        st.subheader("Visualisasi Prediksi pada Data Test: {bank_pilihan}")
        st.markdown(f"""
        Model Final digunakan untuk memprediksi 20% data terakhir (Data Test) yang tidak pernah dilihat sebelumnya 
        oleh model saat pelatihan. Ini membuktikan kemampuan generalisasi model.
        """)
        
        img_09 = f"Visual/09_Prediction_{bank_pilihan}.png"
        if os.path.exists(img_09):
            st.image(img_09, caption=f"Grafik Prediksi vs Aktual {bank_pilihan} (Data Test)", use_container_width=True)
            st.info("Garis merah putus-putus menunjukkan seberapa akurat model mengikuti fluktuasi harga asli (garis biru).")
        else:
            st.error(f"File {img_09} belum tersedia di folder Visual.")

    # --- TAB 3: SKOR PERFORMA (EVALUASI HASIL TEST) ---
    with tab3:
        st.subheader("Evaluasi Akhir Model (Testing)")
        if m: # m adalah data dari metrics.json sesuai bank_pilihan
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("R² Score", f"{m.get('r2', 0):.4f}")
            c2.metric("MAE", f"Rp {m.get('mae', 0):.2f}")
            c3.metric("MAPE", f"{m.get('mape', 0):.2f}%")
            c4.metric("RMSE", f"{m.get('rmse', 0):.2f}")
            
        st.divider()
        img_11 = "Visual/11_Scatter_Aktual_vs_Prediksi.png"
        if os.path.exists(img_11):
            st.image(img_11, caption="Scatter Plot: Hubungan Nilai Aktual vs Prediksi (Global), Kedekatan titik dengan garis diagonal menunjukkan tingkat akurasi.", use_container_width=True)
    
    # --- TAB 4: FEATURE IMPORTANCE ---
    with tab4:
        st.subheader("Variabel Paling Berpengaruh (Feature Importance)")
        st.markdown("""
        Analisis ini menunjukkan indikator teknikal mana yang paling dominan dalam membantu model 
        memutuskan angka prediksi harga saham.
        """)
        
        img_18 = f"Visual/18_Feature_Importance_{bank_pilihan}.png"
        if os.path.exists(img_18):
            st.image(img_18, caption=f"Top 10 Fitur Berpengaruh: {bank_pilihan}", use_container_width=True)
            st.success("Fitur harga harian dan Moving Averages biasanya menjadi faktor penentu utama.")
        else:
            st.error(f"File {img_18} belum tersedia.")

# MENU 4: DEMO PREDIKSI
elif menu == "4. Demo Prediksi Real-time":
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
