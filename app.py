import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest

# --- 1. KONFIGURASI HALAMAN ---
st.set_page_config(page_title="Presensi Anomaly Detector", layout="wide")

st.title("🕵️‍♀️ Deteksi Anomali Presensi Pegawai")
st.markdown("""
Aplikasi ini menggunakan **Machine Learning (Isolation Forest)** untuk mendeteksi perubahan pola jam datang pegawai.
Sistem mencari pegawai yang memiliki **volatilitas (ketidakstabilan) tinggi** dibandingkan riwayatnya sendiri atau mayoritas kantor.
""")

# --- 2. FUNGSI GENERATE DUMMY DATA ---
def generate_dummy_data():
    # Buat data dari Januari sampai Maret 2025
    dates = pd.date_range(start='2025-01-01', end='2025-03-31', freq='B')
    data = []
    
    for date in dates:
        # Skenario 1: Budi (Stabil) - Datang 07:45 +/- 5 menit
        noise_budi = np.random.normal(0, 5) 
        data.append(['Budi (Stabil)', date, 7.75 * 60 + noise_budi])
        
        # Skenario 2: Siti (Burnout di Bulan Maret)
        if date.month < 3:
            # Januari-Februari stabil (08:00 +/- 5 menit)
            noise_siti = np.random.normal(0, 5)
        else:
            # Maret mulai kacau (08:00 +/- 45 menit)
            noise_siti = np.random.normal(0, 45) 
        data.append(['Siti (Gejala Burnout)', date, 8.0 * 60 + noise_siti])
        
        # Skenario 3: Ahmad (Agak Siang tapi Rutin) - Datang 08:30 +/- 3 menit
        data.append(['Ahmad (Siang tapi Rutin)', date, 8.5 * 60 + np.random.normal(0, 3)])

    return pd.DataFrame(data, columns=['Nama', 'Tanggal', 'Menit_Kedatangan'])

# --- 3. SIDEBAR (INPUT DATA) ---
st.sidebar.header("Pengaturan Data")
data_source = st.sidebar.radio("Pilih Sumber Data:", ["Gunakan Data Dummy (Demo)", "Upload CSV Sendiri"])

df = None # Variabel penampung data

# LOGIKA SUMBER DATA
if data_source == "Gunakan Data Dummy (Demo)":
    df = generate_dummy_data()
    st.sidebar.success("Data Dummy Berhasil Dimuat!")
    
elif data_source == "Upload CSV Sendiri":
    st.sidebar.info("Pastikan format CSV: Nama, Tanggal (YYYY-MM-DD), Jam_Datang (HH:MM)")
    uploaded_file = st.sidebar.file_uploader("Upload file CSV", type=["csv"])
    
    if uploaded_file is not None:
        try:
            temp_df = pd.read_csv(uploaded_file)
            
            # Validasi Kolom
            required_columns = ['Nama', 'Tanggal', 'Jam_Datang']
            if not all(col in temp_df.columns for col in required_columns):
                st.error(f"Format Kolom Salah! Wajib ada: {', '.join(required_columns)}")
            else:
                # Konversi Data
                temp_df['Tanggal'] = pd.to_datetime(temp_df['Tanggal'])
                
                # Fungsi Helper: Jam String ke Menit Integer
                def jam_ke_menit(jam_str):
                    try:
                        parts = str(jam_str).split(':')
                        return int(parts[0]) * 60 + int(parts[1])
                    except:
                        return None

                temp_df['Menit_Kedatangan'] = temp_df['Jam_Datang'].apply(jam_ke_menit)
                df = temp_df.dropna(subset=['Menit_Kedatangan']) # Hapus data error
                st.sidebar.success(f"Upload Berhasil: {len(df)} baris data.")
                
        except Exception as e:
            st.error(f"Error membaca file: {e}")

# --- 4. PROSES MACHINE LEARNING & DASHBOARD ---
if df is not None:
    # A. Feature Engineering (Persiapan Data)
    # Buat kolom format jam (hh:mm) untuk tampilan tabel
    df['Jam_Format'] = df['Menit_Kedatangan'].apply(lambda x: f"{int(x//60):02d}:{int(x%60):02d}")
    
    # HITUNG VOLATILITAS (CORE ML FEATURE)
    # Menghitung standar deviasi rolling 5 hari
    df['Volatilitas'] = df.groupby('Nama')['Menit_Kedatangan'].transform(lambda x: x.rolling(window=5).std())
    
    # Hapus data awal yang NaN karena rolling window
    df_clean = df.dropna().copy()
    
    st.write("### Preview Data (Siap diproses)")
    st.dataframe(df_clean.head())

    # B. Tombol Eksekusi
    col_btn1, col_btn2 = st.columns([1, 1])
    
    with col_btn1:
        if st.button("Jalankan Analisis Anomali 🚀"):
            
            with st.spinner('Robot sedang menganalisis pola kedatangan...'):
                # --- TRAINING MODEL ---
                # Kita hanya pakai fitur 'Volatilitas' untuk menentukan anomali
                model = IsolationForest(contamination=0.05, random_state=42)
                
                df_clean['Anomaly_Score'] = model.fit_predict(df_clean[['Volatilitas']])
                
                # Mapping hasil: -1 itu Anomali, 1 itu Normal
                df_clean['Status'] = df_clean['Anomaly_Score'].apply(lambda x: 'ANOMALI' if x == -1 else 'Normal')
                
                # Simpan hasil analisis ke session state
                st.session_state['analysis_completed'] = True
                st.session_state['df_analyzed'] = df_clean.copy()
    
    with col_btn2:
        if st.session_state.get('analysis_completed', False):
            if st.button("Reset Analisis 🔄"):
                st.session_state['analysis_completed'] = False
                if 'df_analyzed' in st.session_state:
                    del st.session_state['df_analyzed']
                st.rerun()
    
    # Tampilkan hasil analisis jika sudah ada di session state
    if st.session_state.get('analysis_completed', False):
        df_analyzed = st.session_state['df_analyzed']
        
        # --- VISUALISASI HASIL ---
        st.divider()
        
        # 1. Metrik Ringkasan
        col1, col2 = st.columns(2)
        total_anomali = df_analyzed[df_analyzed['Status'] == 'ANOMALI'].shape[0]
        col1.metric("Total Data Presensi", len(df_analyzed))
        col2.metric("Pola Mencurigakan (Anomali)", total_anomali, delta_color="inverse")
        
        # 2. Grafik Scatter Plot
        st.subheader("Grafik Pola Kedatangan")
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Gambar titik Normal (Transparan)
        sns.scatterplot(data=df_analyzed[df_analyzed['Status'] == 'Normal'], 
                        x='Tanggal', y='Menit_Kedatangan', hue='Nama', style='Nama', alpha=0.5, s=50, ax=ax)
        
        # Gambar titik Anomali (Merah Besar)
        anomali_data = df_analyzed[df_analyzed['Status'] == 'ANOMALI']
        ax.scatter(anomali_data['Tanggal'], anomali_data['Menit_Kedatangan'], 
                   color='red', s=150, label='ANOMALI (High Volatility)', marker='X', zorder=10)
        
        # Rapikan Sumbu Y (Menit ke Jam)
        ax.set_yticks([420, 450, 480, 510, 540])
        ax.set_yticklabels(['07:00', '07:30', '08:00', '08:30', '09:00'])
        ax.set_ylabel("Waktu Kedatangan")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
        
        st.pyplot(fig)
        
        # 3. Cek Status Personal
        st.divider()
        st.subheader("🔍 Cek Status Individu")
        
        selected_user = st.selectbox("Pilih Nama Pegawai:", df_analyzed['Nama'].unique())
        user_data = df_analyzed[df_analyzed['Nama'] == selected_user]
        
        # Cek 7 hari terakhir user tersebut
        last_record = user_data.tail(7)
        is_danger = 'ANOMALI' in last_record['Status'].values
        
        if is_danger:
            st.error(f"⚠️ PERINGATAN: {selected_user} terdeteksi memiliki pola anomali dalam 7 hari kerja terakhir.")
            st.write("Saran: HC/PM disarankan melakukan ngobrol santai.")
        else:
            st.success(f"✅ AMAN: Pola {selected_user} stabil dalam batas wajar.")
        
        with st.expander(f"Lihat Detail Data {selected_user}"):
            st.dataframe(user_data[['Tanggal', 'Jam_Format', 'Volatilitas', 'Status']].sort_values(by='Tanggal', ascending=False))

else:
    st.info("Silakan pilih sumber data di menu sebelah kiri (Sidebar).")