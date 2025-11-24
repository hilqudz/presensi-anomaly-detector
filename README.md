# 🕵️‍♀️ Presensi Anomaly Detector

Aplikasi Machine Learning untuk mendeteksi anomali pola kedatangan pegawai menggunakan **Isolation Forest Algorithm**.

## 📋 Fitur Utama

- **🤖 Machine Learning**: Deteksi anomali otomatis menggunakan Isolation Forest
- **📊 Visualisasi Interaktif**: Grafik scatter plot dengan matplotlib dan seaborn  
- **📈 Volatilitas Analysis**: Analisis pola kedatangan per individu
- **🔍 Individual Check**: Pemeriksaan status anomali per pegawai
- **📤 File Upload**: Support upload CSV custom atau gunakan data dummy
- **🎯 Real-time Results**: Analisis dan hasil langsung

## 🚀 Cara Menjalankan

### 1. Clone Repository
```bash
git clone https://github.com/hilqudz/presensi-anomaly-detector.git
cd presensi-anomaly-detector
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Jalankan Aplikasi
```bash
streamlit run app.py
```

Aplikasi akan terbuka di browser pada `http://localhost:8501`

## 📁 Format Data CSV

Untuk testing, Anda bisa download file `sample_data.csv` yang sudah tersedia di repository ini.

Atau jika ingin upload data sendiri, pastikan format CSV seperti ini:

```csv
Nama,Tanggal,Jam_Datang
John Doe,2025-01-01,07:45
Jane Smith,2025-01-01,08:30
John Doe,2025-01-02,07:50
Jane Smith,2025-01-02,08:25
```

**Requirements:**
- Kolom wajib: `Nama`, `Tanggal` (YYYY-MM-DD), `Jam_Datang` (HH:MM)
- Data hanya hari kerja (Senin-Jumat)
- Minimal 5 hari data per pegawai untuk analisis rolling

## 🧮 Cara Kerja Algorithm

1. **Feature Engineering**: Menghitung volatilitas (standar deviasi) rolling 5 hari per pegawai
2. **Machine Learning**: Isolation Forest mendeteksi outlier berdasarkan volatilitas
3. **Anomaly Detection**: Pegawai dengan pola kedatangan tidak konsisten terdeteksi sebagai anomali
4. **Visualization**: Hasil ditampilkan dalam bentuk grafik dan metrik

## 🎯 Use Case

- **HR Department**: Monitor pola kedatangan pegawai
- **Team Lead**: Early warning untuk burnout atau masalah pribadi
- **Management**: Analisis produktivitas dan kesejahteraan tim

## 🛠️ Tech Stack

- **Frontend**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Machine Learning**: Scikit-learn (Isolation Forest)
- **Visualization**: Matplotlib, Seaborn
- **Language**: Python 3.8+

## 📊 Demo Data

Aplikasi menyediakan data demo dengan 3 skenario:
- **Budi**: Pegawai stabil (07:45 ± 5 menit)
- **Ahmad**: Pegawai konsisten tapi agak siang (08:30 ± 3 menit)  
- **Siti**: Pegawai dengan gejala burnout (volatilitas tinggi di bulan Maret)

## 📝 License

MIT License - silakan digunakan untuk keperluan komersial maupun personal.

## 🤝 Contributing

Pull requests welcome! Untuk perubahan besar, silakan buka issue terlebih dahulu.

---

Made with ❤️ for better workplace wellness monitoring