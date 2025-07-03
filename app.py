import streamlit as st
import pandas as pd
import joblib
import numpy as np

# Load model dan scaler
try:
    model = joblib.load('Random_Forest_tuned_model.joblib')
    target_encoder = joblib.load('target_label_encoder.joblib')
    scaler = joblib.load('scaler.joblib')
except FileNotFoundError:
    st.error("❌ File model atau scaler tidak ditemukan. Pastikan semua file berada dalam direktori yang sama.")
    st.stop()

# Fitur yang digunakan dalam model
feature_order = [
    'Age', 'Gender', 'Height', 'Weight', 'CALC', 'FAVC', 'FCVC', 'NCP',
    'SCC', 'SMOKE', 'CH2O', 'family_history_with_overweight',
    'FAF', 'TUE', 'CAEC', 'MTRANS'
]

# Mapping opsi Bahasa Indonesia
gender_map = {0: 'Perempuan', 1: 'Laki-laki'}
calc_map = {0: 'Selalu', 1: 'Sering', 2: 'Kadang-kadang', 3: 'Tidak Pernah'}
favc_map = {0: 'Tidak', 1: 'Ya'}
scc_map = {0: 'Tidak', 1: 'Ya'}
smoke_map = {0: 'Tidak', 1: 'Ya'}
family_history_map = {0: 'Tidak', 1: 'Ya'}
caec_map = {0: 'Selalu', 1: 'Sering', 2: 'Kadang-kadang', 3: 'Tidak Pernah'}
mtrans_map = {
    0: 'Mobil',
    1: 'Sepeda',
    2: 'Motor',
    3: 'Transportasi Umum',
    4: 'Jalan Kaki'
}

# Judul dan layout halaman
st.set_page_config(page_title="Prediksi Obesitas", layout="centered")
st.title("💪 Prediksi Tingkat Obesitas")
st.markdown("Masukkan data fisik dan kebiasaan harian Anda untuk mengetahui kemungkinan tingkat obesitas menggunakan model *Machine Learning*.")

st.markdown("---")
st.subheader("📝 Formulir Input Data")

# Form input pengguna
with st.form("form_prediksi"):
    col1, col2 = st.columns(2)

    with col1:
        age = st.number_input("Usia", 0.0, 150.0, 25.0)
        gender = st.selectbox("Jenis Kelamin", list(gender_map.values()))
        height = st.number_input("Tinggi Badan (meter)", 0.5, 2.5, 1.70)
        weight = st.number_input("Berat Badan (kg)", 20.0, 200.0, 70.0)
        favc = st.selectbox("Konsumsi Makanan Tinggi Kalori", list(favc_map.values()))
        fcvc = st.slider("Frekuensi Konsumsi Sayur & Buah (0 = Tidak Pernah, 3 = Sangat Sering)", 0.0, 3.0, 2.0)
        ncp = st.slider("Jumlah Makan Utama per Hari", 0.0, 4.0, 3.0)
        smoke = st.selectbox("Apakah Anda Merokok?", list(smoke_map.values()))
        ch2o = st.slider(
            "Konsumsi Air Putih per Hari (Liter)",
            0.0, 4.0, 2.0, step=0.1,
            help="Idealnya 2–3 liter per hari. Nilai ini mencerminkan estimasi konsumsi air Anda."
        )

    with col2:
        calc = st.selectbox("Frekuensi Konsumsi Alkohol", list(calc_map.values()))
        scc = st.selectbox("Apakah Anda Memantau Kalori?", list(scc_map.values()))
        family_history = st.selectbox("Riwayat Obesitas dalam Keluarga?", list(family_history_map.values()))
        faf = st.slider(
            "Frekuensi Aktivitas Fisik (0 = Tidak Pernah, 4 = Sangat Sering)",
            0.0, 4.0, 1.0,
            help="Semakin tinggi nilainya, semakin sering Anda melakukan aktivitas fisik seperti olahraga."
        )
        tue = st.slider(
            "Frekuensi Penggunaan Teknologi / Layar (0 = Tidak Pernah, 4 = Sangat Sering)",
            0.0, 4.0, 1.0,
            help="Menunjukkan seberapa sering Anda menghabiskan waktu di depan layar setiap hari (TV, HP, Laptop)."
        )
        caec = st.selectbox("Kebiasaan Ngemil", list(caec_map.values()))
        mtrans = st.selectbox("Transportasi Utama", list(mtrans_map.values()))

    submitted = st.form_submit_button("🔍 Prediksi Tingkat Obesitas")

# Jika tombol diklik
if submitted:
    # Encoding input
    encoded_input = [
        age,
        {v: k for k, v in gender_map.items()}[gender],
        height,
        weight,
        {v: k for k, v in calc_map.items()}[calc],
        {v: k for k, v in favc_map.items()}[favc],
        fcvc,
        ncp,
        {v: k for k, v in scc_map.items()}[scc],
        {v: k for k, v in smoke_map.items()}[smoke],
        ch2o,
        {v: k for k, v in family_history_map.items()}[family_history],
        faf,
        tue,
        {v: k for k, v in caec_map.items()}[caec],
        {v: k for k, v in mtrans_map.items()}[mtrans],
    ]

    # Preprocessing
    input_df = pd.DataFrame([encoded_input], columns=feature_order)
    input_scaled = scaler.transform(input_df)

    # Prediksi
    prediction = model.predict(input_scaled)
    predicted_label = target_encoder.inverse_transform(prediction)[0]

    st.markdown("---")
    st.subheader("📊 Hasil Prediksi")
    st.success(f"Tingkat Obesitas Anda: **{predicted_label}**")

    # Validasi tambahan untuk konsumsi air
    if ch2o < 1.5:
        st.warning("⚠️ Konsumsi air Anda cukup rendah. Disarankan minum minimal 2 liter per hari.")
    elif ch2o > 3.5:
        st.info("ℹ️ Konsumsi air Anda sangat tinggi. Pastikan sesuai kebutuhan tubuh.")

    # Penjelasan hasil
    label_info = {
        'Insufficient_Weight': '💡 Berat badan kurang. Perlu evaluasi pola makan dan asupan gizi.',
        'Normal_Weight': '✅ Berat badan normal. Pertahankan gaya hidup sehat Anda!',
        'Overweight_Level_I': '⚠️ Kelebihan berat badan tingkat I.',
        'Overweight_Level_II': '⚠️ Kelebihan berat badan tingkat II.',
        'Obesity_Type_I': '❗ Obesitas tipe I. Perlu perhatian khusus pada pola makan dan aktivitas fisik.',
        'Obesity_Type_II': '❗❗ Obesitas tipe II. Konsultasi medis disarankan.',
        'Obesity_Type_III': '🚨 Obesitas tipe III (morbid). Sangat dianjurkan untuk berkonsultasi dengan tenaga medis.'
    }

    st.info(label_info.get(predicted_label, "Informasi tambahan tidak tersedia."))

st.markdown("---")
st.caption("🧠 *Catatan:* Aplikasi ini hanya untuk edukasi dan bukan sebagai pengganti diagnosis medis.")