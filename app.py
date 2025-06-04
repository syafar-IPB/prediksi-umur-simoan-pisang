import streamlit as st
import torch
import numpy as np
import os
from prediksi_umur_simpan import muat_data, pra_pemrosesan_data, ModelUmurSimpan, prediksi

# Muat ulang data dan model
file_path_data = os.path.join('..', 'data', 'data_pisang.xlsx')
df = muat_data(file_path_data)
X_train, y_train, X_test, y_test, scaler_X, scaler_y = pra_pemrosesan_data(df)
model = ModelUmurSimpan(4, 16, 8, 1)
model.load_state_dict(torch.load("model_umur_simpan.pth"))  # atau langsung gunakan model yang sudah dilatih sebelumnya
model.eval()

# UI
st.title("Prediksi Umur Simpan Pisang (by syafar)🍌")

suhu = st.slider("Suhu Penyimpanan (°C)", 15, 35, 25)
kelembaban = st.slider("Kelembaban (%)", 40, 100, 60)
warna = st.selectbox("Warna Kulit (1: Hijau - 5: Cokelat)", [1, 2, 3, 4, 5])
tekstur = st.selectbox("Kekerasan Tekstur (1: Sangat Lunak - 5: Keras)", [1, 2, 3, 4, 5])

if st.button("Prediksi"):
    input_data = [suhu, kelembaban, warna, tekstur]
    hasil = prediksi(model, input_data, scaler_X, scaler_y)
    st.success(f"Estimasi sisa umur simpan: {hasil:.2f} hari")
