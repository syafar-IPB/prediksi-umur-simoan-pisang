import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import os

# 1. Memuat Data
def muat_data(file_path):
    try:
        df = pd.read_excel(file_path)
        print("Data berhasil dimuat:")
        print(df.head())
        return df
    except FileNotFoundError:
        print(f"❌ Error: File tidak ditemukan di {file_path}")
        return None

# 2. Pra-pemrosesan Data
def pra_pemrosesan_data(df):
    if df is None:
        return None, None, None, None, None, None

    # --- START Perbaikan di sini ---
    # Membersihkan kolom 'Warna_Kulit (1-5)'
    # Mengekstrak angka dari string (contoh: '1 (Hijau)' menjadi '1')
    # Mengonversi kolom ini ke tipe data numerik (float)
    df['Warna_Kulit (1-5)'] = df['Warna_Kulit (1-5)'].astype(str).str.extract(r'(\d+)').astype(float)

    # Membersihkan kolom 'Kekerasan_Tekstur (1-5)'
    # Mengekstrak angka dari string (contoh: '5 (Keras)' menjadi '5')
    # Mengonversi kolom ini ke tipe data numerik (float)
    df['Kekerasan_Tekstur (1-5)'] = df['Kekerasan_Tekstur (1-5)'].astype(str).str.extract(r'(\d+)').astype(float)
    # --- END Perbaikan di sini ---

    fitur = ['Suhu_Penyimpanan_Celcius', 'Kelembaban_Persen', 'Warna_Kulit (1-5)', 'Kekerasan_Tekstur (1-5)']
    target = 'Estimasi_Sisa_Umur_Simpan_Hari'

    X = df[fitur].values
    y = df[target].values.reshape(-1, 1)

    scaler_X = MinMaxScaler()
    X_scaled = scaler_X.fit_transform(X)

    scaler_y = MinMaxScaler()
    y_scaled = scaler_y.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_scaled, test_size=0.2, random_state=42)

    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    print(f"\n✅ Bentuk X_train_tensor: {X_train_tensor.shape}")
    print(f"✅ Bentuk y_train_tensor: {y_train_tensor.shape}")

    return X_train_tensor, y_train_tensor, X_test_tensor, y_test_tensor, scaler_X, scaler_y

# 3. Model Neural Network
class ModelUmurSimpan(nn.Module):
    def __init__(self, input_size, hidden_size1, hidden_size2, output_size):
        super(ModelUmurSimpan, self).__init__()
        self.layer1 = nn.Linear(input_size, hidden_size1)
        self.relu1 = nn.ReLU()
        self.layer2 = nn.Linear(hidden_size1, hidden_size2)
        self.relu2 = nn.ReLU()
        self.output_layer = nn.Linear(hidden_size2, output_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.relu1(x)
        x = self.layer2(x)
        x = self.relu2(x)
        x = self.output_layer(x)
        return x

# 4. Fungsi Pelatihan
def latih_model(model, X_train, y_train, X_test, y_test, learning_rate=0.01, epochs=200):
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("\n🚀 Memulai pelatihan model...")
    for epoch in range(epochs):
        model.train()
        outputs = model(X_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if (epoch + 1) % 20 == 0:
            model.eval()
            with torch.no_grad():
                test_outputs = model(X_test)
                test_loss = criterion(test_outputs, y_test)
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {loss.item():.4f}, Test Loss: {test_loss.item():.4f}")
    
    print("✅ Pelatihan selesai.")

# 5. Fungsi Prediksi
def prediksi(model, data_input, scaler_X, scaler_y):
    model.eval()
    with torch.no_grad():
        # Pastikan data_input memiliki urutan fitur yang sama dengan fitur
        # yang digunakan untuk melatih model:
        # ['Suhu_Penyimpanan_Celcius', 'Kelembaban_Persen', 'Warna_Kulit (1-5)', 'Kekerasan_Tekstur (1-5)']
        data_input_scaled = scaler_X.transform(np.array(data_input).reshape(1, -1))
        input_tensor = torch.FloatTensor(data_input_scaled)
        pred_scaled = model(input_tensor)
        pred_original_scale = scaler_y.inverse_transform(pred_scaled.numpy())
        return pred_original_scale[0][0]

# --- MAIN ---
if __name__ == "__main__":
    # Path ke file Excel
    # Pastikan file 'data_pisang.xlsx' ada di dalam folder 'data'
    # yang berada di direktori yang sama dengan script python ini.
    file_path_data = os.path.join('..', 'data', 'data_pisang.xlsx')


    # Muat data
    df_pisang = muat_data(file_path_data)

    # Pra-pemrosesan
    X_train, y_train, X_test, y_test, scaler_X, scaler_y = pra_pemrosesan_data(df_pisang)

    if X_train is not None:
        # Inisialisasi model
        input_dim = X_train.shape[1]
        model = ModelUmurSimpan(input_size=input_dim, hidden_size1=16, hidden_size2=8, output_size=1)
        print(f"\n🧠 Arsitektur Model:\n{model}")

        # Latih model
        latih_model(model, X_train, y_train, X_test, y_test, learning_rate=0.01, epochs=500)

        # Prediksi contoh
        # Urutan input harus sesuai dengan 'fitur' yang didefinisikan:
        # ['Suhu_Penyimpanan_Celcius', 'Kelembaban_Persen', 'Warna_Kulit (1-5)', 'Kekerasan_Tekstur (1-5)']
        
        # Contoh 1: Data yang mirip dengan baris pertama data Anda
        # Suhu=25, Kelembaban=60, Warna=1 (Hijau), Kekerasan=5 (Keras)
        data_baru_1 = [25, 60, 1, 5]
        hasil_prediksi_1 = prediksi(model, data_baru_1, scaler_X, scaler_y)
        print(f"\n📈 Prediksi umur simpan untuk data {data_baru_1}: {hasil_prediksi_1:.2f} hari")

        # Contoh 2: Data yang mirip dengan baris kelima data Anda
        # Suhu=26, Kelembaban=65, Warna=3 (Kuning), Kekerasan=3 (Cukup Lunak)
        data_baru_2 = [26, 65, 3, 3]
        hasil_prediksi_2 = prediksi(model, data_baru_2, scaler_X, scaler_y)
        print(f"📈 Prediksi umur simpan untuk data {data_baru_2}: {hasil_prediksi_2:.2f} hari")

        # Contoh 3: Data lain
        # Suhu=20, Kelembaban=70, Warna=1 (Hijau), Kekerasan=5 (Keras)
        data_baru_3 = [20, 70, 1, 5]
        hasil_prediksi_3 = prediksi(model, data_baru_3, scaler_X, scaler_y)
        print(f"📈 Prediksi umur simpan untuk data {data_baru_3}: {hasil_prediksi_3:.2f} hari")
# Simpan model setelah pelatihan

print("📁 Model berhasil disimpan ke 'model_umur_simpan.pth'")


