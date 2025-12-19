import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, root_mean_squared_error
from model import AirQualityGRU

# 1. Modeli ve Scaler'ı Yükle
checkpoint = torch.load("multi_air_model.pth", weights_only=False)
scaler = checkpoint['scaler']
features = checkpoint['features']
model = AirQualityGRU(input_size=len(features))
model.load_state_dict(checkpoint['model_state'])
model.eval()

# 2. Veriyi Hazırla (Test Setini Yeniden Oluşturuyoruz)
df = pd.read_csv('BeijingPM20100101_20151231.csv')
df = df[features].ffill().dropna()
data_scaled = scaler.transform(df)

seq_length = 24
X, y = [], []
for i in range(len(data_scaled) - seq_length):
    X.append(data_scaled[i:i + seq_length])
    y.append(data_scaled[i + seq_length, 0])

X, y = np.array(X), np.array(y).reshape(-1, 1)

# Eğitim/Test bölmesini train.py ile aynı yapıyoruz (%80-%20)
split = int(len(X) * 0.8)
X_test = torch.tensor(X[split:], dtype=torch.float32)
y_test = y[split:]

# 3. Tahminleri Gerçekleştir
with torch.no_grad():
    test_preds_scaled = model(X_test).numpy()

# 4. Verileri Gerçek Birime (ug/m^3) Geri Çevir
# Inverse transform için dummy matris kullanıyoruz
def inverse_transform_pm(scaled_data, scaler, n_features):
    dummy = np.zeros((len(scaled_data), n_features))
    dummy[:, 0] = scaled_data.flatten()
    return scaler.inverse_transform(dummy)[:, 0]

y_true = inverse_transform_pm(y_test, scaler, len(features))
y_pred = inverse_transform_pm(test_preds_scaled, scaler, len(features))

# 5. GÖRSELLEŞTİRME
plt.figure(figsize=(15, 6))

# Daha net bir görünüm için test setinden rastgele 100 saatlik bir kesit alalım
# (Tüm testi çizmek çok kalabalık görünebilir)
start_point = 0
end_point = 150 

plt.plot(y_true[start_point:end_point], label="Gerçek PM2.5", color='blue', linewidth=2, alpha=0.7)
plt.plot(y_pred[start_point:end_point], label="Model Tahmini", color='red', linestyle='--', linewidth=2)

# Bilgi kutusu ekleyelim
mae = mean_absolute_error(y_true, y_pred)
rmse = root_mean_squared_error(y_true, y_pred)
plt.text(5, max(y_true[start_point:end_point])*0.9, f'MAE: {mae:.2f}\nRMSE: {rmse:.2f}', 
         bbox=dict(facecolor='white', alpha=0.5))

plt.title("Pekin Hava Kalitesi: Gerçek Değerler vs Model Tahminleri (Test Seti Kesiti)")
plt.xlabel("Zaman (Saat)")
plt.ylabel("PM2.5 (ug/m^3)")
plt.legend()
plt.grid(True, which='both', linestyle='--', alpha=0.5)

# Grafiği kaydet ve göster
plt.savefig("performans_grafigi.png")
print("Grafik 'performans_grafigi.png' adıyla kaydedildi.")
plt.show()