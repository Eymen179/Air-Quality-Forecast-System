import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error
from model import AirQualityGRU

# 1. Veri Hazırlama
df = pd.read_csv('BeijingPM20100101_20151231.csv')
# Kullanacağımız sütunlar: PM2.5 + Meteorolojik veriler
features = ['PM_US Post', 'DEWP', 'HUMI', 'PRES', 'TEMP', 'Iws']
df = df[features].ffill().dropna()

# Normalizasyon (Tüm sütunlar için)
scaler = MinMaxScaler()
data_scaled = scaler.fit_transform(df)

seq_length = 24
X, y = [], []

for i in range(len(data_scaled) - seq_length):
    # X: 24 saatlik tüm özellikler (PM, Temp, Hum vb.)
    X.append(data_scaled[i:i + seq_length])
    # y: Sadece bir sonraki saatin PM2.5 değeri (0. indeksteki veri)
    y.append(data_scaled[i + seq_length, 0])

X, y = np.array(X), np.array(y).reshape(-1, 1)

# Eğitim/Test Bölme
split = int(len(X) * 0.8)
X_train, X_test = torch.tensor(X[:split], dtype=torch.float32), torch.tensor(X[split:], dtype=torch.float32)
y_train, y_test = torch.tensor(y[:split], dtype=torch.float32), torch.tensor(y[split:], dtype=torch.float32)

# 2. Model ve Eğitim (input_size=6)
model = AirQualityGRU(input_size=len(features))
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.005) #lr degisti

print("Çok değişkenli model eğitimi başlıyor...")
for epoch in range(100):
    model.train()
    optimizer.zero_grad()
    output = model(X_train)
    loss = criterion(output, y_train)
    loss.backward()
    optimizer.step()
    if (epoch+1) % 10 == 0: print(f"Epoch {epoch+1}, Loss: {loss.item():.6f}")

# 3. Değerlendirme
model.eval()
with torch.no_grad():
    preds = model(X_test).numpy()
    # Ters ölçekleme (Sadece PM2.5 sütunu için geri dönüşüm)
    # Scaler 6 sütunlu olduğu için dummy bir matris oluşturup ters çeviriyoruz
    dummy = np.zeros((len(preds), len(features)))
    dummy[:, 0] = preds.flatten()
    y_test_pred = scaler.inverse_transform(dummy)[:, 0]
    
    dummy_true = np.zeros((len(y_test), len(features)))
    dummy_true[:, 0] = y_test.flatten()
    y_test_true = scaler.inverse_transform(dummy_true)[:, 0]

print(f"\nYENİ MAE: {mean_absolute_error(y_test_true, y_test_pred):.2f}")
print(f"YENİ RMSE: {root_mean_squared_error(y_test_true, y_test_pred):.2f}")

torch.save({'model_state': model.state_dict(), 'scaler': scaler, 'features': features}, "multi_air_model.pth")