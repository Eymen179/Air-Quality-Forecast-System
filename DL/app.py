import gradio as gr
import torch
import numpy as np
import pandas as pd
import sklearn.preprocessing
from model import AirQualityGRU

# MinMaxScaler nesnesini güvenli listeye ekle
torch.serialization.add_safe_globals([sklearn.preprocessing._data.MinMaxScaler])

# Modeli ve Scaler'ı Yükle
try:
    checkpoint = torch.load("multi_air_model.pth", weights_only=False)
    scaler = checkpoint['scaler']
    features = checkpoint['features']
    model = AirQualityGRU(input_size=len(features))
    model.load_state_dict(checkpoint['model_state'])
    model.eval()
    print("Çok değişkenli model başarıyla yüklendi.")
except Exception as e:
    print(f"Yükleme hatası: {e}")

def process_excel_and_predict(file):
    try:
        if file is None:
            return "Lütfen bir Excel dosyası yükleyin."
        
        # 1. Dosyayı Oku
        df_input = pd.read_excel(file.name)
        
        # 2. Sütun isimlerini temizle
        df_input.columns = [c.strip() for c in df_input.columns]
        
        # 3. SADECE gerekli özellikleri seç ve sayısal tipe zorla
        # errors='coerce' sayıya çevrilemeyenleri (tarih gibi) NaN yapar
        for col in features:
            if col in df_input.columns:
                df_input[col] = pd.to_numeric(df_input[col], errors='coerce')
            else:
                return f"Hata: Excel dosyasında '{col}' sütunu bulunamadı!"
        
        # Eksik verileri (NaN olan tarihleri veya boşlukları) temizle
        df_input = df_input[features].ffill().dropna()
        
        if len(df_input) < 24:
            return f"Hata: Geçerli 24 satır sayısal veri bulunamadı. (Şu an: {len(df_input)})"
        
        # Sadece son 24 saati al
        data = df_input.tail(24).values
        
        # 4. Tahmin Süreci
        data_scaled = scaler.transform(data)
        input_tensor = torch.tensor(data_scaled, dtype=torch.float32).unsqueeze(0)
        
        with torch.no_grad():
            pred_scaled = model(input_tensor)
            # Ters ölçekleme
            dummy = np.zeros((1, len(features)))
            dummy[0, 0] = pred_scaled.item()
            prediction = scaler.inverse_transform(dummy)[0, 0]
            
        return f"Yüklenen Veriye Göre Tahmin: {prediction:.2f} ug/m^3"

    except Exception as e:
        return f"Hata oluştu: {str(e)}"

# Gradio Arayüzü
interface = gr.Interface(
    fn=process_excel_and_predict,
    inputs=gr.File(label="Excel Dosyasını Yükle (.xlsx veya .xls)", file_types=[".xlsx", ".xls"]),
    outputs=gr.Textbox(label="Model Çıktısı"),
    title="Hava Kalitesi Tahmin Sistemi",
    description="Hazırladığınız 24 saatlik veriyi içeren Excel dosyasını yükleyin. "
                "Dosyanız şu sütunları içermelidir: PM_US Post, DEWP, HUMI, PRES, TEMP, Iws",
)

if __name__ == "__main__":
    interface.launch()