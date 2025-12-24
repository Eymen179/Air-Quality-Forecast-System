# Hava Kalitesi (PM2.5) Tahmin Sistemi

Bu proje, Pekin şehrindeki hava kirliliği (PM2.5) seviyelerini geçmiş meteorolojik verilere dayanarak tahmin etmek amacıyla geliştirilmiş bir **Derin Öğrenme** uygulamasıdır. Projede zaman serisi verilerini işlemek için **GRU (Gated Recurrent Unit)** mimarisi kullanılmıştır.

---

## 📌 Proje Konusu ve Önemi
Hava kirliliği, halk sağlığını ve kentsel yaşam kalitesini doğrudan etkileyen küresel bir sorundur. Pekin gibi endüstriyel ve nüfus yoğunluğu yüksek şehirlerde, PM2.5 seviyelerinin (ince partikül madde) önceden tahmin edilmesi, sağlık uyarıları ve çevre politikaları için hayati önem taşır. Bu proje, sadece geçmiş kirlilik verilerini değil; sıcaklık, nem ve rüzgar hızı gibi çok değişkenli (multivariate) verileri analiz ederek gelecekteki kirlilik seviyesini öngörmeyi amaçlar.

---

## 📊 Veri Seti Özellikleri
Projede **UCI Machine Learning Repository** üzerinde yer alan "Beijing PM2.5 Data" kullanılmıştır.
* **Zaman Aralığı:** 2010 - 2015 yılları arasındaki saatlik veriler.
* **Kullanılan Değişkenler:** * `PM_US Post`: ABD Elçiliği tarafından ölçülen PM2.5 seviyesi (Hedef Değişken).
    * `DEWP`: Çiğ noktası.
    * `HUMI`: Nem oranı.
    * `PRES`: Hava basıncı.
    * `TEMP`: Sıcaklık.
    * `Iws`: Rüzgar hızı.
* **Veri Temizleme:** Eksik veriler (NaN), zaman serisi bütünlüğünü korumak adına `ffill` (ileri doldurma) yöntemiyle işlenmiştir.

---

## 🧠 Model Mimarisi: Çok Değişkenli GRU
Zaman serisi tahminlerinde yaygın olarak kullanılan LSTM'e alternatif olarak daha düşük hesaplama maliyeti sunan **GRU (Gated Recurrent Unit)** tercih edilmiştir.
* **Girdi Yapısı:** Son 24 saatlik çok değişkenli veri penceresi (24x6).
* **Katmanlar:** 2 katmanlı GRU ve çıktı için 1 adet Tam Bağlantılı (Linear) katman.
* **Optimizasyon:** Adam Optimizer ve MSE Loss fonksiyonu.



---

## 📈 Model Performansı ve Değerlendirme
Modelin başarısı, test seti üzerinde aşağıdaki metrikler kullanılarak ölçülmüştür:

| Metrik | Sonuç | Açıklama |
| :--- | :--- | :--- |
| **MAE (Ortalama Mutlak Hata)** | **12.59** $ug/m^3$ | Tahminlerin gerçek değerden ortalama sapma miktarı. |
| **RMSE (Kök Ortalama Kare Hata)** | **22.59** $ug/m^3$ | Büyük hatalara daha duyarlı hata payı göstergesi. |

> **Analiz:** Çok değişkenli yapıya geçiş ve hiper-parametre optimizasyonu sonrası, tek değişkenli modele göre hata payında %50'ye yakın iyileşme sağlanmıştır.

---

## 🛠️ Kurulum ve Kullanım

### Gereksinimler
* Python 3.11
* Kütüphaneler: `torch`, `gradio`, `numpy`, `pandas`, `scikit-learn`, `matplotlib`, `openpyxl`

### Çalıştırma
1. **Eğitim:** `python train.py`
2. **Arayüz:** `python serve.py`
3. **Görselleştirme:** `python visualize.py`

---

## 🖥️ Kullanıcı Arayüzü Özellikleri
**Gradio** tabanlı arayüz şunları destekler:
* **Excel Desteği:** 24 saatlik veriyi içeren Excel dosyalarını doğrudan yükleyebilme.
* **Otomatik Veri Doğrulama:** Sayısal olmayan veya hatalı formatlanmış verilerin otomatik ayıklanması.
* **Anlık Tahmin:** Gelecek saat için $ug/m^3$ cinsinden PM2.5 tahmini.
