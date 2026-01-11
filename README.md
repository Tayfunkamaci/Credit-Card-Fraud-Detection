# 💳 Credit Card Fraud Detection – Policy-Driven Decision Engine

Bu proje, bankacılık sektöründe kullanılan **gerçekçi fraud detection yaklaşımlarını** temel alarak,
**makine öğrenmesi + kural tabanlı (policy-driven)** bir karar motoru geliştirmeyi amaçlamaktadır.

Proje yalnızca bir model eğitmekle kalmaz;  
**“Model çıktısına bakarak bankanın nasıl hareket etmesi gerekir?”** sorusuna cevap verir.

---

## 📌 Proje Amacı

- Gerçek fraud işlemlerini yakalamak (**False Negative maliyeti yüksek**)
- Gereksiz müşteri mağduriyetini azaltmak (**False Positive kontrolü**)
- Model skorunu **tek başına karar mekanizması olarak kullanmamak**
- Bankacılıkta kullanılan **ALLOW / CHALLENGE / BLOCK** karar yapısını simüle etmek

---

## 🧠 Temel Yaklaşım

Bu projede şu gerçek kabul edilmiştir:

> **Fraud modeli ≠ Fraud kararı**

Gerçek bankacılık sistemlerinde:
- Model yalnızca bir **risk sinyali** üretir
- Nihai karar, **iş kuralları + davranışsal çıkarımlar + model skoru** birlikte değerlendirilerek verilir

Bu nedenle proje iki ana bileşenden oluşur:

1. **Model Geliştirme (main.py)**
2. **Karar Motoru & Uygulama (app.py)**

---
## 📊 Veri Seti

Bu projede kullanılan veri seti, kredi kartı işlemleri üzerinden oluşturulmuş ve
fraud tespiti çalışmalarında yaygın olarak kullanılan açık bir veri setidir.

**Kaynak:**
> European cardholders credit card transactions dataset  
> (Kaggle – Credit Card Fraud Detection)

Veri seti, iki gün boyunca gerçekleşen kredi kartı işlemlerini içermektedir ve
işlemlerin çok küçük bir kısmı **fraud (dolandırıcılık)** olarak etiketlenmiştir.

---

### 🔹 Veri Seti Özellikleri

- Toplam işlem sayısı: ~284.000
- Fraud oranı: ~%0.17 (yüksek derecede dengesiz veri)
- Hedef değişken:
  - `Class = 1` → Fraud
  - `Class = 0` → Normal işlem

---

### 🔹 Feature Yapısı

Veri seti şu kolonlardan oluşmaktadır:

- **`V1` – `V28`**
  - PCA (Principal Component Analysis) ile dönüştürülmüş,
    gizlilik nedeniyle anonimleştirilmiş işlem özellikleri
- **`Amount`**
  - İşlem tutarı
- **`Time`**
  - İlk işlemden itibaren geçen süre (saniye)
- **`Class`**
  - Fraud etiketi

> PCA dönüşümü nedeniyle `V1–V28` kolonlarının doğrudan iş anlamı yoktur.
> Bu nedenle projede bu değişkenlerden **türetilmiş agregasyon feature’ları**
> oluşturularak davranışsal çıkarımlar elde edilmiştir.

---

### 🔹 Veri Setinin Projeye Etkisi

Bu veri setinin yapısı, projede şu kararların alınmasına neden olmuştur:

- **Class imbalance** nedeniyle accuracy yerine **cost-based yaklaşım** benimsenmiştir
- PCA feature’ları yorumlanamadığı için:
  - `PCA_Abs_Mean`
  - `PCA_Pos_Sum`
  - `PCA_Neg_Sum`
  gibi özet istatistikler üretilmiştir
- Zamansal bilgi sınırlı olduğu için:
  - `Time_Diff`
  - `Is_Night`
  gibi **işlemsel davranış feature’ları** eklenmiştir

Bu çıkarımlar, hem `main.py`’de model eğitimi aşamasında,
hem de `app.py`’de karar motorunun tasarımında doğrudan kullanılmıştır.

---

## 1️⃣ Model Geliştirme – `main.py`

### 🔹 Veri Seti
- Credit Card Transactions (imbalanced dataset)
- Fraud oranı çok düşük → **class imbalance problemi**

### 🔹 Veri Bölme (Time-Aware Split)

```text
|---------------- TRAIN (Geçmiş %80) ----------------|---- TEST (Gelecek %20) ----|
```
- Random split kullanılmadı
- Geleceği tahmin edebilmek için zamansal ayrım yapıldı
- Data leakage önlendi

---

### 🔹 Feature Engineering

Ham PCA bileşenlerine ek olarak aşağıdaki davranışsal çıkarımlar üretildi:

- `Amount_Log` → tutar ölçekleme  
- `Time_Diff` → ardışık işlem hızı  
- `Is_Night` → gece işlemi bayrağı  
- PCA agregasyonları:
  - `PCA_Abs_Mean`
  - `PCA_Pos_Sum`
  - `PCA_Neg_Sum`


### Amaç:

PCA uzayındaki “olağandışı davranışı” tek değişkenle yakalayabilmek

---

### 🔹 Modelleme

- Tree-based classifier (fraud detection için uygun)
- Sınıf dengesizliği dikkate alındı
- Model çıktısı: Fraud olasılığı (probability)

---

### 🔹 Cost-Based Threshold Optimization

Fraud problemlerinde:
False Negative (gerçek fraud kaçırmak) çok pahalıdır
False Positive (yanlış alarm) müşteri deneyimini bozar

Bu nedenle:
FP ve FN için farklı maliyetler tanımlandı
Threshold, accuracy değil toplam maliyeti minimize edecek şekilde seçildi

Sonuç:  
Modelden gelen skor karar eşiğiyle birlikte saklandı

---

## 2️⃣ Karar Motoru – `app.py`
### 🔹 Temel Felsefe

app.py, modelden gelen çıktıyı doğrudan “fraud” kabul etmez.
Bunun yerine şu soruyu sorar:
- “Bu işlem, geçmiş verilerde gördüğümüz risk desenlerine göre bankanın nasıl tepki vermesi gereken bir işlem mi?”

---

### 🔹 Kullanılan Risk Faktörleri

Model ve EDA çıktılarından elde edilen güçlü fraud sinyalleri:
- Yüksek işlem tutarı
- Çok kısa sürede ardışık işlemler
- Gece saatlerinde yapılan işlemler
- Modelin yüksek fraud olasılığı vermesi
Bu sinyaller tek tek değil, birlikte değerlendirilir.

---

## 🔹 Karar Politikası (Policy)

Her işlem için aşağıdaki risk bayrakları oluşturulur:
- high_amount → Amount ≥ 50.000
- fast_tx → Time_Diff ≤ 10 sn
- night_tx → 22:00 – 06:00
- high_model_risk → model_proba ≥ threshold
Bu bayraklara göre risk skoru hesaplanır.

---

## 🔹 Nihai Karar Mantığı
| Risk Skoru | Karar | Anlam |
|-----------|-------|-------|
| ≥ 3 | BLOCK | Fraud kabul edilir |
| 2 | CHALLENGE | Ek doğrulama (OTP vb.) |
| ≤ 1 | ALLOW | Normal işlem |

Bu yapı sayesinde:
- Model “güvenli” dese bile mantıksız işlemler geçmez
- Gerçek bankacılık davranışı simüle edilir

---

## 🖥️ Streamlit Uygulaması

app.py ile:
- Kullanıcı işlem bilgilerini girer
- Model fraud olasılığını üretir
- Policy motoru nihai kararı verir
- Kararın gerekçesi şeffaf şekilde gösterilir

Örnek: 55.000 ₺ – Gece – 5 sn sonra yapılan işlem --> Model skoru düşük olsa bile BLOCK

---

## 📊 Çıktılar (outputs/)

- Model performans metrikleri
- Cost-based threshold grafikleri
- FP–FN maliyet analizi
- Karar dağılımları.

Bu çıktılar, model ve karar politikalarının
**sezgisel değil, ölçülebilir ve maliyet temelli** olarak belirlendiğini göstermek amacıyla üretilmiştir.

---

## 🔬 Teknik Analiz ve Rapor (Jupyter Notebook)

Bu proje dosyasında özetlenen iş mantığının (Business Logic) arkasındaki **istatistiksel analizleri, veri görselleştirmelerini ve matematiksel hesaplamaları** derinlemesine incelemek için teknik raporumuza göz atabilirsiniz.

**Notebook İçeriği:**
* 📊 **EDA (Keşifçi Veri Analizi):** Fraud işlemlerinin zamansal ve tutar bazlı dağılımları.
* 🧮 **Cost Function Türetimi:** $10₺$ (FP) ve $1000₺$ (FN) maliyetlerinin matematiksel optimizasyonu.
* 🤖 **Model Kıyaslaması:** XGBoost, LightGBM ve Random Forest modellerinin performans detayları.

👉 **[Teknik Analiz Raporunu İncele (Fraud_Detection_Report.ipynb)](notebooks/Fraud_Detection_Report.ipynb)**

---

## 🧠 Neden Bu Yaklaşım?

Bu proje şunu göstermeyi amaçlar:
- Sadece model eğitmek yeterli değildir
- Fraud problemi iş kararı problemidir
- Model + kural + çıkarım birlikte çalışmalıdır

Amaç: “En iyi modeli” değil,  
**gerçek dünyada en az finansal zararla doğru bankacılık kararını verebilen bir sistem** oluşturmaktır.

Bu veri seti PCA ile anonimleştirildiği için
gerçek dünyadaki kullanıcı davranışlarını tam olarak temsil etmez.
Bu nedenle proje, modelden ziyade **karar motoru tasarımına** odaklanmaktadır.

---

## 🚀 Kurulum ve Çalıştırma

Bu projeyi kendi bilgisayarınızda çalıştırmak için aşağıdaki adımları takip edebilirsiniz.

### 1. Gereksinimlerin Yüklenmesi
Projenin çalışması için gerekli kütüphaneler `requirements.txt` dosyasında belirtilmiştir. Terminalde proje dizinine giderek şu komutu çalıştırın:

```bash
pip install -r requirements.txt
```

---

### 2. Veri Setinin Hazırlanması

Proje, model eğitimi için Kaggle üzerindeki `creditcard.csv` veri setini kullanır. Dosya boyutu nedeniyle bu veri seti GitHub deposuna eklenmemiştir.

1.  **İndirme:** Veri setini [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) sayfasından indirin.
2.  **Klasörleme:** Projenin ana dizininde `data` isminde yeni bir klasör oluşturun.
3.  **Taşıma:** İndirdiğiniz arşivden çıkan `creditcard.csv` dosyasını bu `data` klasörünün içine atın.

**Beklenen Klasör Yapısı:**
Dosya yerleşimi tam olarak aşağıdaki gibi olmalıdır:

```text
📂 Project Root
├── 📂 data
│   └── creditcard.csv       <-- Veri seti burada olmalı
├── 📂 notebooks
│   └── fraud_analysis.ipynb <-- Teknik analiz notebook'u
├── 📂 models                <-- main.py çalışınca otomatik oluşur
├── 📂 outputs               <-- Grafikler buraya kaydedilir
├── main.py
├── app.py
├── requirements.txt         <-- Kütüphane listesi
└── README.md
```

---

### 3. Modelin Eğitilmesi

Karar motorunun (uygulamanın) çalışabilmesi için önce makine öğrenmesi modelinin eğitilmesi ve diske kaydedilmesi gerekmektedir.

Proje ana dizininde terminali açın ve aşağıdaki komutu çalıştırın:

```bash
python main.py
```

Bu işlem tamamlandığında:

- 📂 `models/` klasörü içinde eğitilmiş model dosyası (`fraud_model.pkl`) oluşturulur.

- 📊 `outputs/` klasörü içine performans grafikleri (`cost_curve.png`, `confusion_matrix.png`) kaydedilir.

**Not:** `main.py` çalıştırılmadan `app.py` başlatılırsa, uygulama model dosyasını bulamayacağı için hata verecektir.

---

### 4. Karar Motorunun (Arayüz) Başlatılması

Model eğitimi tamamlandıktan sonra, interaktif karar motorunu (Streamlit arayüzü) ayağa kaldırabilirsiniz.

Terminalde şu komutu çalıştırın:

```bash
streamlit run app.py
```

**Komut sonrası:**
- Tarayıcınızda otomatik olarak http://localhost:8501 adresi açılacaktır.
- Sol menüden işlem tutarı ve zaman bilgilerini girerek modelin ve kural motorunun kararlarını simüle edebilirsiniz.