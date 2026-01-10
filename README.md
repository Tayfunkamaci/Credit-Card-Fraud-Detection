# 💳 Credit Card Fraud Detection: Sızıntısız ve İş Odaklı AI Modeli

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Scikit-Learn](https://img.shields.io/badge/Library-Scikit--Learn-orange.svg)
![Status](https://img.shields.io/badge/Status-Production%20Ready-green.svg)

Bu proje, bankacılık verileri üzerinde **dolandırıcılık tespiti (fraud detection)** yapan, gerçek hayat senaryolarına uygun (sızıntısız) ve finansal riskleri minimize etmek için optimize edilmiş bir yapay zeka çözümüdür.

---

## 📖 Detaylı Analiz Raporu (Notebook)
Kodun satır satır açıklaması, görselleştirmeler ve analiz mantığı için hazırladığım interaktif teknik rapora buradan ulaşabilirsiniz:
> **👉 [Fraud_Detection_Report.ipynb](./notebooks/Fraud_Detection_Report.ipynb)**

---

## 🎯 Projenin Amacı ve İş Problemi
Kredi kartı dolandırıcılığında en büyük sorun, dolandırıcılık işlemlerinin çok nadir görülmesidir (**%0.17**). Standart bir yapay zeka modeli, "Kimse dolandırıcı değil" tahminini yapsa bile %99.8 başarı elde eder (Accuracy Paradox). Ancak banka için önemli olan o %0.17'lik kısmı yakalamaktır.

**Bu projenin hedefi:** Yanlış alarm oranını (False Positive) yönetilebilir seviyede tutarak, dolandırıcılık vakalarının tamamına yakınını (**High Recall**) yakalamaktır.

---

## 🛠️ Teknik Mimari ve Metodoloji

### 1. Veri Hazırlığı ve Zaman Analizi
Dolandırıcılık işlemlerinde "hız" ve "zamanlama" kritik faktörlerdir. Modelin bunu anlaması için ham veriden şu özellikler türetildi:
- **`Time_Diff` (İşlem Hızı):** Bir kartın art arda yaptığı iki işlem arasındaki saniye farkı. (Saniyeler içinde yapılan çoklu harcamalar şüphelidir).
- **`Is_Night` (Gece İşlemi):** İşlemin gece saatlerinde (22:00 - 06:00) yapılıp yapılmadığı.
- **`Amount_Log`:** İşlem tutarlarındaki uçurumları (1 TL vs 50.000 TL) dengelemek için logaritmik dönüşüm.

### 2. Veri Sızıntısını Önleyen Yapı (Leakage-Free Pipeline)
Çoğu projede yapılan hata, SMOTE (yapay veri üretimi) işleminin tüm veriye uygulanmasıdır. Bu, test verisinin eğitim aşamasında görülmesine (kopya çekmeye) neden olur.
Bu projede `ImbPipeline` kullanılarak, SMOTE işlemi **sadece eğitim (train) setine** uygulanmış, test seti tamamen izole ve saf bırakılmıştır.

### 3. Hibrit Model (Ensemble Learning)
Tek bir modele güvenmek yerine, 3 güçlü algoritmanın "ortak kararı" (Voting) kullanılmıştır:
* **XGBoost:** Hızlı ve yüksek performanslı.
* **LightGBM:** Büyük veride ve dengesiz sınıflarda başarılı.
* **Random Forest:** Kararlılığı artırır ve varyansı düşürür.

---

## 📈 Performans ve Karar Mekanizması (Gerçek Sonuçlar)
Standart modeller `%50` ihtimalin üzerini "Dolandırıcılık" sayar. Ancak bankacılıkta bir dolandırıcılığı kaçırmanın maliyeti çok yüksektir. Bu projede, finansal güvenliği maksimize etmek adına **Recall (Yakalama Oranı)** önceliklendirilmiştir.

Modelin hassasiyet eşiği (threshold) **`0.05`** seviyesine çekilerek agresif bir güvenlik politikası izlenmiştir.

**Test Seti Sonuçları (98 Adet Gerçek Fraud İşlemi Üzerinden):**

| Metrik | Değer | İş Anlamı |
|---|---|---|
| **Recall (Fraud)** | **%95** | **Başarı:** 98 dolandırıcının **93 tanesi** yakalandı. |
| **False Negative** | **~5** | Binlerce işlem arasından sadece 5 vaka gözden kaçtı. |
| **Precision** | **%1** | **Trade-off:** Dolandırıcıları kaçırmamak için yüksek sayıda "Şüpheli İşlem" alarmı üretildi (Güvenlik Önceliği). |

### Neden Düşük Precision?
Precision değerinin düşük olması bilinçli bir **mühendislik tercihidir.** Eşik değerini 0.05 gibi çok düşük bir seviyede tutmak, normal işlemlerin bir kısmının da "incelemeye takılmasına" neden olur.
* **Senaryo:** Banka, 1 dolandırıcıyı kaçırıp 50.000$ kaybetmektense, 100 müşteriye "Bu işlemi siz mi yaptınız?" diye SMS atmayı (False Positive) tercih eder.

![Confusion Matrix](outputs/final_confusion_matrix.png)

---

## 💻 Kurulum ve Çalıştırma

1. Repoyu klonlayın:
   ```bash
   git clone [https://github.com/Tayfunkamaci/Credit-Card-Fraud-Detection.git](https://github.com/Tayfunkamaci/Credit-Card-Fraud-Detection.git)

2. Gerekli kütüphaneleri yükleyin:
    ```bash
   pip install -r requirements.txt
   
3. Modeli eğitin ve sonuçları görün:
    ```bash
   python src/main.py

---

## 📚 Veri Seti ve Kaynakça

Bu proje, **Machine Learning Group (MLG) - Université Libre de Bruxelles (ULB)** tarafından sağlanan ve Worldline iş birliğiyle oluşturulan veri setini temel almaktadır.

**Veri Seti Bağlantısı:**
👉 [Kaggle Credit Card Fraud Detection Dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

### Akademik Atıf
Projeyi akademik veya ticari bir çalışmada kullanacaksanız, lütfen orijinal makaleye atıfta bulunun:

> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. **Calibrating Probability with Undersampling for Unbalanced Classification.** In *Symposium on Computational Intelligence and Data Mining (CIDM)*, IEEE, 2015.