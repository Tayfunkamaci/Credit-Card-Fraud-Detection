## 📖 Detaylı Analiz Raporu
Projenin tüm analiz adımlarına, grafiklerine ve detaylı kod açıklamalarına aşağıdaki notebook üzerinden ulaşabilirsiniz:
👉 [Fraud Detection Report (Jupyter Notebook)](./notebooks/Fraud_Detection_Report.ipynb)

💳 Credit Card Fraud Detection: End-to-End Professional Pipeline
Bu proje, Avrupa'daki kart sahiplerinin transactions veri seti üzerinde, gerçek dünya bankacılık problemlerine yönelik geliştirilmiş bir makine öğrenmesi çözümüdür. Projenin temel odağı; veri sızıntısını (data leakage) engellemek, özellik mühendisliği (feature engineering) ile modelin görmediği desenleri yakalamak ve iş odaklı eşik değer (threshold) optimizasyonu yapmaktır.

📂 Veri Seti Hakkında (Reference)
Bu çalışmada kullanılan veri seti, Kaggle üzerinde paylaşılan Credit Card Fraud Detection veri setidir.

İçerik: Eylül 2013'te Avrupa'daki kart sahipleri tarafından yapılan işlemler.

Kısıtlamalar: Gizlilik nedeniyle veriler PCA (Temel Bileşenler Analizi) ile dönüştürülmüştür (V1-V28). Sadece 'Time' ve 'Amount' ham halde bırakılmıştır.

Zorluk: Veri seti aşırı dengesizdir (İşlemlerin yalnızca %0.17'si dolandırıcılıktır).

🛠️ Teknik Süreç ve Metodoloji
1. Keşifçi Veri Analizi (EDA) ve Örnekleme
Veri setindeki %0.17'lik fraud oranı, standart modellerin "her şeye normal" diyerek %99.8 başarı illüzyonuna kapılmasına neden olur.

Sınıf Dağılımı: outputs/class_distribution.png grafiğinde görüldüğü üzere, aşırı dengesizlik SMOTE (Synthetic Minority Over-sampling Technique) kullanımını zorunlu kılmıştır.

Zaman Analizi: outputs/time_distribution.png grafiği ile işlemlerin gün içindeki yoğunlukları incelenmiş, dolandırıcıların tercih ettiği "ölü saatler" için Is_Night değişkeni üretilmiştir.

2. Özellik Mühendisliği (Neyi, Neden Yaptık?)
Sadece ham veriyi modele vermek yerine, bankacılık tecrübesine dayalı yeni metrikler türetilmiştir:

Time_Diff (Velocity Check): Bir işlem ile bir önceki işlem arasındaki saniye farkı. Çok kısa sürede yapılan çok sayıda işlem yüksek risk taşır.

Amount_Log: Harcama tutarlarındaki aşırı uç değerleri (skewness) normalleştirmek için Log dönüşümü uygulanmıştır.

PCA Stats (PCA_Abs_Mean, vb.): V1-V28 arasındaki bileşenlerin genel şiddeti hesaplanarak, dolandırıcılık vakalarındaki "sıradışı sapmalar" tek bir değişkende özetlenmiştir.

3. Veri Sızıntısını Önleyen Pipeline Yapısı
Projenin en kritik teknik başarısı imblearn.pipeline kullanımıdır.

Hata: Eğer SMOTE veya Scaling işlemini train_test_split yapmadan önce tüm veriye uygularsanız, test verisindeki bilgiler eğitim verisine "sızar" ve sonuçlar yalancı bir %100 çıkar.

Çözüm: Pipeline kullanarak, ölçeklendirme ve SMOTE işlemlerinin sadece Cross-Validation sırasında, o anki eğitim katmanına uygulanması sağlanmıştır.

Python

# Profesyonel Pipeline Mimari
return ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', voting_clf)
])
4. Hibrit Modelleme (Voting Classifier)
Tek bir model yerine; XGBoost, LightGBM ve Random Forest algoritmaları "Soft Voting" yöntemiyle birleştirilmiştir. Bu, modelin genelleme yeteneğini artırır ve varyansı düşürür.

5. İş Odaklı Eşik Değer (Threshold) Optimizasyonu
Bankacılıkta 1 dolandırıcılığı kaçırmanın maliyeti, 10 tane yanlış alarmdan çok daha yüksektir. Bu yüzden modelin karar verme eşiği varsayılan 0.50'den 0.05'e çekilmiştir.

📈 Final Performans Sonuçları
Model, dolandırıcılık vakalarını yakalama (Recall) konusunda optimize edilmiştir.

Toplam Yakalanan Fraud: 96

Gözden Kaçan (False Negative): Sadece 2!

Recall Skoru: ~%98

Sonuç: Bu model, bankanın finansal kaybını minimize ederken, operasyonel olarak yönetilebilir bir hatalı alarm oranı sunmaktadır.

💻 Kurulum
Veriyi data/ klasörüne indirin.

Kütüphaneleri yükleyin: pip install -r requirements.txt

Çalıştırın: python src/main.py

## 📚 Kaynakça ve Veri Seti Atfı
Bu projede kullanılan veriler, makine öğrenmesi topluluğu tarafından dolandırıcılık tespiti (fraud detection) çalışmalarında standart bir referans olarak kabul edilmektedir.

**Veri Seti Sahibi:**
Worldline and the Machine Learning Group (MLG) of ULB (Université Libre de Bruxelles).

**Resmi Atıf:**
> Andrea Dal Pozzolo, Olivier Caelen, Reid A. Johnson and Gianluca Bontempi. **Calibrating Probability with Undersampling for Unbalanced Classification.** In *Symposium on Computational Intelligence and Data Mining (CIDM)*, IEEE, 2015.

**Erişim:**
Veri setine [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud) adresi üzerinden ulaşılabilir.