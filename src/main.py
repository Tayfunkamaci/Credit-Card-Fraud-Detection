import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, confusion_matrix
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import warnings
import os

# 1. AYARLAR VE DİNAMİK DOSYA YOLLARI
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 300)

# Dosya yollarını otomatik ayarla (src içinde çalışsa bile ana dizini bulur)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'creditcard.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')

# Çıktı klasörü yoksa oluştur
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)


#################################################################################
# 2. ANALİZ VE GÖRSELLEŞTİRME (EDA)
#################################################################################

def perform_eda(df):
    print("Analiz grafikleri 'outputs/' klasörüne kaydediliyor...")

    # Sınıf Dağılımı
    plt.figure(figsize=(6, 6))
    df['Class'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#66b3ff', '#ff6666'], labels=['Normal', 'Fraud'])
    plt.title("İşlem Sınıf Dağılımı")
    plt.savefig(os.path.join(OUTPUT_DIR, 'class_distribution.png'))
    plt.close()

    # Korelasyon Heatmap
    plt.figure(figsize=(12, 8))
    sns.heatmap(df.corr(), cmap="coolwarm", annot=False)
    plt.title("Özellik Korelasyon Matrisi")
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_heatmap.png'))
    plt.close()

    # Zaman Dağılımı
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Time'], bins=50, kde=True, color='teal')
    plt.title("Zamana Göre İşlem Yoğunluğu")
    plt.savefig(os.path.join(OUTPUT_DIR, 'time_distribution.png'))
    plt.close()


#################################################################################
# 3. ÖZELLİK MÜHENDİSLİĞİ VE MODELLEME
#################################################################################

def create_features_advanced(dataframe):
    df_copy = dataframe.copy()

    # Yeni Özellikler
    df_copy['Amount_Log'] = np.log1p(df_copy['Amount'])
    df_copy['Time_Diff'] = df_copy['Time'].diff().fillna(0)
    df_copy['Hour'] = (df_copy['Time'] // 3600) % 24
    df_copy['Is_Night'] = df_copy['Hour'].apply(lambda x: 1 if (x < 6 or x >= 22) else 0)

    # PCA İstatistikleri (V1-V28)
    pca_cols = [col for col in df_copy.columns if col.startswith('V')]
    df_copy['PCA_Abs_Mean'] = df_copy[pca_cols].abs().mean(axis=1)
    df_copy['PCA_Pos_Sum'] = df_copy[pca_cols].apply(lambda x: x[x > 0].sum(), axis=1)
    df_copy['PCA_Neg_Sum'] = df_copy[pca_cols].apply(lambda x: x[x < 0].sum(), axis=1)

    df_copy.drop(['Time', 'Amount'], axis=1, inplace=True)
    return df_copy


def get_voting_pipeline():
    # Güçlü modellerin birleşimi
    xgb = XGBClassifier(eval_metric="logloss", random_state=17)
    lgbm = LGBMClassifier(random_state=17, verbosity=-1)
    rf = RandomForestClassifier(random_state=17, max_depth=5)

    voting_clf = VotingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm), ('rf', rf)],
        voting='soft'
    )

    # Veri Sızıntısını Önleyen Pipeline
    return ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=42)),
        ('classifier', voting_clf)
    ])


#################################################################################
# 4. ANA ÇALIŞTIRMA BLOĞU
#################################################################################

if __name__ == "__main__":
    # 1. Veri Yükleme
    if not os.path.exists(DATA_PATH):
        print(f"HATA: {DATA_PATH} bulunamadı! Lütfen veriyi 'data' klasörüne koyun.")
    else:
        print("Veri yükleniyor...")
        df = pd.read_csv(DATA_PATH)

        # 2. Analiz
        perform_eda(df)

        # 3. Örnekleme (Veriyi hızlandırmak için)
        fraud = df[df['Class'] == 1]
        non_fraud = df[df['Class'] == 0].sample(frac=0.2, random_state=17)
        df_sampled = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=17)

        # 4. Özellik Mühendisliği
        df_final = create_features_advanced(df_sampled)
        X = df_final.drop('Class', axis=1)
        y = df_final['Class']

        # 5. Veriyi Bölme
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17, stratify=y)

        # 6. Eğitim
        print("Model eğitiliyor (Bu işlem biraz sürebilir)...")
        pipeline = get_voting_pipeline()
        pipeline.fit(X_train, y_train)

        # 7. Tahmin ve Eşik Değer (Threshold) Optimizasyonu
        print("Sonuçlar değerlendiriliyor...")
        y_proba = pipeline.predict_proba(X_test)[:, 1]

        # Bankacılık standartları için %5 eşiği (Recall odaklı)
        threshold = 0.05
        y_pred = (y_proba >= threshold).astype(int)

        # 8. Raporlama
        print(f"\n=== PERFORMANS RAPORU (Threshold: {threshold}) ===")
        print(classification_report(y_test, y_pred))

        # 9. Confusion Matrix Kaydetme
        plt.figure(figsize=(6, 4))
        sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
        plt.title(f"Final Confusion Matrix (Threshold: {threshold})")
        plt.xlabel("Tahmin Edilen")
        plt.ylabel("Gerçek Sınıf")
        plt.savefig(os.path.join(OUTPUT_DIR, 'final_confusion_matrix.png'))

        print(f"\nTüm işlem tamam! Çıktılar şurada: {OUTPUT_DIR}")