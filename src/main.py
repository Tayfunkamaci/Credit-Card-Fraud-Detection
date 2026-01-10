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
import joblib

# 1. AYARLAR VE DİNAMİK DOSYA YOLLARI
warnings.filterwarnings("ignore")
pd.set_option('display.max_columns', None)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, 'data', 'creditcard.csv')
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')
MODEL_DIR = os.path.join(BASE_DIR, 'models')

if not os.path.exists(OUTPUT_DIR): os.makedirs(OUTPUT_DIR)
if not os.path.exists(MODEL_DIR): os.makedirs(MODEL_DIR)


#################################################################################
# 2. GELİŞMİŞ ÖZELLİK MÜHENDİSLİĞİ
#################################################################################

def create_features_advanced(dataframe):
    """
    Önemli: Bu fonksiyon veri karıştırılmadan (shuffle) ve
    zaman sırası bozulmadan önce çalıştırılmalıdır.
    """
    df_copy = dataframe.copy()

    # 1. Temel Dönüşümler
    df_copy['Amount_Log'] = np.log1p(df_copy['Amount'])

    # Zaman farkı (Time_Diff): Gerçek zamanlı ardışık işlem hızı
    df_copy['Time_Diff'] = df_copy['Time'].diff().fillna(0)

    # Saat ve Gece Değişkeni
    df_copy['Hour'] = (df_copy['Time'] // 3600) % 24
    df_copy['Is_Night'] = df_copy['Hour'].apply(lambda x: 1 if (x < 6 or x >= 22) else 0)

    # 2. Gelişmiş PCA İstatistikleri (V1-V28)
    pca_cols = [col for col in df_copy.columns if col.startswith('V')]
    df_copy['PCA_Abs_Mean'] = df_copy[pca_cols].abs().mean(axis=1)
    df_copy['PCA_Pos_Sum'] = df_copy[pca_cols].apply(lambda x: x[x > 0].sum(), axis=1)
    df_copy['PCA_Neg_Sum'] = df_copy[pca_cols].apply(lambda x: x[x < 0].sum(), axis=1)

    # Ham verileri düşürüyoruz
    df_copy.drop(['Time', 'Amount'], axis=1, inplace=True)
    return df_copy


#################################################################################
# 3. MODEL MİMARİSİ (VOTING PIPELINE)
#################################################################################

def get_voting_pipeline():
    # Zayıf yönleri birbirini dengeleyen 3 güçlü model
    xgb = XGBClassifier(eval_metric="logloss", random_state=17)
    lgbm = LGBMClassifier(random_state=17, verbosity=-1)
    rf = RandomForestClassifier(random_state=17, max_depth=5)

    voting_clf = VotingClassifier(
        estimators=[('xgb', xgb), ('lgbm', lgbm), ('rf', rf)],
        voting='soft'
    )

    # Sızıntısız İşlem Hattı
    pipeline = ImbPipeline([
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=17)),
        ('classifier', voting_clf)
    ])
    return pipeline


#################################################################################
# 4. ANA ÇALIŞTIRICI (MAIN)
#################################################################################

def main():
    # 1. Veri Yükleme ve Sıralama
    if not os.path.exists(DATA_PATH):
        print(f"Hata: Veri dosyası bulunamadı! Yol: {DATA_PATH}")
        return

    print("Veri yükleniyor ve zaman sırasına göre diziliyor...")
    df = pd.read_csv(DATA_PATH)
    df = df.sort_values('Time')  # Zaman farkı analizi için KRİTİK ADIM

    # 2. Özellik Mühendisliği (TÜM VERİ ÜZERİNDE)
    print("Özellik mühendisliği uygulanıyor...")
    df_processed = create_features_advanced(df)

    # 3. Veri Örnekleme (Opsiyonel - Hız için 100k normal işlem alıyoruz)
    # NOT: Özellikler hesaplandıktan sonra örnekleme yapıyoruz!
    #fraud = df_processed[df_processed['Class'] == 1]
    #non_fraud = df_processed[df_processed['Class'] == 0].sample(n=100000, random_state=17)
    #df_final = pd.concat([fraud, non_fraud]).sample(frac=1, random_state=17)

    # Tüm veriyi kullanıyoruz, örnekleme yok!
    print("Tüm veri seti kullanılıyor (Sampling devre dışı)...")
    df_final = df_processed.copy()

    X = df_final.drop('Class', axis=1)
    y = df_final['Class']

    # 4. Eğitim ve Test Ayrımı
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=17, stratify=y)

    # 5. Eğitim
    print(f"Model eğitiliyor (Veri Boyutu: {df_final.shape})...")
    pipeline = get_voting_pipeline()
    pipeline.fit(X_train, y_train)

    # 6. Tahmin ve Threshold (Eşik Değer) Optimizasyonu
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    threshold = 0.05
    y_pred = (y_proba >= threshold).astype(int)

    # 7. Raporlama
    print(f"\n=== PERFORMANS RAPORU (Threshold: {threshold}) ===")
    print(classification_report(y_test, y_pred))

    # 8. Çıktıları ve Modeli Kaydetme
    print("\nGrafikler ve model dosyası kaydediliyor...")

    # Confusion Matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt='d', cmap='Blues')
    plt.title(f'Final Confusion Matrix (Recall Optimized - Threshold: {threshold})')
    plt.savefig(os.path.join(OUTPUT_DIR, 'final_confusion_matrix.png'))

    # Modeli Streamlit için kaydet
    joblib.dump(pipeline, os.path.join(MODEL_DIR, 'fraud_model.pkl'))

    print(f"Başarılı! Çıktılar '{OUTPUT_DIR}' klasöründe, model '{MODEL_DIR}' içinde.")


if __name__ == "__main__":
    main()