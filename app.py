import streamlit as st
import pandas as pd
import numpy as np
import joblib
import time
import os

# Sayfa Ayarları (Sekme adı ve ikonu)
st.set_page_config(page_title="FraudGuard AI", page_icon="🛡️", layout="wide")

# --- CSS İle Özelleştirme (Opsiyonel Görsellik) ---
st.markdown("""
    <style>
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        height: 3em;
        font-size: 20px;
    }
    </style>
    """, unsafe_allow_html=True)


# --- 1. MODELİ YÜKLEME ---
@st.cache_resource
def load_model():
    model_path = 'models/fraud_model.pkl'
    if not os.path.exists(model_path):
        st.error("Model dosyası bulunamadı! Lütfen önce 'python src/main.py' komutunu çalıştırın.")
        return None
    return joblib.load(model_path)


model = load_model()

# --- 2. BAŞLIK VE GİRİŞ ---
col1, col2 = st.columns([3, 1])
with col1:
    st.title("🛡️ FraudGuard: Yapay Zeka Dolandırıcılık Tespiti")
    st.markdown("""
    Bu sistem, **Gerçek Zamanlı İşlem Analizi** yaparak şüpheli kredi kartı hareketlerini yakalar.
    Modelimiz, **%95 Recall** oranıyla dolandırıcılık vakalarını kaçırmamak üzere optimize edilmiştir.
    """)
with col2:
    # Logon varsa buraya koyabilirsin, yoksa boş kalabilir veya emoji
    st.image("https://cdn-icons-png.flaticon.com/512/2058/2058768.png", width=100)

st.divider()

# --- 3. SOL PANEL: KULLANICI GİRİŞLERİ ---
st.sidebar.header("📝 İşlem Detayları")


def user_input_features():
    # Temel Bilgiler
    amount = st.sidebar.number_input("İşlem Tutarı ($)", min_value=0.0, value=150.0, step=10.0)

    # Zaman Simülasyonu (Bizim Time_Diff özelliğimiz için kritik!)
    st.sidebar.subheader("⏳ Zamanlama Analizi")
    hour = st.sidebar.slider("Günün Saati (0-24)", 0, 23, 14, help="İşlemin yapıldığı saat.")
    time_diff = st.sidebar.slider("Son İşlemden Geçen Süre (Saniye)", 0, 3600, 300,
                                  help="Bu kartla yapılan bir önceki işlemle arasındaki fark. Düşük süre (örn: 10 sn) yüksek risk demektir!")

    # V1-V28 Gizli Değişkenler (Demo için genelde V4, V11, V14 önemlidir)
    st.sidebar.subheader("🔒 Şifreli Banka Verileri (PCA)")
    with st.sidebar.expander("Gelişmiş Veri Girişi (V1-V28)"):
        v4 = st.number_input("V4 (Genel Anomali)", value=0.0)
        v11 = st.number_input("V11 (Risk Faktörü)", value=0.0)
        v14 = st.number_input("V14 (Negatif Etki)", value=0.0)
        # Diğerleri 0 varsayılabilir demo için

    # Veriyi DataFrame'e Çevir (Modelin beklediği ham format)
    # Not: Model V1...V28'in tamamını bekler, olmayanları 0 ile dolduruyoruz.
    data = {
        'Amount': amount,
        'Time_Diff_Simulated': time_diff,  # Bunu aşağıda işleyeceğiz
        'Hour_Simulated': hour,
        'V4': v4, 'V11': v11, 'V14': v14
    }

    # Diğer V sütunlarını 0 olarak ekle
    for i in range(1, 29):
        col_name = f'V{i}'
        if col_name not in data:
            data[col_name] = 0.0

    return pd.DataFrame(data, index=[0])


input_df = user_input_features()


# --- 4. ÖZELLİK MÜHENDİSLİĞİ (PIPELINE İLE AYNI OLMALI) ---
# --- 4. ÖZELLİK MÜHENDİSLİĞİ (GÜNCELLENMİŞ VERSİYON) ---
def preprocess_input(df):
    df_new = df.copy()

    # A. Özellikleri Oluştur
    df_new['Amount_Log'] = np.log1p(df_new['Amount'])
    df_new['Time_Diff'] = df_new['Time_Diff_Simulated']
    df_new['Hour'] = df_new['Hour_Simulated']
    df_new['Is_Night'] = df_new['Hour'].apply(lambda x: 1 if (x < 6 or x >= 22) else 0)

    # PCA İstatistikleri
    pca_cols = [f'V{i}' for i in range(1, 29)]
    df_new['PCA_Abs_Mean'] = df_new[pca_cols].abs().mean(axis=1)
    df_new['PCA_Pos_Sum'] = df_new[pca_cols].apply(lambda x: x[x > 0].sum(), axis=1)
    df_new['PCA_Neg_Sum'] = df_new[pca_cols].apply(lambda x: x[x < 0].sum(), axis=1)

    # B. Modelin Beklediği Sütun Sıralaması (HAYATİ KISIM)
    # main.py'de eğitim sırasında oluşan sıranın aynısı olmalı
    expected_columns = [
        *[f'V{i}' for i in range(1, 29)],  # V1'den V28'e kadar
        'Amount_Log',
        'Time_Diff',
        'Hour',
        'Is_Night',
        'PCA_Abs_Mean',
        'PCA_Pos_Sum',
        'PCA_Neg_Sum'
    ]

    # Veriyi tam olarak bu sıraya diziyoruz
    df_final = df_new[expected_columns]

    return df_final


if model:
    processed_df = preprocess_input(input_df)

    # --- 5. TAHMİN PANELİ ---
    c1, c2 = st.columns(2)

    with c1:
        st.subheader("📊 Simülasyon Verileri")
        st.dataframe(input_df[['Amount', 'Hour_Simulated', 'Time_Diff_Simulated']])

        # EŞİK DEĞER (THRESHOLD) AYARI
        st.markdown("### 🎚️ Hassasiyet Ayarı")
        threshold = st.slider("Karar Eşiği (Threshold)", 0.0, 1.0, 0.05, 0.01,
                              help="Düşük eşik değeri (0.05) güvenliği artırır ama yanlış alarmları da artırabilir.")

        if threshold == 0.05:
            st.caption("✅ **Önerilen Ayar:** Bankacılık standartları için optimize edilmiştir.")

    with c2:
        st.subheader("🤖 Yapay Zeka Kararı")

        predict_btn = st.button("ANALİZ ET")

        if predict_btn:
            with st.spinner('İşlem inceleniyor...'):
                time.sleep(1)  # Gerilim müziği efekti :)

                # Olasılık tahmini
                proba = model.predict_proba(processed_df)[0, 1]

                # Gösterge
                st.metric(label="Dolandırıcılık Riski", value=f"%{proba * 100:.2f}")

                # Karar
                if proba >= threshold:
                    st.error("🚨 DİKKAT: ŞÜPHELİ İŞLEM!")
                    st.markdown(f"""
                    **Sebep Analizi:**
                    - Risk skoru belirlenen eşiğin ({threshold}) üzerinde.
                    - **Öneri:** İşlemi bloke et ve müşteriye SMS gönder.
                    """)
                else:
                    st.success("✅ İŞLEM GÜVENLİ")
                    st.markdown("""
                    - Risk skoru kabul edilebilir seviyede.
                    - İşlem onaylanabilir.
                    """)

                # Bar Chart ile risk görselleştirme
                chart_data = pd.DataFrame({'Risk': [proba], 'Güven': [1 - proba]}, index=['Durum'])
                st.bar_chart(chart_data.T)

else:
    st.warning("Model yüklenemedi. Lütfen kurulum adımlarını kontrol edin.")