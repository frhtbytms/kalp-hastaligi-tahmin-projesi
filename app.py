"""
Kalp Hastalığı Risk Tahmin Sistemi - Web Uygulaması
İnteraktif Streamlit Arayüzü

Bu web uygulaması, kullanıcıların kalp hastalığı riskini değerlendirmesine olanak tanır.

Özellikler:
🏠 Ana Sayfa - Proje özeti ve genel istatistikler
📊 Veri Analizi - İnteraktif grafikler ve korelasyon analizleri
🤖 Model Performansı - 3 modelin detaylı karşılaştırması
🔮 Tahmin Yap - Gerçek zamanlı risk hesaplama aracı
ℹ️ Hakkında - Proje detayları ve metodoloji

Teknolojiler: Streamlit, Plotly, Scikit-learn
Geliştirici: Ferhat Bayutmuş
Tarih: 17 Kasım 2025
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
from datetime import datetime

# Sayfa yapılandırması
st.set_page_config(
    page_title="Kalp Hastalığı Tahmini",
    page_icon="❤️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS ile özel stil
st.markdown("""
    <style>
    .main {
        background-color: #f0f2f6;
    }
    .stButton>button {
        background-color: #ff4b4b;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 10px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #ff6b6b;
    }
    .metric-card {
        background-color: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# Başlık
st.title("❤️ Kalp Hastalığı Tahmin Sistemi")
st.markdown("### Makine Öğrenmesi ile Risk Değerlendirmesi")
st.markdown("---")

# Sidebar menü
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=100)
    st.title("Navigasyon")
    
    menu = st.radio(
        "Menü",
        ["🏠 Ana Sayfa", "📊 Veri Analizi", "🤖 Model Performansı", "🔮 Tahmin Yap", "ℹ️ Hakkında"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.info("💡 **Not:** Bu uygulama eğitim amaçlıdır.")
    st.markdown("**Geliştirici:** Ferhat Bayutmuş")
    st.markdown(f"**Tarih:** {datetime.now().strftime('%d.%m.%Y')}")

# Veri yükleme fonksiyonu
@st.cache_data
def load_data():
    """Veri setini yükler"""
    if os.path.exists("heart.csv"):
        return pd.read_csv("heart.csv")
    else:
        st.error("⚠️ heart.csv dosyası bulunamadı!")
        return None

@st.cache_resource
def load_model():
    """Eğitilmiş modeli yükler"""
    if os.path.exists("results/best_model.pkl"):
        return joblib.load("results/best_model.pkl")
    else:
        st.warning("⚠️ Model dosyası bulunamadı. Önce main.py'yi çalıştırın!")
        return None

# Ana sayfa
if menu == "🏠 Ana Sayfa":
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
            <h3>📋 Veri Seti</h3>
            <p>UCI Machine Learning Repository'den alınan kalp hastalığı veri seti</p>
            <h2>303</h2>
            <p>Hasta Kaydı</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Model</h3>
            <p>3 farklı makine öğrenmesi algoritması</p>
            <h2>67.6%</h2>
            <p>En İyi F1-Score</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 Doğruluk</h3>
            <p>Test seti üzerinde ölçülen performans</p>
            <h2>60%</h2>
            <p>Accuracy</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Proje açıklaması
    st.header("📖 Proje Hakkında")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Hedefler")
        st.markdown("""
        - ✅ Kalp hastalığı risk faktörlerini analiz etmek
        - ✅ Makine öğrenmesi modelleri geliştirmek
        - ✅ En iyi tahmin modelini belirlemek
        - ✅ İnteraktif bir arayüz sunmak
        """)
        
        st.subheader("🛠️ Kullanılan Teknolojiler")
        st.markdown("""
        - **Python 3.13**
        - **Scikit-learn** - Makine öğrenmesi
        - **Streamlit** - Web arayüzü
        - **Plotly** - İnteraktif grafikler
        """)
    
    with col2:
        st.subheader("🤖 Modeller")
        st.markdown("""
        1. **Logistic Regression** ⭐
           - En iyi performans
           - F1-Score: 0.68
        
        2. **Random Forest**
           - Ensemble learning
           - F1-Score: 0.58
        
        3. **K-Nearest Neighbors**
           - Mesafe tabanlı
           - F1-Score: 0.58
        """)
        
        st.subheader("📊 Özellikler")
        st.markdown("""
        - Yaş, cinsiyet, göğüs ağrısı tipi
        - Kan basıncı, kolesterol
        - EKG sonuçları, kalp atış hızı
        - ve daha fazlası...
        """)

# Veri analizi sayfası
elif menu == "📊 Veri Analizi":
    st.header("📊 Veri Analizi ve Görselleştirme")
    
    df = load_data()
    
    if df is not None:
        tab1, tab2, tab3 = st.tabs(["📈 Genel Bakış", "🔍 Detaylı Analiz", "📉 Korelasyon"])
        
        with tab1:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Toplam Kayıt", len(df))
            with col2:
                st.metric("Özellik Sayısı", len(df.columns)-1)
            with col3:
                hasta_oran = (df['target'].sum() / len(df) * 100)
                st.metric("Hasta Oranı", f"{hasta_oran:.1f}%")
            with col4:
                st.metric("Sağlıklı Oranı", f"{100-hasta_oran:.1f}%")
            
            st.markdown("---")
            
            # Sınıf dağılımı
            st.subheader("🎯 Hedef Değişken Dağılımı")
            
            fig = px.pie(
                df, 
                names='target',
                title='Kalp Hastalığı Dağılımı',
                labels={'target': 'Durum', 0: 'Sağlıklı', 1: 'Hasta'},
                color='target',
                color_discrete_map={0: '#00cc96', 1: '#ff6b6b'}
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)
            
            # İlk kayıtlar
            st.subheader("📋 Veri Seti Önizleme")
            st.dataframe(df.head(10), use_container_width=True)
        
        with tab2:
            st.subheader("📊 Yaş Dağılımı")
            
            fig = px.histogram(
                df, 
                x='age', 
                color='target',
                nbins=30,
                title='Yaşa Göre Kalp Hastalığı Dağılımı',
                labels={'age': 'Yaş', 'target': 'Durum'},
                color_discrete_map={0: '#00cc96', 1: '#ff6b6b'}
            )
            st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("❤️ Maksimum Kalp Atış Hızı")
                fig = px.box(
                    df, 
                    x='target', 
                    y='thalach',
                    color='target',
                    title='Kalp Atış Hızı Karşılaştırması',
                    labels={'target': 'Durum', 'thalach': 'Max Kalp Atış Hızı'},
                    color_discrete_map={0: '#00cc96', 1: '#ff6b6b'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.subheader("🩺 Kolesterol Seviyeleri")
                fig = px.box(
                    df, 
                    x='target', 
                    y='chol',
                    color='target',
                    title='Kolesterol Karşılaştırması',
                    labels={'target': 'Durum', 'chol': 'Kolesterol (mg/dl)'},
                    color_discrete_map={0: '#00cc96', 1: '#ff6b6b'}
                )
                st.plotly_chart(fig, use_container_width=True)
        
        with tab3:
            st.subheader("🔗 Korelasyon Matrisi")
            
            # Sadece sayısal sütunlar
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            
            fig = px.imshow(
                corr,
                text_auto='.2f',
                aspect='auto',
                color_continuous_scale='RdBu_r',
                title='Özellikler Arası Korelasyon'
            )
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("💡 **Yorumlama:** Koyu renkler güçlü ilişkiyi, açık renkler zayıf ilişkiyi gösterir.")

# Model performansı sayfası
elif menu == "🤖 Model Performansı":
    st.header("🤖 Model Performans Analizi")
    
    # Metrikleri yükle
    if os.path.exists("results/metrics_table.csv"):
        metrics_df = pd.read_csv("results/metrics_table.csv", index_col=0)
        
        st.subheader("📊 Model Karşılaştırması")
        
        # Metrik kartları
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>🥇 En İyi Model</h3>
                <h2>Logistic Regression</h2>
                <p>F1-Score: 0.6757</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🎯 Doğruluk</h3>
                <h2>60.0%</h2>
                <p>Test Accuracy</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card">
                <h3>📈 AUC Score</h3>
                <h2>0.5895</h2>
                <p>ROC-AUC</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Performans tablosu
        st.subheader("📋 Detaylı Metrikler")
        st.dataframe(metrics_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
        
        # Performans grafikleri
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart
            fig = px.bar(
                metrics_df.reset_index(),
                x='index',
                y=['accuracy', 'precision', 'recall', 'f1'],
                title='Model Metrikleri Karşılaştırması',
                labels={'index': 'Model', 'value': 'Skor', 'variable': 'Metrik'},
                barmode='group'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Radar chart
            categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            fig = go.Figure()
            
            for model_name in metrics_df.index:
                fig.add_trace(go.Scatterpolar(
                    r=[
                        metrics_df.loc[model_name, 'accuracy'],
                        metrics_df.loc[model_name, 'precision'],
                        metrics_df.loc[model_name, 'recall'],
                        metrics_df.loc[model_name, 'f1']
                    ],
                    theta=categories,
                    fill='toself',
                    name=model_name
                ))
            
            fig.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
                showlegend=True,
                title='Model Performans Radarı'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # Confusion Matrix ve ROC görselleri
        st.subheader("📊 Confusion Matrix ve ROC Curves")
        
        tab1, tab2, tab3 = st.tabs(["Logistic Regression", "Random Forest", "KNN"])
        
        with tab1:
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists("results/confusion_logistic_regression.png"):
                    st.image("results/confusion_logistic_regression.png", use_container_width=True)
            with col2:
                if os.path.exists("results/roc_logistic_regression.png"):
                    st.image("results/roc_logistic_regression.png", use_container_width=True)
        
        with tab2:
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists("results/confusion_random_forest.png"):
                    st.image("results/confusion_random_forest.png", use_container_width=True)
            with col2:
                if os.path.exists("results/roc_random_forest.png"):
                    st.image("results/roc_random_forest.png", use_container_width=True)
        
        with tab3:
            col1, col2 = st.columns(2)
            with col1:
                if os.path.exists("results/confusion_k-nearest_neighbors.png"):
                    st.image("results/confusion_k-nearest_neighbors.png", use_container_width=True)
            with col2:
                if os.path.exists("results/roc_k-nearest_neighbors.png"):
                    st.image("results/roc_k-nearest_neighbors.png", use_container_width=True)
    else:
        st.warning("⚠️ Metrik dosyaları bulunamadı. Önce main.py'yi çalıştırın!")

# Tahmin sayfası
elif menu == "🔮 Tahmin Yap":
    st.header("🔮 Kalp Hastalığı Risk Tahmini")
    st.markdown("Hasta bilgilerini girerek kalp hastalığı riskini tahmin edin.")
    
    model = load_model()
    
    if model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Genel Bilgiler")
            age = st.slider("Yaş", 20, 100, 50)
            sex = st.selectbox("Cinsiyet", ["Kadın (0)", "Erkek (1)"])
            sex_val = int(sex.split("(")[1][0])
            
            st.subheader("💊 Klinik Değerler")
            trestbps = st.number_input("Dinlenme Kan Basıncı (mm Hg)", 90, 200, 120)
            chol = st.number_input("Kolesterol (mg/dl)", 100, 400, 200)
            thalach = st.number_input("Max Kalp Atış Hızı", 60, 220, 150)
            oldpeak = st.number_input("ST Depresyonu", 0.0, 6.0, 1.0, 0.1)
        
        with col2:
            st.subheader("🩺 Semptomlar")
            cp = st.selectbox("Göğüs Ağrısı Tipi", [
                "0 - Tipik Anjina",
                "1 - Atipik Anjina", 
                "2 - Anjina Olmayan Ağrı",
                "3 - Asemptomatik"
            ])
            cp_val = int(cp.split("-")[0].strip())
            
            fbs = st.selectbox("Açlık Kan Şekeri > 120 mg/dl", ["Hayır (0)", "Evet (1)"])
            fbs_val = int(fbs.split("(")[1][0])
            
            exang = st.selectbox("Egzersiz Anjinası", ["Hayır (0)", "Evet (1)"])
            exang_val = int(exang.split("(")[1][0])
            
            st.subheader("📋 Test Sonuçları")
            restecg = st.selectbox("Dinlenme EKG", ["0 - Normal", "1 - ST-T Anormalliği", "2 - LVH"])
            restecg_val = int(restecg.split("-")[0].strip())
            
            slope = st.selectbox("ST Segment Eğimi", ["0 - Yükseliyor", "1 - Düz", "2 - İniyor"])
            slope_val = int(slope.split("-")[0].strip())
            
            ca = st.selectbox("Renkli Damar Sayısı", ["0", "1", "2", "3"])
            ca_val = int(ca)
            
            thal = st.selectbox("Talassemi", ["1 - Normal", "2 - Sabit Defekt", "3 - Geri Döndürülebilir"])
            thal_val = int(thal.split("-")[0].strip())
        
        st.markdown("---")
        
        if st.button("🔮 TAHMİN YAP", use_container_width=True):
            # Veri hazırlama
            input_data = np.array([[
                age, sex_val, cp_val, trestbps, chol, fbs_val, restecg_val,
                thalach, exang_val, oldpeak, slope_val, ca_val, thal_val
            ]])
            
            # MinMaxScaler ile ölçeklendirme (0-1 arası)
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler()
            
            # Örnek verilerle scaler'ı fit et (gerçek min-max değerleri)
            sample_data = np.array([
                [20, 0, 0, 90, 100, 0, 0, 60, 0, 0, 0, 0, 1],
                [100, 1, 3, 200, 400, 1, 2, 220, 1, 6, 2, 3, 3]
            ])
            scaler.fit(sample_data)
            input_scaled = scaler.transform(input_data)
            
            # Tahmin
            prediction = model.predict(input_scaled)[0]
            prediction_proba = model.predict_proba(input_scaled)[0]
            
            # Sonuç gösterimi
            st.markdown("---")
            st.subheader("📊 Tahmin Sonucu")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if prediction == 1:
                    st.error("⚠️ **RİSK TESPİT EDİLDİ**")
                    st.markdown(f"""
                    <div style="background-color: #ffe6e6; padding: 20px; border-radius: 10px; border-left: 5px solid #ff4b4b;">
                        <h2 style="color: #ff4b4b;">Dikkat!</h2>
                        <p>Model kalp hastalığı riski tespit etti.</p>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.success("✅ **RİSK DÜŞÜK**")
                    st.markdown(f"""
                    <div style="background-color: #e6ffe6; padding: 20px; border-radius: 10px; border-left: 5px solid #00cc96;">
                        <h2 style="color: #00cc96;">İyi Haber!</h2>
                        <p>Model düşük risk tespit etti.</p>
                    </div>
                    """, unsafe_allow_html=True)
            
            with col2:
                st.metric("Risk Olasılığı", f"{prediction_proba[1]*100:.1f}%")
                st.progress(prediction_proba[1])
            
            with col3:
                st.metric("Sağlıklı Olasılığı", f"{prediction_proba[0]*100:.1f}%")
                st.progress(prediction_proba[0])
            
            st.warning("⚠️ **DİKKAT:** Bu tahmin sadece eğitim amaçlıdır. Kesin teşhis için mutlaka bir doktora başvurun!")
    else:
        st.error("❌ Model yüklenemedi! Önce main.py'yi çalıştırarak modeli eğitin.")

# Hakkında sayfası
elif menu == "ℹ️ Hakkında":
    st.header("ℹ️ Proje Hakkında")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Proje Detayları")
        st.markdown("""
        **Proje Adı:** Kalp Hastalığı Tahmin Sistemi  
        **Ders:** Veri Bilimine Giriş  
        **Dönem:** Güz 2025  
        **Öğrenci:** Ferhat Bayutmuş  
        **Üniversite:** İstanbul Medeniyet Üniversitesi
        
        ---
        
        ### 🎯 Proje Amacı
        
        Bu proje, makine öğrenmesi algoritmaları kullanarak 
        kalp hastalığı riskini tahmin etmeyi amaçlamaktadır.
        
        ### 📊 Veri Seti
        
        - **Kaynak:** UCI Machine Learning Repository
        - **Örneklem:** 303 hasta
        - **Özellik:** 13 farklı klinik parametre
        
        ### 🤖 Metodoloji
        
        1. Veri ön işleme ve temizleme
        2. Keşifsel veri analizi (EDA)
        3. Model eğitimi (3 algoritma)
        4. Performans değerlendirmesi
        5. Web uygulaması geliştirme
        """)
    
    with col2:
        st.subheader("🛠️ Teknolojiler")
        st.markdown("""
        ### Python Kütüphaneleri:
        
        - **Pandas** - Veri işleme
        - **NumPy** - Sayısal hesaplamalar
        - **Scikit-learn** - Makine öğrenmesi
        - **Matplotlib/Seaborn** - Statik grafikler
        - **Plotly** - İnteraktif grafikler
        - **Streamlit** - Web arayüzü
        - **Python-docx** - Rapor oluşturma
        
        ---
        
        ### 📈 Model Performansları:
        
        | Model | F1-Score |
        |-------|----------|
        | Logistic Regression | **0.6757** |
        | Random Forest | 0.5789 |
        | KNN | 0.5846 |
        
        ---
        
        ### 📞 İletişim
        
        **Geliştirici:** Ferhat Bayutmuş  
        **E-posta:** ferhatbayutmus58@gmail.com  
        **Üniversite:** İstanbul Medeniyet Üniversitesi
        
        ---
        
        ### 📄 Lisans
        
        Bu proje eğitim amaçlı hazırlanmıştır.  
        © 2025 Ferhat Bayutmuş - Tüm hakları saklıdır.
        """)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 20px;">
        <p>❤️ Sağlık için teknoloji</p>
        <p>Veri Bilimine Giriş Final Projesi - 2025</p>
    </div>
    """, unsafe_allow_html=True)

# Footer (tüm sayfalarda)
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 12px;">
    Made with ❤️ using Streamlit | © 2025 Ferhat Bayutmuş | Eğitim Amaçlıdır
</div>
""", unsafe_allow_html=True)
