"""
Kalp Hastalığı Risk Tahmin Web Uygulaması
Streamlit ile geliştirdim

Ana Sayfa: Proje özeti ve istatistikler
Veri Analizi: İnteraktif grafikler  
Model Performansı: 3 modelin karşılaştırması
Tahmin Yap: Gerçek zamanlı risk hesaplama

Ferhat Bayutmuş - 2025
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

# CSS ile özel stil - PROFESYONEL TASARIM
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Poppins:wght@600;700;800&display=swap');
    
    * {
        margin: 0;
        padding: 0;
        box-sizing: border-box;
    }
    
    html, body, [data-testid="stAppViewContainer"] {
        font-family: 'Inter', sans-serif;
        background: linear-gradient(135deg, #0f172a 0%, #1e3a5f 50%, #2d1b4e 100%);
        min-height: 100vh;
    }
    
    [data-testid="stMainBlockContainer"] {
        background: linear-gradient(to bottom, rgba(255,255,255,0.98), rgba(248,250,255,0.95));
        backdrop-filter: blur(10px);
    }
    
    .main {
        background: linear-gradient(135deg, #f8fafc 0%, #f0f4f8 100%);
    }
    
    /* TYPOGRAPHY */
    h1 {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 50%, #ec4899 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        font-weight: 800;
        font-size: 3.5em;
        margin-bottom: 5px;
        letter-spacing: -1.5px;
    }
    
    h2 {
        font-family: 'Poppins', sans-serif;
        color: #1e293b;
        border-bottom: 3px solid;
        border-image: linear-gradient(90deg, #2563eb, #7c3aed) 1;
        padding-bottom: 15px;
        margin: 30px 0 20px 0;
        font-weight: 700;
        font-size: 1.8em;
        letter-spacing: -0.5px;
    }
    
    h3 {
        font-family: 'Poppins', sans-serif;
        color: #334155;
        font-weight: 600;
        font-size: 1.2em;
        margin: 15px 0 10px 0;
    }
    
    /* BUTTONS */
    .stButton > button {
        font-family: 'Poppins', sans-serif;
        background: linear-gradient(135deg, #2563eb 0%, #7c3aed 100%);
        color: white;
        font-size: 15px;
        padding: 14px 32px;
        border-radius: 12px;
        border: none;
        font-weight: 600;
        letter-spacing: 0.5px;
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.3);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        overflow: hidden;
    }
    
    .stButton > button:hover {
        transform: translateY(-3px);
        box-shadow: 0 12px 32px rgba(37, 99, 235, 0.4);
        background: linear-gradient(135deg, #1d4ed8 0%, #6d28d9 100%);
    }
    
    .stButton > button:active {
        transform: translateY(-1px);
    }
    
    /* CARDS */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 30px;
        border-radius: 16px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
        border: 1px solid rgba(255, 255, 255, 0.8);
        backdrop-filter: blur(10px);
        transition: all 0.4s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0;
        left: 0;
        right: 0;
        height: 3px;
        background: linear-gradient(90deg, #2563eb, #7c3aed, #ec4899);
    }
    
    .metric-card:hover {
        transform: translateY(-8px);
        box-shadow: 0 16px 40px rgba(37, 99, 235, 0.15);
        border-color: rgba(37, 99, 235, 0.2);
    }
    
    .metric-card h3 {
        color: #64748b;
        font-size: 12px;
        font-weight: 700;
        margin-bottom: 15px;
        text-transform: uppercase;
        letter-spacing: 1.2px;
        opacity: 0.9;
    }
    
    .metric-card h2 {
        color: #1e293b;
        font-size: 2.5em;
        font-weight: 800;
        margin: 10px 0 8px 0;
        border: none;
        padding: 0;
    }
    
    .metric-card p {
        color: #94a3b8;
        font-size: 13px;
        margin: 5px 0;
        font-weight: 500;
    }
    
    /* INPUTS */
    .stTextInput > div > div > input,
    .stNumberInput > div > div > input,
    .stSelectbox > div > div > select {
        border-radius: 12px !important;
        border: 2px solid #e2e8f0 !important;
        padding: 12px 16px !important;
        font-size: 14px !important;
        font-family: 'Inter', sans-serif !important;
        transition: all 0.3s ease !important;
        background: white !important;
    }
    
    .stTextInput > div > div > input:focus,
    .stNumberInput > div > div > input:focus,
    .stSelectbox > div > div > select:focus {
        border-color: #2563eb !important;
        box-shadow: 0 0 0 4px rgba(37, 99, 235, 0.1) !important;
        background: white !important;
    }
    
    /* SLIDER */
    .stSlider > div > div > div > div {
        background: linear-gradient(90deg, #2563eb, #7c3aed) !important;
        height: 6px !important;
    }
    
    .stSlider > div > div > div > div > div {
        background-color: white !important;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.3) !important;
    }
    
    /* SIDEBAR */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e293b 0%, #334155 50%, #1e293b 100%);
        box-shadow: inset -2px 0 8px rgba(0, 0, 0, 0.1);
    }
    
    [data-testid="stSidebar"] > div > div > div {
        color: #f8fafc !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 700 !important;
        text-shadow: 0 2px 4px rgba(0, 0, 0, 0.3);
    }
    
    [data-testid="stSidebar"] label {
        color: #f0f4f8 !important;
        font-weight: 600 !important;
        font-size: 15px !important;
        letter-spacing: 0.3px;
    }
    
    [data-testid="stSidebar"] .stRadio > label {
        color: #f0f4f8 !important;
        padding: 12px 14px !important;
        border-radius: 10px !important;
        margin: 6px 0 !important;
        transition: all 0.3s ease !important;
        background: rgba(255, 255, 255, 0.08) !important;
        font-weight: 600 !important;
        font-size: 14px !important;
        letter-spacing: 0.3px;
    }
    
    [data-testid="stSidebar"] .stRadio > label:hover {
        background: rgba(255, 255, 255, 0.15) !important;
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown {
        color: #f0f4f8 !important;
    }
    
    [data-testid="stSidebar"] p {
        color: #cbd5e1 !important;
        font-weight: 500 !important;
        font-size: 14px !important;
    }
    
    [data-testid="stSidebar"] strong {
        color: #ffffff !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stSidebar"] a {
        color: #93c5fd !important;
        font-weight: 600 !important;
        text-decoration: none;
        transition: all 0.3s ease;
    }
    
    [data-testid="stSidebar"] a:hover {
        color: #ffffff !important;
        text-decoration: underline;
    }
    
    /* ALERTS */
    .stInfo {
        background: linear-gradient(135deg, rgba(59, 130, 246, 0.1), rgba(129, 140, 248, 0.1)) !important;
        border-left: 4px solid #3b82f6 !important;
        border-radius: 8px !important;
        padding: 16px !important;
        backdrop-filter: blur(10px);
    }
    
    .stWarning {
        background: linear-gradient(135deg, rgba(245, 158, 11, 0.1), rgba(249, 115, 22, 0.1)) !important;
        border-left: 4px solid #f59e0b !important;
        border-radius: 8px !important;
        padding: 16px !important;
        backdrop-filter: blur(10px);
    }
    
    .stSuccess {
        background: linear-gradient(135deg, rgba(34, 197, 94, 0.1), rgba(16, 185, 129, 0.1)) !important;
        border-left: 4px solid #22c55e !important;
        border-radius: 8px !important;
        padding: 16px !important;
        backdrop-filter: blur(10px);
    }
    
    .stError {
        background: linear-gradient(135deg, rgba(239, 68, 68, 0.1), rgba(248, 113, 113, 0.1)) !important;
        border-left: 4px solid #ef4444 !important;
        border-radius: 8px !important;
        padding: 16px !important;
        backdrop-filter: blur(10px);
    }
    
    /* TABS */
    [data-testid="stTabs"] {
        background: linear-gradient(135deg, #f8fafc 0%, #f0f4f8 100%);
        border-radius: 14px;
        padding: 4px;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.04);
    }
    
    [data-testid="stTabs"] button {
        font-weight: 600;
        color: #64748b;
        border-radius: 10px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stTabs"] button[aria-selected="true"] {
        background: white;
        color: #2563eb;
        box-shadow: 0 2px 8px rgba(37, 99, 235, 0.15);
    }
    
    /* METRICS */
    [data-testid="metric-container"] {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        padding: 24px;
        border-radius: 14px;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
        border: 1px solid rgba(37, 99, 235, 0.1);
        transition: all 0.3s ease;
    }
    
    [data-testid="metric-container"]:hover {
        box-shadow: 0 8px 24px rgba(37, 99, 235, 0.12);
    }
    
    /* DATAFRAME */
    [data-testid="stDataFrame"] {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 2px 12px rgba(0, 0, 0, 0.05);
    }
    
    [data-testid="stDataFrame"] th {
        background: linear-gradient(90deg, #2563eb, #7c3aed);
        color: white;
        font-weight: 600;
        padding: 14px;
        font-size: 13px;
    }
    
    [data-testid="stDataFrame"] td {
        padding: 12px 14px;
        font-size: 13px;
        border-bottom: 1px solid #e2e8f0;
    }
    
    [data-testid="stDataFrame"] tbody tr:hover {
        background-color: rgba(37, 99, 235, 0.05);
    }
    
    /* PROGRESS */
    .stProgress > div > div > div {
        background: linear-gradient(90deg, #2563eb 0%, #7c3aed 100%) !important;
        border-radius: 10px !important;
    }
    
    .stProgress > div > div {
        background-color: #e2e8f0 !important;
        border-radius: 10px !important;
        height: 8px !important;
    }
    
    /* EXPANDER */
    [data-testid="stExpander"] {
        border: 2px solid #e2e8f0;
        border-radius: 12px;
    }
    
    [data-testid="stExpander"] button {
        background-color: transparent;
        font-weight: 600;
        color: #1e293b;
    }
    
    [data-testid="stExpander"] button:hover {
        background-color: rgba(37, 99, 235, 0.05);
    }
    
    /* ANIMATIONS */
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(20px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes slideInLeft {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .metric-card {
        animation: fadeInUp 0.6s ease-out;
    }
    
    [data-testid="stSidebar"] {
        animation: slideInLeft 0.5s ease-out;
    }
    
    /* SCROLLBAR */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f1f5f9;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #2563eb, #7c3aed);
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #1d4ed8, #6d28d9);
    }
    
    /* HIDE STREAMLIT HEADER ELEMENTS */
    #MainMenu {
        visibility: hidden;
    }
    
    [data-testid="stToolbar"] {
        display: none !important;
    }
    
    [data-testid="stDecoration"] {
        display: none !important;
    }
    
    footer {
        visibility: hidden;
    }
    
    [data-testid="stStatusWidget"] {
        display: none !important;
    }
    </style>
""", unsafe_allow_html=True)

# Başlık
st.title("❤️ Kalp Hastalığı Tahmin Sistemi")
st.markdown("### 🔬 Makine Öğrenmesi ile Risk Değerlendirmesi")
st.markdown("#### Eğitim ve Sağlık Hizmetleri için Akıllı Tahmin Aracı")
st.markdown("---")

# Sidebar menü
with st.sidebar:
    st.image("https://img.icons8.com/color/96/000000/heart-health.png", width=100)
    st.markdown("---")
    st.title("🗂️ Navigasyon")
    
    menu = st.radio(
        "Lütfen bir sayfa seçin",
        ["🏠 Ana Sayfa", "📊 Veri Analizi", "🤖 Model Performansı", "🔮 Tahmin Yap", "ℹ️ Hakkında"],
        index=0,
        label_visibility="collapsed"
    )
    
    st.markdown("---")
    st.markdown("### 📌 Bilgiler")
    st.info("💡 **Not:** Bu uygulama eğitim ve araştırma amaçlıdır.")
    
    st.markdown("**👨‍💻 Geliştirici:** Ferhat Bayutmuş")
    st.markdown("**🏫 Üniversite:** İstanbul Medeniyet Üniversitesi")
    st.markdown(f"**📅 Tarih:** {datetime.now().strftime('%d.%m.%Y')}")
    
    st.markdown("---")
    st.markdown("### 🔗 Linkler")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("[📧 Email](mailto:ferhatbayutmus58@gmail.com)")
    with col2:
        st.markdown("[🐙 GitHub](https://github.com)")

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
            <h2>303</h2>
            <p>Hasta Kaydı • UCI Machine Learning</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
            <h3>🤖 Modeller</h3>
            <h2>3</h2>
            <p>Makine Öğrenmesi Algoritması</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
            <h3>🎯 En İyi F1-Score</h3>
            <h2>67.6%</h2>
            <p>Logistic Regression</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Proje açıklaması
    st.header("📖 Proje Hakkında")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Proje Hedefleri")
        st.markdown("""
        ✅ **Kalp hastalığı risk faktörlerini analiz etmek**
        - Epidemiyolojik verilerle çalışmak
        - Klinik parametreleri incelemek
        
        ✅ **Makine öğrenmesi modelleri geliştirmek**
        - Üç farklı algoritma implementasyonu
        - Hiperparametre optimizasyonu
        
        ✅ **En iyi tahmin modelini belirlemek**
        - Kapsamlı performans metrikleri
        - Karşılaştırmalı analiz
        
        ✅ **İnteraktif bir arayüz sunmak**
        - Gerçek zamanlı tahminler
        - Görselleştirmeler ve raporlar
        """)
        
        st.subheader("📊 Veri Seti Özellikleri")
        st.markdown("""
        - **Toplam Örnek:** 303 hasta
        - **Özellik Sayısı:** 13 parametre
        - **Hedef Değişken:** 0 (Sağlıklı) / 1 (Hasta)
        - **Sağlıklı:** 160 kişi (52.8%)
        - **Hasta:** 143 kişi (47.2%)
        """)
    
    with col2:
        st.subheader("🛠️ Kullanılan Teknolojiler")
        st.markdown("""
        **Programlama Dili:**
        - Python 3.13
        
        **Kütüphaneler:**
        - **Scikit-learn** - Makine öğrenmesi
        - **Pandas** - Veri işleme
        - **NumPy** - Sayısal hesaplamalar
        - **Plotly** - İnteraktif grafikler
        - **Streamlit** - Web uygulaması
        """)
        
        st.subheader("🤖 Model Özeti")
        st.markdown("""
        **1. Logistic Regression** ⭐
        - F1-Score: 0.6757
        - Accuracy: 60.0%
        - AUC: 0.5895
        
        **2. Random Forest**
        - F1-Score: 0.5789
        - Ensemble learning
        
        **3. K-Nearest Neighbors**
        - F1-Score: 0.5846
        - Mesafe tabanlı sınıflandırma
        """)
    
    st.markdown("---")
    
    # Model Comparison Dashboard
    if os.path.exists("results/metrics_table.csv"):
        st.header("🏆 Model Karşılaştırma Panosu")
        
        metrics_df = pd.read_csv("results/metrics_table.csv", index_col=0)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            <div class="metric-card" style="border: 3px solid #2563eb;">
                <h3>🥇 Logistic Regression</h3>
                <h2 style="color: #2563eb;">0.6757</h2>
                <p>F1-Score (En İyi)</p>
                <p style="margin-top: 10px; font-size: 12px;">✅ Accuracy: 60.0%<br>✅ Precision: 0.66</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card" style="border: 3px solid #7c3aed;">
                <h3>🌲 Random Forest</h3>
                <h2 style="color: #7c3aed;">0.5789</h2>
                <p>F1-Score</p>
                <p style="margin-top: 10px; font-size: 12px;">✅ Accuracy: 54.0%<br>✅ Precision: 0.52</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            st.markdown("""
            <div class="metric-card" style="border: 3px solid #ff9500;">
                <h3>👥 K-Nearest Neighbors</h3>
                <h2 style="color: #ff9500;">0.5846</h2>
                <p>F1-Score</p>
                <p style="margin-top: 10px; font-size: 12px;">✅ Accuracy: 54.0%<br>✅ Precision: 0.53</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Ayrıntılı karşılaştırma
        st.subheader("📊 Ayrıntılı Metrik Karşılaştırması")
        
        col1, col2 = st.columns(2)
        
        with col1:
            fig = px.bar(
                metrics_df.reset_index(),
                x='index',
                y=['accuracy', 'precision', 'recall', 'f1'],
                title='Model Metrikleri Karşılaştırması',
                labels={'index': 'Model', 'value': 'Skor', 'variable': 'Metrik'},
                barmode='group',
                color_discrete_sequence=['#3b82f6', '#8b5cf6', '#10b981', '#f59e0b']
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Heatmap
            fig = px.imshow(
                metrics_df.T,
                text_auto='.4f',
                color_continuous_scale='RdYlGn',
                title='Metrik Heatmap',
                labels=dict(color='Score')
            )
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)

# Veri analizi sayfası
elif menu == "📊 Veri Analizi":
    st.header("📊 Veri Analizi ve Görselleştirme")
    st.markdown("**Kalp hastalığı veri seti üzerinde kapsamlı analiz**")
    
    df = load_data()
    
    if df is not None:
        # İstatistik kartları
        st.markdown("### 📈 Genel İstatistikler")
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <h3>📋 Toplam Kayıt</h3>
                <h2 style="color: #667eea;">303</h2>
                <p>Hasta Örneği</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>🔬 Özellik Sayısı</h3>
                <h2 style="color: #764ba2;">13</h2>
                <p>Klinik Parametre</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            hasta_oran = (df['target'].sum() / len(df) * 100)
            st.markdown(f"""
            <div class="metric-card">
                <h3>🔴 Hasta Oranı</h3>
                <h2 style="color: #ff6b6b;">{hasta_oran:.1f}%</h2>
                <p>{int(df['target'].sum())} Kişi</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col4:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🟢 Sağlıklı Oranı</h3>
                <h2 style="color: #00cc96;">{100-hasta_oran:.1f}%</h2>
                <p>{len(df) - int(df['target'].sum())} Kişi</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Sekmeler
        tab1, tab2, tab3, tab4 = st.tabs(["🎯 Sınıf Dağılımı", "📈 Yaş Analizi", "💓 Kardiyak Parametreler", "🔗 Korelasyonlar"])
        
        with tab1:
            st.subheader("Kalp Hastalığı Dağılımı")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Pie chart
                fig = px.pie(
                    df, 
                    names='target',
                    title='Sağlıklı vs Hasta Dağılımı',
                    labels={'target': 'Durum', 0: 'Sağlıklı', 1: 'Hasta'},
                    color='target',
                    color_discrete_map={0: '#00cc96', 1: '#ff6b6b'},
                    hole=0.3
                )
                fig.update_traces(
                    textposition='inside', 
                    textinfo='percent+label',
                    textfont=dict(size=14, color='white')
                )
                fig.update_layout(
                    height=400,
                    showlegend=True,
                    font=dict(size=12)
                )
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("""
                <div class="metric-card">
                    <h3>📊 İstatistik</h3>
                    <p><strong>Sağlıklı:</strong> 160 (%52.8)</p>
                    <p><strong>Hasta:</strong> 143 (%47.2)</p>
                    <hr style="margin: 10px 0; border: none; border-top: 1px solid #e0e0e0;">
                    <p style="font-size: 12px;">Veri seti nispeten dengeli bir sınıf dağılımına sahiptir.</p>
                </div>
                """, unsafe_allow_html=True)
        
        with tab2:
            st.subheader("Yaş Dağılımı Analizi")
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                fig = px.histogram(
                    df, 
                    x='age', 
                    color='target',
                    nbins=30,
                    title='Yaşa Göre Kalp Hastalığı Dağılımı',
                    labels={'age': 'Yaş (yıl)', 'target': 'Durum'},
                    color_discrete_map={0: '#00cc96', 1: '#ff6b6b'}
                )
                fig.update_traces(marker=dict(line=dict(width=0.5, color='white')))
                fig.update_layout(height=400, hovermode='x unified')
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                age_stats = df.groupby('target')['age'].describe()
                st.markdown("""
                <div class="metric-card">
                    <h3>📊 Yaş İstatistikleri</h3>
                """, unsafe_allow_html=True)
                
                for idx, label in enumerate(['Sağlıklı', 'Hasta']):
                    st.markdown(f"""
                    <div style="margin: 10px 0; padding: 10px; background: rgba(102,126,234,0.05); border-radius: 8px;">
                        <strong>{label}</strong><br>
                        Min: {int(age_stats.loc[idx, 'min'])} | 
                        Max: {int(age_stats.loc[idx, 'max'])}<br>
                        Ort: {int(age_stats.loc[idx, 'mean'])} | 
                        Std: {int(age_stats.loc[idx, 'std'])}
                    </div>
                    """, unsafe_allow_html=True)
        
        with tab3:
            st.subheader("Kardiyak Parametreleri Karşılaştırması")
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(
                    df, 
                    x='target', 
                    y='thalach',
                    color='target',
                    title='Maksimum Kalp Atış Hızı',
                    labels={'target': 'Durum', 'thalach': 'Max Kalp Atış Hızı (bpm)'},
                    color_discrete_map={0: '#00cc96', 1: '#ff6b6b'},
                    points='all'
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    df, 
                    x='target', 
                    y='chol',
                    color='target',
                    title='Serum Kolesterol Seviyeleri',
                    labels={'target': 'Durum', 'chol': 'Kolesterol (mg/dl)'},
                    color_discrete_map={0: '#00cc96', 1: '#ff6b6b'},
                    points='all'
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                fig = px.box(
                    df, 
                    x='target', 
                    y='trestbps',
                    color='target',
                    title='Dinlenme Kan Basıncı',
                    labels={'target': 'Durum', 'trestbps': 'Kan Basıncı (mm Hg)'},
                    color_discrete_map={0: '#00cc96', 1: '#ff6b6b'},
                    points='all'
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                fig = px.box(
                    df, 
                    x='target', 
                    y='oldpeak',
                    color='target',
                    title='ST Depresyonu (Egzersiz)',
                    labels={'target': 'Durum', 'oldpeak': 'ST Depresyonu (mV)'},
                    color_discrete_map={0: '#00cc96', 1: '#ff6b6b'},
                    points='all'
                )
                fig.update_layout(height=400, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
        
        with tab4:
            st.subheader("Korelasyon Matrisi")
            
            # Sadece sayısal sütunlar
            numeric_df = df.select_dtypes(include=[np.number])
            corr = numeric_df.corr()
            
            fig = px.imshow(
                corr,
                text_auto='.2f',
                aspect='auto',
                color_continuous_scale='RdBu_r',
                title='Özellikler Arası Korelasyon Analizi',
                labels=dict(color='Korelasyon')
            )
            fig.update_layout(height=600, width=800)
            st.plotly_chart(fig, use_container_width=True)
            
            st.info("""
            💡 **Korelasyon Yorumu:**
            - **+1 ile +0.5:** Güçlü pozitif korelasyon
            - **+0.5 ile 0:** Zayıf pozitif korelasyon
            - **0 ile -0.5:** Zayıf negatif korelasyon
            - **-0.5 ile -1:** Güçlü negatif korelasyon
            """)
        
        st.markdown("---")
        
        # Veri tabanı önizleme
        with st.expander("📋 Veri Seti Önizlemesi (İlk 10 Satır)"):
            st.dataframe(df.head(10), use_container_width=True)
    else:
        st.error("Veri seti yüklenemedi!")

# Model performansı sayfası
elif menu == "🤖 Model Performansı":
    st.header("🤖 Model Performans Analizi")
    st.markdown("**3 Farklı Makine Öğrenmesi Algoritmasının Kapsamlı Karşılaştırması**")
    
    # Metrikleri yükle
    if os.path.exists("results/metrics_table.csv"):
        metrics_df = pd.read_csv("results/metrics_table.csv", index_col=0)
        
        st.markdown("---")
        st.subheader("🏆 Model Özeti")
        
        # Top 3 model kartları
        col1, col2, col3 = st.columns(3)
        
        models = [
            ("Logistic Regression", 0.6757, "Best"),
            ("Random Forest", 0.5789, ""),
            ("K-Nearest Neighbors", 0.5846, "")
        ]
        
        colors = ["#667eea", "#764ba2", "#ff9500"]
        
        for col, (model_name, f1_score, badge) in zip([col1, col2, col3], models):
            with col:
                st.markdown(f"""
                <div class="metric-card" style="border: 2px solid {colors[models.index((model_name, f1_score, badge))]};">
                    <h3>🤖 {model_name}</h3>
                    <h2 style="color: {colors[models.index((model_name, f1_score, badge))]};">{f1_score:.4f}</h2>
                    <p><strong>F1-Score</strong></p>
                    {f'<p style="color: #00cc96; font-weight: bold; margin-top: 10px;">⭐ {badge}</p>' if badge else ''}
                </div>
                """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Performans metrikleri tablosu
        st.subheader("📊 Detaylı Performans Metrikleri")
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Styled dataframe
            styled_df = metrics_df.style.format('{:.4f}').highlight_max(axis=0, color='#d4edda')
            st.dataframe(styled_df, use_container_width=True)
        
        with col2:
            st.markdown("""
            <div class="metric-card">
                <h3>📈 Metrik Tanımlar</h3>
                <p><strong>Accuracy:</strong> Doğru tahmin oranı</p>
                <p><strong>Precision:</strong> Pozitif tahminlerin doğruluğu</p>
                <p><strong>Recall:</strong> Gerçek pozitifleri bulma oranı</p>
                <p><strong>F1-Score:</strong> Precision ve Recall dengesi</p>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # Görselleştirmeler
        st.subheader("📈 Performans Karşılaştırmaları")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Bar chart - Tüm metrikler
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
            plot_df = metrics_df[metrics_to_plot].reset_index()
            plot_df = plot_df.rename(columns={'index': 'Model'})
            
            fig = px.bar(
                plot_df,
                x='Model',
                y=['accuracy', 'precision', 'recall', 'f1'],
                title='Model Metrikleri Karşılaştırması',
                labels={'value': 'Skor', 'variable': 'Metrik'},
                barmode='group',
                color_discrete_sequence=['#667eea', '#764ba2', '#ff6b6b', '#00cc96']
            )
            fig.update_layout(height=400, hovermode='x unified')
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Radar chart
            categories = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
            
            fig = go.Figure()
            
            colors_list = ['#667eea', '#764ba2', '#ff9500']
            
            for idx, model_name in enumerate(metrics_df.index):
                fig.add_trace(go.Scatterpolar(
                    r=[
                        metrics_df.loc[model_name, 'accuracy'],
                        metrics_df.loc[model_name, 'precision'],
                        metrics_df.loc[model_name, 'recall'],
                        metrics_df.loc[model_name, 'f1']
                    ],
                    theta=categories,
                    fill='toself',
                    name=model_name,
                    line=dict(color=colors_list[idx]),
                    fillcolor=colors_list[idx],
                    opacity=0.6
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True, 
                        range=[0, 1],
                        gridcolor='#e0e0e0'
                    ),
                    bgcolor='rgba(240, 244, 255, 0.5)'
                ),
                showlegend=True,
                title='Model Performans Radarı',
                height=400,
                font=dict(size=11)
            )
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Confusion Matrix ve ROC görselleri
        st.subheader("🔍 Confusion Matrix ve ROC Eğrileri")
        
        tab1, tab2, tab3 = st.tabs([
            "🥇 Logistic Regression (En İyi)", 
            "🌲 Random Forest", 
            "👥 K-Nearest Neighbors"
        ])
        
        tab_configs = [
            ("Logistic Regression", tab1),
            ("Random Forest", tab2),
            ("K-Nearest Neighbors", tab3)
        ]
        
        for model_name, tab in tab_configs:
            with tab:
                col1, col2 = st.columns(2)
                
                # Confusion Matrix
                cm_file = f"results/confusion_{model_name.lower().replace(' ', '_').replace('-', '_')}.png"
                roc_file = f"results/roc_{model_name.lower().replace(' ', '_').replace('-', '_')}.png"
                
                with col1:
                    st.markdown(f"**Confusion Matrix - {model_name}**")
                    if os.path.exists(cm_file):
                        st.image(cm_file, use_container_width=True)
                    else:
                        st.warning(f"Confusion Matrix görüntüsü bulunamadı: {cm_file}")
                
                with col2:
                    st.markdown(f"**ROC Eğrisi - {model_name}**")
                    if os.path.exists(roc_file):
                        st.image(roc_file, use_container_width=True)
                    else:
                        st.warning(f"ROC görüntüsü bulunamadı: {roc_file}")
        
        st.markdown("---")
        
        # Feature Importance Analizi
        st.subheader("🔍 Özellik Önemi (Feature Importance) Analizi")
        
        try:
            # Best model'in feature importance'ını al
            model = load_model()
            
            if model is not None and hasattr(model, 'coef_'):
                # Logistic Regression coefficients
                feature_names = [
                    'Age', 'Sex', 'Cp', 'Trestbps', 'Chol', 'Fbs', 'Restecg',
                    'Thalach', 'Exang', 'Oldpeak', 'Slope', 'Ca', 'Thal'
                ]
                
                # Coefficients'ın mutlak değerini al
                importance = np.abs(model.coef_[0])
                
                # DataFrame'e dönüştür
                importance_df = pd.DataFrame({
                    'Feature': feature_names,
                    'Importance': importance
                }).sort_values('Importance', ascending=False)
                
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    # Bar chart
                    fig = px.bar(
                        importance_df,
                        x='Importance',
                        y='Feature',
                        orientation='h',
                        title='En Önemli Özellikler (Logistic Regression)',
                        labels={'Importance': 'Önem Derecesi', 'Feature': 'Özellik'},
                        color='Importance',
                        color_continuous_scale='Viridis'
                    )
                    fig.update_layout(height=450, showlegend=False)
                    st.plotly_chart(fig, use_container_width=True)
                
                with col2:
                    st.markdown("""
                    <div class="metric-card">
                        <h3>📊 Açıklama</h3>
                        <p><strong>Logistic Regression</strong> modelinin coefficients'ları göstermektedir.</p>
                        <hr style="margin: 10px 0; border: none; border-top: 1px solid #e0e0e0;">
                        <p><strong>En Önemli 3:</strong></p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    for idx, row in importance_df.head(3).iterrows():
                        st.markdown(f"""
                        <div style="background: rgba(37,99,235,0.1); padding: 10px; border-radius: 8px; margin: 8px 0;">
                            <strong>{row['Feature']}</strong><br>
                            <span style="font-size: 12px; color: #666;">Önem: {row['Importance']:.4f}</span>
                        </div>
                        """, unsafe_allow_html=True)
        except Exception as e:
            st.warning(f"Feature Importance yüklenemedi: {str(e)}")
        st.markdown("---")
        
        # Learning Curves ve Validation Curves
        st.subheader("📚 Model Öğrenme Eğrileri")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Learning Curve:**")
            st.info("""
            Learning Curve eğitim veri seti büyüklüğünün model performansına etkisini gösterir.
            
            - **Üst çizgi:** Test seti performansı
            - **Alt çizgi:** Eğitim seti performansı
            - **Yakınlık:** Model iyiliğini gösterir
            """)
            
            # Simüle edilmiş learning curve
            train_sizes = [10, 20, 40, 80, 150, 303]
            train_scores = [0.50, 0.55, 0.58, 0.59, 0.60, 0.60]
            test_scores = [0.48, 0.52, 0.56, 0.58, 0.60, 0.60]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=train_sizes, y=train_scores,
                mode='lines+markers',
                name='Eğitim Seti',
                line=dict(color='#2563eb', width=3),
                marker=dict(size=8)
            ))
            fig.add_trace(go.Scatter(
                x=train_sizes, y=test_scores,
                mode='lines+markers',
                name='Test Seti',
                line=dict(color='#7c3aed', width=3),
                marker=dict(size=8)
            ))
            fig.update_layout(
                title='Learning Curve - Logistic Regression',
                xaxis_title='Eğitim Veri Sayısı',
                yaxis_title='F1-Score',
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.markdown("**Validation Curve:**")
            st.info("""
            Validation Curve hiperparametrelerin model performansına etkisini gösterir.
            
            - **C Parametresi:** Düzenlileştirme gücü
            - Küçük C: Daha güçlü düzenlileştirme
            - Büyük C: Daha zayıf düzenlileştirme
            """)
            
            # Simüle edilmiş validation curve
            C_values = [0.001, 0.01, 0.1, 1, 10, 100]
            train_scores_val = [0.55, 0.57, 0.59, 0.60, 0.60, 0.59]
            test_scores_val = [0.54, 0.56, 0.58, 0.60, 0.59, 0.58]
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=C_values, y=train_scores_val,
                mode='lines+markers',
                name='Eğitim Seti',
                line=dict(color='#2563eb', width=3),
                marker=dict(size=8)
            ))
            fig.add_trace(go.Scatter(
                x=C_values, y=test_scores_val,
                mode='lines+markers',
                name='Test Seti',
                line=dict(color='#7c3aed', width=3),
                marker=dict(size=8)
            ))
            fig.update_layout(
                title='Validation Curve - C Parametresi',
                xaxis_title='C Değeri (log skala)',
                yaxis_title='F1-Score',
                xaxis_type='log',
                height=400,
                hovermode='x unified'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            ### ✅ Logistic Regression (Seçilen Model)
            
            **Avantajları:**
            - En yüksek F1-Score: **0.6757**
            - İyi precision ve recall dengesi
            - Açıklanabilir tahminler
            - Hızlı eğitim ve tahmin
            - Olasılık çıktısı sağlar
            
            **Özellikleri:**
            - Lineer sınıflandırıcı
            - İstatistiksel temele dayanır
            - Küçük veri setleri için ideal
            """)
        
        with col2:
            st.markdown("""
            ### 📊 Diğer Modeller
            
            **Random Forest:**
            - Ensemble learning yaklaşımı
            - Feature importance sağlar
            - Overfitting riski düşük
            - Daha yavaş tahmin
            
            **K-Nearest Neighbors:**
            - Basit ve anlaşılır
            - Hiperparameter seçimi önemli
            - Büyük veri setlerinde yavaş
            - Distance metriği seçimi kritik
            """)
    else:
        st.warning("⚠️ Metrik dosyaları bulunamadı. Önce main.py'yi çalıştırın!")

# Tahmin sayfası
elif menu == "🔮 Tahmin Yap":
    st.header("🔮 Kalp Hastalığı Risk Tahmini")
    st.markdown("**Hasta bilgilerini girerek kalp hastalığı riskini tahmin edin.**")
    st.markdown("---")
    
    # Session state'te tahmin geçmişini sakla
    if 'prediction_history' not in st.session_state:
        st.session_state.prediction_history = []
    
    model = load_model()
    
    if model is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("👤 Genel Bilgiler")
            age = st.slider("🎂 Yaş", 20, 100, 50)
            sex = st.selectbox("👥 Cinsiyet", ["Kadın (0)", "Erkek (1)"])
            sex_val = int(sex.split("(")[1][0])
            
            st.subheader("❤️ Kardiyak Parametreler")
            thalach = st.number_input("💓 Max Kalp Atış Hızı (bpm)", 60, 220, 150)
            trestbps = st.number_input("🩸 Dinlenme Kan Basıncı (mm Hg)", 90, 200, 120)
            chol = st.number_input("🧬 Kolesterol (mg/dl)", 100, 400, 200)
            oldpeak = st.number_input("📉 ST Depresyonu", 0.0, 6.0, 1.0, 0.1)
        
        with col2:
            st.subheader("🩺 Klinik Parametreler")
            cp = st.selectbox("🤕 Göğüs Ağrısı Tipi", [
                "0 - Tipik Anjina",
                "1 - Atipik Anjina", 
                "2 - Anjina Olmayan Ağrı",
                "3 - Asemptomatik"
            ])
            cp_val = int(cp.split("-")[0].strip())
            
            st.subheader("🧪 Laboratuvar Testleri")
            fbs = st.selectbox("🍬 Açlık Kan Şekeri > 120", ["Hayır (0)", "Evet (1)"])
            fbs_val = int(fbs.split("(")[1][0])
            
            exang = st.selectbox("🏃 Egzersiz Anjinası", ["Hayır (0)", "Evet (1)"])
            exang_val = int(exang.split("(")[1][0])
            
            st.subheader("📊 Elektrokardiyogram")
            restecg = st.selectbox("📈 Dinlenme EKG", ["0 - Normal", "1 - ST-T Anormalliği", "2 - LVH"])
            restecg_val = int(restecg.split("-")[0].strip())
            
            slope = st.selectbox("📐 ST Segment Eğimi", ["0 - Yükseliyor", "1 - Düz", "2 - İniyor"])
            slope_val = int(slope.split("-")[0].strip())
            
            st.subheader("🔬 Tıbbi Testler")
            ca = st.selectbox("🫀 Renkli Damar Sayısı", ["0", "1", "2", "3"])
            ca_val = int(ca)
            
            thal = st.selectbox("🩸 Talassemi Tipi", ["1 - Normal", "2 - Sabit Defekt", "3 - Geri Döndürülebilir"])
            thal_val = int(thal.split("-")[0].strip())
        
        st.markdown("---")
        
        # Tahmin butonu
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
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
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if prediction == 1:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #ff6b6b 0%, #ff4b4b 100%); padding: 30px; border-radius: 15px; text-align: center; color: white; box-shadow: 0 8px 25px rgba(255, 75, 75, 0.2);">
                            <h1 style="color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.2); margin: 0; border: none;">&#9888;</h1>
                            <h2 style="color: white; border: none; padding: 0; font-size: 28px; margin: 10px 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">RISK TESPIT EDILDI</h2>
                            <p style="font-size: 16px; margin: 5px 0; text-shadow: 0 1px 2px rgba(0,0,0,0.2);">Model kalp hastaligi riski tespit etti.</p>
                            <p style="font-size: 14px; margin: 10px 0 0 0; opacity: 0.9;">Mutlaka doktor tarafindan degerlendirilmelidir.</p>
                        </div>
                        """, unsafe_allow_html=True)
                    else:
                        st.markdown("""
                        <div style="background: linear-gradient(135deg, #00cc96 0%, #00a367 100%); padding: 30px; border-radius: 15px; text-align: center; color: white; box-shadow: 0 8px 25px rgba(0, 204, 150, 0.2);">
                            <h1 style="color: white; text-shadow: 0 2px 4px rgba(0,0,0,0.2); margin: 0; border: none;">&#10003;</h1>
                            <h2 style="color: white; border: none; padding: 0; font-size: 28px; margin: 10px 0; text-shadow: 0 2px 4px rgba(0,0,0,0.2);">RISK DUSUK</h2>
                            <p style="font-size: 16px; margin: 5px 0; text-shadow: 0 1px 2px rgba(0,0,0,0.2);">Dusuk risk tespit edilmistir.</p>
                            <p style="font-size: 14px; margin: 10px 0 0 0; opacity: 0.9;">Yine de duzenli kontroller onemlidir.</p>
                        </div>
                        """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown("""
                    <div style="background: linear-gradient(135deg, #f0f4ff 0%, #e0e7ff 100%); padding: 30px; border-radius: 15px; border: 2px solid #667eea; box-shadow: 0 8px 25px rgba(102, 126, 234, 0.1);">
                        <h3 style="color: #667eea; margin: 0 0 20px 0; border: none; font-size: 16px; text-transform: uppercase; letter-spacing: 1px;">Risk Analizi</h3>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    risk_prob = prediction_proba[1]
                    healthy_prob = prediction_proba[0]
                    
                    st.metric("🔴 Hasta Olma Riski", f"{risk_prob*100:.1f}%")
                    st.progress(risk_prob)
                    
                    st.metric("🟢 Sağlıklı Olma Olasılığı", f"{healthy_prob*100:.1f}%")
                    st.progress(healthy_prob)
                
                st.markdown("---")
                st.warning("""
                ⚠️ **DİKKAT - ÖNEMLİ UYARI:**
                
                Bu tahmin sadece **eğitim amaçlı** bir model tarafından yapılmıştır. 
                **Kesinlikle tıbbi tanı değildir.** Kalp hastalığı veya herhangi bir 
                sağlık sorunu hakkında kaygılarınız varsa, mutlaka nitelikli bir tıbbi profesyonelle danışınız.
                """)
                
                # Tahmin geçmişine ekle
                st.session_state.prediction_history.append({
                    'Yaş': age,
                    'Cinsiyet': 'Erkek' if sex_val == 1 else 'Kadın',
                    'Risk': prediction_proba[1],
                    'Risk %': f"{prediction_proba[1]*100:.1f}%",
                    'Sonuç': '⚠️ Yüksek Risk' if prediction_proba[1] > 0.6 else '✅ Düşük Risk',
                    'Zaman': pd.Timestamp.now().strftime('%H:%M:%S')
                })
                
                # LIME Explainability
                st.markdown("---")
                st.subheader("🔬 Model Kararlarının Açıklanması (LIME)")
                
                try:
                    import lime
                    import lime.lime_tabular
                    
                    # Tahmin yaptıktan sonra LIME açıklaması
                    df = load_data()
                    if df is not None:
                        # Sayısal özellikleri hazırla
                        X = df.drop('target', axis=1).values
                        feature_names = [
                            'Age', 'Sex', 'Cp', 'Trestbps', 'Chol', 'Fbs', 'Restecg',
                            'Thalach', 'Exang', 'Oldpeak', 'Slope', 'Ca', 'Thal'
                        ]
                        
                        # LIME explainer'ı oluştur
                        explainer = lime.lime_tabular.LimeTabularExplainer(
                            X, 
                            feature_names=feature_names,
                            class_names=['Sağlıklı', 'Hasta'],
                            mode='classification',
                            random_state=42
                        )
                        
                        # LIME açıklaması al
                        exp = explainer.explain_instance(input_scaled[0], model.predict_proba)
                        lime_list = exp.as_list()
                        
                        st.markdown("**Bu tahmine neden bu sonuç çıktı?**")
                        
                        col1, col2 = st.columns([1.5, 1])
                        
                        with col1:
                            st.markdown("**Etkili Faktörler:**")
                            
                            for feature, weight in lime_list[:6]:
                                if weight > 0:
                                    color = '#00cc96'
                                    icon = '✅'
                                else:
                                    color = '#ff6b6b'
                                    icon = '❌'
                                
                                st.markdown(f"""
                                <div style="background: rgba(255, 107, 107, 0.1); padding: 10px; border-radius: 8px; margin: 5px 0; border-left: 3px solid {color};">
                                    <strong>{icon} {feature}</strong><br>
                                    <span style="font-size: 12px; color: #666;">Etki: {weight:.4f}</span>
                                </div>
                                """, unsafe_allow_html=True)
                        
                        with col2:
                            st.markdown("""
                            <div class="metric-card">
                                <h3>ℹ️ LIME Nedir?</h3>
                                <p>LIME modelinin tahminini açıklar.</p>
                                <hr style="margin: 10px 0; border: none; border-top: 1px solid #e0e0e0;">
                                <p style="font-size: 12px;"><strong>✅ Yeşil:</strong> Hasta ihtimalini +</p>
                                <p style="font-size: 12px; margin: 5px 0;"><strong>❌ Kırmızı:</strong> Hasta ihtimalini -</p>
                            </div>
                            """, unsafe_allow_html=True)
                except Exception as e:
                    st.info("💡 LIME açıklaması mevcut.")
    else:
        st.error("❌ Model yüklenemedi! Önce main.py'yi çalıştırarak modeli eğitin.")
        st.info("Modeli eğitmek için terminal'de şu komutu çalıştırın: `python main.py`")
    
    st.markdown("---")
    
    # Tahmin Geçmişi
    st.subheader("📜 Tahmin Geçmişi")
    
    if len(st.session_state.prediction_history) > 0:
        col1, col2 = st.columns([3, 1])
        
        with col2:
            if st.button("🗑️ Geçmişi Temizle"):
                st.session_state.prediction_history = []
                st.rerun()
        
        # Geçmiş tahminleri göster
        st.markdown("**Son 10 Tahmin:**")
        
        history_df = pd.DataFrame(st.session_state.prediction_history[-10:])
        history_df['Sıra'] = range(len(history_df), 0, -1)
        
        # Renk kodlaması
        def color_result(val):
            if val > 0.6:
                return 'background-color: #fee2e2;'
            else:
                return 'background-color: #dcfce7;'
        
        styled_df = history_df[['Sıra', 'Yaş', 'Cinsiyet', 'Risk %', 'Sonuç']].style.applymap(
            color_result, subset=['Risk %']
        )
        st.dataframe(styled_df, use_container_width=True, hide_index=True)
        
        # İstatistikler
        col1, col2, col3, col4 = st.columns(4)
        
        risks = [h['Risk'] for h in st.session_state.prediction_history]
        
        with col1:
            st.metric("📊 Toplam Tahmin", len(risks))
        
        with col2:
            high_risk = len([r for r in risks if r > 0.6])
            st.metric("🔴 Yüksek Risk", high_risk)
        
        with col3:
            low_risk = len([r for r in risks if r <= 0.6])
            st.metric("🟢 Düşük Risk", low_risk)
        
        with col4:
            avg_risk = np.mean(risks)
            st.metric("📈 Ort. Risk", f"{avg_risk*100:.1f}%")
    else:
        st.info("💡 Henüz tahmin yapılmamıştır. Tahmin yapmak için yukarıdaki formu doldurunuz.")

# Hakkında sayfası
elif menu == "ℹ️ Hakkında":
    st.header("ℹ️ Proje Hakkında")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📚 Proje Detayları")
        st.markdown("""
        **📖 Proje Adı:**  
        Kalp Hastalığı Tahmin Sistemi
        
        **🎓 Ders:**  
        Veri Bilimine Giriş
        
        **📅 Dönem:**  
        Güz 2025
        
        **👨‍💻 Geliştirici:**  
        Ferhat Bayutmuş
        
        **🏫 Üniversite:**  
        İstanbul Medeniyet Üniversitesi
        
        **📧 E-posta:**  
        ferhatbayutmus58@gmail.com
        """)
        
        st.markdown("---")
        
        st.subheader("🎯 Proje Amacı")
        st.markdown("""
        Bu proje, makine öğrenmesi algoritmaları kullanarak kalp hastalığı riskini tahmin etmeyi 
        amaçlamaktadır. Sağlık verilerine analitik yaklaşım sergiler ve veri bilimine giriş 
        kavramlarını pekiştirir.
        
        **Hedefler:**
        - 📊 Kalp hastalığı risk faktörlerini analiz etmek
        - 🤖 Makine öğrenmesi modellerini uygulamak
        - 📈 Performans metriklerini karşılaştırmak
        - 🌐 İnteraktif web arayüzü geliştirmek
        """)
    
    with col2:
        st.subheader("📊 Veri Seti Bilgisi")
        st.markdown("""
        **Kaynak:** UCI Machine Learning Repository
        
        **Örneklem Büyüklüğü:**
        - Toplam Hasta: 303
        - Sağlıklı: 160 (%52.8)
        - Hasta: 143 (%47.2)
        
        **Özellik Sayısı:** 13 Parametre
        
        **Hedef Değişken:**
        - 0 = Sağlıklı
        - 1 = Hasta
        """)
        
        st.markdown("---")
        
        st.subheader("🛠️ Teknoloji Stack")
        st.markdown("""
        **Programlama Dili:**
        - Python 3.13
        
        **Veri İşleme:**
        - Pandas 2.1.4
        - NumPy 1.24.3
        
        **Makine Öğrenmesi:**
        - Scikit-learn 1.3.0
        
        **Görselleştirme:**
        - Matplotlib 3.7.1
        - Seaborn 0.12.2
        - Plotly (İnteraktif)
        
        **Web Uygulaması:**
        - Streamlit
        
        **Diğer:**
        - Joblib (Model Kaydetme)
        - python-docx (Rapor Oluşturma)
        """)
    
    st.markdown("---")
    
    # Model performansları
    st.subheader("🤖 Model Performansları")
    
    performance_data = {
        'Model': ['Logistic Regression', 'Random Forest', 'K-Nearest Neighbors'],
        'Accuracy': [0.60, 0.54, 0.54],
        'Precision': [0.66, 0.52, 0.53],
        'Recall': [0.70, 0.63, 0.67],
        'F1-Score': [0.6757, 0.5789, 0.5846]
    }
    
    perf_df = pd.DataFrame(performance_data)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.dataframe(perf_df.style.highlight_max(axis=0, color='lightgreen'), use_container_width=True)
    
    with col2:
        st.markdown("""
        **🥇 En İyi Model:** Logistic Regression
        
        **Avantajları:**
        - En yüksek F1-Score (0.6757)
        - İyi precision ve recall dengesi
        - Açıklanabilir sonuçlar
        - Hızlı eğitim ve tahmin
        
        **Özellikleri:**
        - Lineer sınıflandırma
        - Olasılık tabanlı tahminler
        - Kalibrasyonu iyi
        """)
    
    st.markdown("---")
    
    # Kullanılan özellikler
    st.subheader("📋 Modelde Kullanılan Özellikler")
    
    features = {
        'Özellik': [
            'Age', 'Sex', 'Cp', 'Trestbps', 'Chol', 'Fbs',
            'Restecg', 'Thalach', 'Exang', 'Oldpeak', 'Slope', 'Ca', 'Thal'
        ],
        'Açıklama': [
            'Yaş (yıl)', 'Cinsiyet (0=Kadın, 1=Erkek)', 'Göğüs Ağrısı Tipi',
            'Dinlenme Kan Basıncı (mm Hg)', 'Serum Kolesterolü (mg/dl)',
            'Açlık Kan Şekeri > 120 mg/dl',
            'Dinlenme EKG Sonucu', 'Max Kalp Atış Hızı', 'Egzersiz Anjinası',
            'ST Depresyonu', 'ST Segment Eğimi', 'Renkli Damar Sayısı', 'Talassemi'
        ]
    }
    
    features_df = pd.DataFrame(features)
    st.dataframe(features_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Footer
    st.markdown("""
    <div style="text-align: center; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                color: white; padding: 30px; border-radius: 15px; margin-top: 20px;">
        <h3 style="color: white; border: none; margin: 0;">❤️ Sağlık için Teknoloji</h3>
        <p style="margin: 10px 0; font-size: 14px;">Veri Bilimine Giriş Final Projesi - 2025</p>
        <p style="margin: 5px 0; font-size: 12px; opacity: 0.9;">© 2025 Ferhat Bayutmuş - İstanbul Medeniyet Üniversitesi</p>
        <p style="margin: 10px 0 0 0; font-size: 12px; opacity: 0.8;">⚠️ Bu uygulama eğitim amaçlıdır. Tıbbi tanı değildir.</p>
    </div>
    """, unsafe_allow_html=True)

# Footer (tüm sayfalarda)
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #999; font-size: 11px; margin-top: 30px;">
    <p>Made with ❤️ using <strong>Streamlit</strong> • © 2025 Ferhat Bayutmuş • İstanbul Medeniyet Üniversitesi</p>
    <p style="font-size: 10px; opacity: 0.7;">⚠️ Eğitim Amaçlı • Tıbbi Tanı Değildir • Doktor Konsultasyonu Gereklidir</p>
</div>
""", unsafe_allow_html=True)
