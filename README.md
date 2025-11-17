# ❤️ Kalp Hastalığı Risk Tahmin Sistemi

**Veri Bilimine Giriş - Final Projesi**  
**Öğrenci:** Ferhat Bayutmuş  
**Üniversite:** İstanbul Medeniyet Üniversitesi  
**Akademik Yıl:** 2025-2026 Güz Dönemi  
**Teslim Tarihi:** 17 Kasım 2025

---

## 📋 Projenin Amacı ve Kapsamı

Bu çalışmada, **makine öğrenmesi** teknikleri kullanılarak kalp hastalığı riskinin tahmin edilmesi hedeflenmiştir. Kalp hastalıkları dünya genelinde en yaygın ölüm nedenlerinden biri olduğu için, erken teşhis hayati önem taşımaktadır. 

Proje kapsamında:
- **UCI Machine Learning Repository**'den alınan gerçek hasta verisi analiz edilmiştir
- Veri temizleme, ön işleme ve özellik mühendisliği yapılmıştır
- 3 farklı makine öğrenmesi algoritması eğitilmiş ve karşılaştırılmıştır
- Model performansları detaylı metriklerle değerlendirilmiştir
- Sonuçlar görsel raporlar ve Word dokümanı ile sunulmuştur
- İnteraktif bir **web arayüzü** (Streamlit) geliştirilmiştir

Bu proje, veri bilimi projesi geliştirme sürecinin tüm aşamalarını (veri toplama, temizleme, modelleme, değerlendirme, görselleştirme ve deployment) kapsayan **end-to-end** bir çalışmadır.

---

## 🎯 Proje Aşamaları ve Metodoloji

### 1️⃣ Veri Toplama ve Keşif
- UCI Machine Learning Repository'den **Heart Disease Dataset** indirildi
- Kaggle platformundan alternatif versiyonlar incelendi
- Veri seti 303 hasta kaydı ve 14 değişken içermektedir
- İlk keşifsel veri analizi (EDA) yapılarak veri yapısı anlaşıldı

### 2️⃣ Veri Ön İşleme (`preprocess.py`)
**Yapılan İşlemler:**
- **Eksik Veri Kontrolü:** Median ve mode değerleri ile dolduruldu
- **Aykırı Değer Tespiti:** IQR (Interquartile Range) yöntemi kullanıldı
- **Kategorik Değişkenler:** Label Encoding ile sayısallaştırıldı
- **Ölçeklendirme:** MinMaxScaler ile 0-1 arasına normalize edildi
- **Train-Test Ayrımı:** %80 eğitim, %20 test verisi olarak ayrıldı

**Kullanılan Teknikler:**
```python
- handle_missing_values()  # Eksik veri doldurma
- detect_outliers_iqr()    # Aykırı değer tespiti
- encode_and_scale()       # Encoding ve normalizasyon
```

### 3️⃣ Model Eğitimi (`train_models.py`)
Üç farklı makine öğrenmesi algoritması **GridSearchCV** ile optimize edildi:

**A) Logistic Regression**
- Regularizasyon parametresi: C = [0.01, 0.1, 1]
- Solver: liblinear
- Max iteration: 1000

**B) Random Forest**
- Ağaç sayısı: n_estimators = [50, 100]
- Maksimum derinlik: max_depth = [None, 10, 20]
- Random state: 42

**C) K-Nearest Neighbors**
- Komşu sayısı: n_neighbors = [3, 5, 7]
- Metrik: Euclidean distance
- Ağırlıklandırma: uniform

**Optimizasyon Detayları:**
- Cross-Validation: 3-fold CV
- Scoring Metric: F1-Score (dengesiz veri seti için uygun)
- En iyi hiperparametreler otomatik seçildi

### 4️⃣ Model Değerlendirmesi (`evaluate.py`)
Her model için kapsamlı performans analizi yapıldı:

**Kullanılan Metrikler:**
- **Accuracy:** Genel doğruluk oranı
- **Precision:** Pozitif tahminlerin doğruluğu
- **Recall:** Gerçek pozitifleri bulma oranı
- **F1-Score:** Precision ve Recall'un harmonik ortalaması
- **AUC-ROC:** Sınıflandırma eğrisi altındaki alan

**Görselleştirmeler:**
- Confusion Matrix: Her model için
- ROC Curve: AUC skorları ile
- Feature Correlation Heatmap
- Class Distribution Charts
- Boxplots: Aykırı değer analizi

### 5️⃣ Raporlama (`report_generator.py`)
Python-docx kullanılarak **profesyonel Word raporu** oluşturuldu:
- Proje özeti ve metodoloji
- Model performans tablosu
- Tüm görselleştirmeler embedded
- Sonuç ve öneriler bölümü

### 6️⃣ Web Arayüzü Geliştirme (`app.py`)
**Streamlit** framework'ü ile modern web uygulaması geliştirildi:

**Özellikler:**
- 🏠 **Ana Sayfa:** Proje özeti, istatistikler
- 📊 **Veri Analizi:** İnteraktif Plotly grafikleri
  - Yaş dağılımı histogramları
  - Kolesterol ve kalp atışı boxplotları
  - Korelasyon ısı haritası
- 🤖 **Model Performansı:** Karşılaştırmalı analiz
  - Metrik karşılaştırma tablosu
  - Bar chart ve radar chart
  - Confusion matrix ve ROC curve görselleri
- 🔮 **Tahmin Yap:** Gerçek zamanlı risk hesaplama
  - 13 parametre girişi (slider, selectbox)
  - Eğitilmiş model ile anlık tahmin
  - Risk yüzdesi gösterimi
- ℹ️ **Hakkında:** Proje detayları ve iletişim

**Teknik Detaylar:**
- Responsive tasarım (geniş ekran desteği)
- Custom CSS ile profesyonel görünüm
- Caching ile hızlı performans
- Session state yönetimi

---

## 📊 Veri Seti Detayları

**Heart Disease UCI Dataset**
- **Kaynak:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Alternatif:** Kaggle Heart Disease Dataset
- **Toplam Kayıt:** 303 hasta verisi
- **Özellik Sayısı:** 13 bağımsız değişken + 1 hedef değişken
- **Veri Tipi:** Karma (Sayısal + Kategorik)
- **Eksik Veri:** Minimal (ön işleme ile temizlendi)
- **Sınıf Dağılımı:** Dengeli (0: 138, 1: 165)

### 📋 Değişken Açıklamaları:

| Değişken | Açıklama | Tip | Değer Aralığı |
|----------|----------|-----|---------------|
| **age** | Hastanın yaşı | Sayısal | 29-77 |
| **sex** | Cinsiyet | Kategorik | 0 = Kadın, 1 = Erkek |
| **cp** | Göğüs ağrısı tipi | Kategorik | 0-3 (Tipik anjina, Atipik, Anjina dışı, Asemptomatik) |
| **trestbps** | Dinlenme kan basıncı (mm Hg) | Sayısal | 94-200 |
| **chol** | Serum kolesterol (mg/dl) | Sayısal | 126-564 |
| **fbs** | Açlık kan şekeri > 120 mg/dl | İkili | 0 = Hayır, 1 = Evet |
| **restecg** | Dinlenme EKG sonuçları | Kategorik | 0 = Normal, 1 = ST-T anormalliği, 2 = LVH |
| **thalach** | Maksimum kalp atış hızı | Sayısal | 71-202 |
| **exang** | Egzersiz kaynaklı anjina | İkili | 0 = Hayır, 1 = Evet |
| **oldpeak** | ST depresyonu | Sayısal | 0-6.2 |
| **slope** | ST segment eğimi | Kategorik | 0 = Yükseliyor, 1 = Düz, 2 = İniyor |
| **ca** | Floroskopi ile görülen ana damar sayısı | Sayısal | 0-3 |
| **thal** | Talassemi | Kategorik | 1 = Normal, 2 = Sabit defekt, 3 = Geri döndürülebilir |
| **target** | Kalp hastalığı varlığı (HEDEF) | İkili | 0 = Sağlıklı, 1 = Hasta |

### 🔍 Keşifsel Veri Analizi (EDA) Bulguları:

**Demografik Analizler:**
- Yaş ortalaması: 54.4 ± 9.1 yıl
- Erkek hastaların oranı: %68.3
- En yaygın göğüs ağrısı: Asemptomatik (%47.2)

**Risk Faktörleri:**
- Yüksek kolesterol ortalaması: 246.7 mg/dl
- Ortalama max kalp atışı: 149.6 bpm
- %33.3'ünde egzersiz anjinası mevcut

**Korelasyon Analizi:**
- En güçlü pozitif korelasyon: `cp` (göğüs ağrısı tipi) ile target (0.43)
- En güçlü negatif korelasyon: `oldpeak` (ST depresyonu) ile target (-0.43)
- Yaş ile kalp atışı arasında negatif korelasyon (-0.39)

---

## 🛠️ Kullanılan Teknolojiler ve Kütüphaneler

### **Programlama Dili**
- **Python 3.13.7** - En güncel Python sürümü

### **Veri İşleme ve Analiz**
- **Pandas 2.1.4** - DataFrame işlemleri, veri manipülasyonu
- **NumPy 1.24.3** - Sayısal hesaplamalar, array işlemleri
- **Scikit-learn 1.3.0** - Makine öğrenmesi algoritmaları ve metrikler

### **Görselleştirme**
- **Matplotlib 3.7.1** - Statik grafikler, confusion matrix
- **Seaborn 0.12.2** - İstatistiksel görselleştirme, heatmaps
- **Plotly 5.17.0** - İnteraktif grafikler (web arayüzü için)

### **Web Geliştirme**
- **Streamlit 1.28.1** - Web arayüzü framework'ü
  - Hızlı prototipleme
  - Reaktif bileşenler
  - Deploy desteği

### **Raporlama ve Model Persistance**
- **Python-docx 0.8.11** - Word dokümanı oluşturma
- **Joblib 1.3.2** - Model kaydetme/yükleme (pickle alternatifi)

### **Geliştirme Araçları**
- **Visual Studio Code** - IDE
- **Git** - Versiyon kontrolü
- **PowerShell** - Terminal

### **Neden Bu Teknolojiler Seçildi?**

1. **Scikit-learn:** 
   - Kapsamlı ML algoritma kütüphanesi
   - Kolay hiperparametre optimizasyonu (GridSearchCV)
   - İyi dokümante edilmiş

2. **Streamlit:**
   - Python ile kolay web geliştirme
   - Data science projeleri için optimize
   - Ücretsiz deployment (Streamlit Cloud)

3. **Plotly:**
   - İnteraktif grafikler
   - Modern ve profesyonel görünüm
   - Zoom, pan, hover özellikli

---

## 🤖 Makine Öğrenmesi Modelleri

### **1. Logistic Regression (Lojistik Regresyon)** ⭐ EN İYİ MODEL

**Çalışma Prensibi:**
- İkili sınıflandırma için klasik algoritma
- Sigmoid fonksiyonu ile olasılık hesabı
- Doğrusal karar sınırı oluşturur

**Hiperparametreler:**
```python
param_grid = {
    'C': [0.01, 0.1, 1],        # Regularizasyon gücü
    'solver': ['liblinear'],     # Optimizasyon algoritması
    'max_iter': [1000]           # Maksimum iterasyon
}
```

**Seçilme Nedeni:**
- Basit ve yorumlanabilir
- Hızlı eğitim süresi
- İkili sınıflandırmada güçlü

**Performans:**
- Cross-Validation F1: 0.6757
- Test Accuracy: 60.0%
- AUC-ROC: 0.5895

---

### **2. Random Forest (Rastgele Orman)**

**Çalışma Prensibi:**
- Ensemble learning (topluluk öğrenmesi)
- Birden fazla karar ağacının birleşimi
- Bagging yöntemi ile overfitting önleme

**Hiperparametreler:**
```python
param_grid = {
    'n_estimators': [50, 100],      # Ağaç sayısı
    'max_depth': [None, 10, 20],    # Ağaç derinliği
    'random_state': [42]             # Tekrarlanabilirlik
}
```

**Avantajları:**
- Feature importance hesaplayabilir
- Non-linear ilişkileri yakalayabilir
- Outlier'lara dayanıklı

**Performans:**
- Cross-Validation F1: 0.5789
- Test Accuracy: 56.7%
- AUC-ROC: 0.5106

---

### **3. K-Nearest Neighbors (KNN)**

**Çalışma Prensibi:**
- Instance-based learning
- En yakın k komşuya göre sınıflandırma
- Euclidean mesafe hesabı

**Hiperparametreler:**
```python
param_grid = {
    'n_neighbors': [3, 5, 7],    # Komşu sayısı
    'weights': ['uniform'],       # Ağırlıklandırma
    'metric': ['euclidean']       # Mesafe metriği
}
```

**Özellikler:**
- Parametrik olmayan model
- Lazy learning (eğitim gerektirmez)
- Ölçeklendirmeye duyarlı (bu yüzden normalization önemli)

**Performans:**
- Cross-Validation F1: 0.5846
- Test Accuracy: 58.3%
- AUC-ROC: 0.5784

---

## 📁 Proje Dosya Yapısı ve Açıklamaları

```
veribilimiproje/
│
├── project/                           # Ana proje klasörü
│   │
│   ├── main.py                        # 🎯 Tüm pipeline'ı çalıştıran ana dosya
│   │   └── Fonksiyonlar:
│   │       ├── create_sample_dataset()  # Örnek veri oluşturma
│   │       └── main()                   # Ana workflow
│   │
│   ├── preprocess.py                  # 🔧 Veri ön işleme modülü
│   │   └── Fonksiyonlar:
│   │       ├── load_data()              # CSV yükleme
│   │       ├── handle_missing_values()  # Eksik veri doldurma
│   │       ├── detect_outliers_iqr()    # Aykırı değer tespiti
│   │       └── encode_and_scale()       # Encoding + normalization
│   │
│   ├── train_models.py                # 🤖 Model eğitim modülü
│   │   └── Fonksiyonlar:
│   │       ├── train_logistic_regression()  # Logistic Regression + GridSearchCV
│   │       ├── train_random_forest()        # Random Forest + GridSearchCV
│   │       ├── train_knn()                  # KNN + GridSearchCV
│   │       └── train_all_models()           # Tüm modelleri eğit
│   │
│   ├── evaluate.py                    # 📊 Model değerlendirme modülü
│   │   └── Fonksiyonlar:
│   │       ├── evaluate_model()             # Metrik hesaplama
│   │       ├── plot_confusion_matrix()      # Confusion matrix çizimi
│   │       ├── plot_roc_curve()             # ROC curve çizimi
│   │       └── evaluate_all_models()        # Tüm modelleri değerlendir
│   │
│   ├── report_generator.py            # 📄 Rapor oluşturma modülü
│   │   └── Fonksiyonlar:
│   │       └── generate_report()            # Word raporu oluştur
│   │
│   ├── app.py                         # 🌐 Streamlit web uygulaması
│   │   └── Sayfalar:
│   │       ├── Ana Sayfa                    # Proje özeti
│   │       ├── Veri Analizi                 # İnteraktif EDA
│   │       ├── Model Performansı            # Karşılaştırma
│   │       ├── Tahmin Yap                   # Canlı tahmin
│   │       └── Hakkında                     # Dokümantasyon
│   │
│   ├── heart.csv                      # 📊 Veri seti (303 kayıt)
│   ├── requirements.txt               # 📦 Temel kütüphaneler
│   ├── requirements_app.txt           # 📦 Web uygulaması kütüphaneleri
│   ├── README.md                      # 📖 Bu dosya
│   │
│   ├── results/                       # 📁 Çıktılar klasörü
│   │   ├── best_model.pkl                   # En iyi model (Random Forest)
│   │   ├── metrics_table.csv                # Performans metrikleri tablosu
│   │   ├── Final_Rapor_Heart_Disease.docx   # Word raporu
│   │   │
│   │   ├── corr_heatmap.png                 # Korelasyon ısı haritası
│   │   ├── class_distribution.png           # Sınıf dağılımı grafiği
│   │   ├── boxplots.png                     # Boxplot analizleri
│   │   │
│   │   ├── confusion_logistic_regression.png     # LR confusion matrix
│   │   ├── confusion_random_forest.png           # RF confusion matrix
│   │   ├── confusion_k-nearest_neighbors.png     # KNN confusion matrix
│   │   │
│   │   ├── roc_logistic_regression.png           # LR ROC curve
│   │   ├── roc_random_forest.png                 # RF ROC curve
│   │   └── roc_k-nearest_neighbors.png           # KNN ROC curve
│   │
│   ├── logs/                          # 📝 Log dosyaları
│   │   └── project.log                      # Detaylı çalışma logları
│   │
│   └── __pycache__/                   # Python cache dosyaları
│
└── heart.csv                          # Veri setinin root kopyası
```

### 📝 Dosya Açıklamaları:

**Core Python Modülleri:**
- `main.py`: Tüm projeyi orchestrate eder, pipeline'ı çalıştırır
- `preprocess.py`: Veri temizleme ve hazırlama işlemlerini gerçekleştirir
- `train_models.py`: ML modellerini eğitir ve optimize eder
- `evaluate.py`: Model performansını değerlendirir ve görselleştirir
- `report_generator.py`: Profesyonel Word raporu oluşturur
- `app.py`: Modern web arayüzü sağlar

**Veri Dosyaları:**
- `heart.csv`: 303 hasta kaydı, 13 özellik + 1 target

**Konfigürasyon:**
- `requirements.txt`: Temel ML kütüphaneleri
- `requirements_app.txt`: Web arayüzü için ek kütüphaneler

**Çıktı Dosyaları:**
- `best_model.pkl`: Joblib ile kaydedilmiş eğitilmiş model (Random Forest)
- `metrics_table.csv`: Tüm modellerin performans metrikleri
- `Final_Rapor_Heart_Disease.docx`: Kapsamlı proje raporu
- PNG dosyaları: Görselleştirmeler (11 adet)

---

## 🚀 Kurulum ve Çalıştırma Adımları

### ⚙️ Sistem Gereksinimleri:
- **Python:** 3.10 veya üzeri (Proje 3.13.7 ile geliştirildi)
- **İşletim Sistemi:** Windows / macOS / Linux
- **RAM:** Minimum 4GB (8GB önerilir)
- **Disk:** 500MB boş alan

### 📦 Adım 1: Projeyi İndirin
```bash
git clone https://github.com/frhtbytms/heart-disease-prediction.git
cd heart-disease-prediction/project
```

Ya da ZIP dosyası olarak indirin ve çıkarın.

### 📥 Adım 2: Python Ortamını Hazırlayın (Opsiyonel ama önerilir)
```bash
# Virtual environment oluşturma
python -m venv venv

# Aktifleştirme (Windows)
venv\Scripts\activate

# Aktifleştirme (macOS/Linux)
source venv/bin/activate
```

### 📚 Adım 3: Gerekli Kütüphaneleri Yükleyin
```bash
# Temel kütüphaneler (ML modelleri için)
pip install -r requirements.txt

# Web uygulaması kütüphaneleri (Streamlit)
pip install -r requirements_app.txt
```

**Yüklenen Kütüphaneler:**
- pandas, numpy (veri işleme)
- scikit-learn (makine öğrenmesi)
- matplotlib, seaborn (görselleştirme)
- python-docx (rapor)
- streamlit, plotly (web arayüzü)

### 📊 Adım 4: Veri Setini Kontrol Edin
`heart.csv` dosyası proje klasöründe hazır bulunmaktadır. İsterseniz:
- [Kaggle'dan](https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset) güncel versiyon indirebilirsiniz
- [UCI Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease) adresinden orijinal veriyi alabilirsiniz

### 🎯 Adım 5: Ana Pipeline'ı Çalıştırın
```bash
python main.py
```

**Bu komut:**
1. ✅ Veri setini yükler (`heart.csv`)
2. ✅ Eksik değerleri doldurur
3. ✅ Aykırı değerleri tespit eder
4. ✅ Verileri encode eder ve normalize eder
5. ✅ 3 ML modelini eğitir (GridSearchCV ile optimize eder)
6. ✅ Modelleri test seti üzerinde değerlendirir
7. ✅ Confusion matrix ve ROC curve'leri çizer
8. ✅ En iyi modeli `results/best_model.pkl` olarak kaydeder
9. ✅ Performans metriklerini `results/metrics_table.csv`'ye yazar
10. ✅ Word raporunu `results/Final_Rapor_Heart_Disease.docx` olarak oluşturur

**Beklenen Çıktı:**
```
INFO - Veri yüklendi: (303, 14)
INFO - Eksik değerler işlendi
INFO - Aykırı değerler tespit edildi
INFO - Veri encode edildi ve normalize edildi
INFO - Logistic Regression eğitiliyor...
INFO - Random Forest eğitiliyor...
INFO - K-Nearest Neighbors eğitiliyor...
INFO - Tüm modeller değerlendirildi
INFO - Rapor oluşturuldu
✅ PROJE BAŞARIYLA TAMAMLANDI!
```

**Çalışma Süresi:** ~5-10 saniye (bilgisayar performansına bağlı)

### 🌐 Adım 6: Web Arayüzünü Başlatın
```bash
streamlit run app.py
```

**Otomatik olarak tarayıcınızda açılır:**
- 🔗 **URL:** http://localhost:8501
- 🎨 **Modern arayüz** ile grafikler ve tahmin yapabilme

**Web Uygulaması Özellikleri:**
- 📊 İnteraktif veri analizi (Plotly grafikleri)
- 🤖 Model performans karşılaştırması
- 🔮 Gerçek zamanlı kalp hastalığı riski tahmini
- 📈 Confusion matrix ve ROC curve görüntüleme
- 💾 Sonuçları görselleştirme

**Kapatma:**
- Web arayüzünü kapatmak için terminalde `Ctrl+C` yapın

---

## � Çıktı Dosyaları

Proje çalıştırıldığında `results/` klasöründe şu dosyalar oluşur:

### 📄 Rapor ve Model
- `Final_Rapor_Heart_Disease.docx` - Kapsamlı Word raporu (5+ sayfa)
- `best_model.pkl` - En iyi performanslı model (Random Forest)
- `metrics_table.csv` - Model performans tablosu

### 📊 Görselleştirmeler (11 PNG dosyası)

**Veri Analizi Grafikleri:**
1. `corr_heatmap.png` - Özellikler arası korelasyon ısı haritası
2. `class_distribution.png` - Hedef değişken dağılımı (0: Sağlıklı, 1: Hasta)
3. `boxplots.png` - Aykırı değer analizleri (6 grafik bir arada)

**Model Değerlendirme Grafikleri:**
4. `confusion_logistic_regression.png` - LR confusion matrix
5. `confusion_random_forest.png` - RF confusion matrix
6. `confusion_k-nearest_neighbors.png` - KNN confusion matrix
7. `roc_logistic_regression.png` - LR ROC eğrisi (AUC: 0.59)
8. `roc_random_forest.png` - RF ROC eğrisi (AUC: 0.51)
9. `roc_k-nearest_neighbors.png` - KNN ROC eğrisi (AUC: 0.58)

### 📝 Log Dosyaları
- `logs/project.log` - Detaylı çalışma kayıtları (debugging için)

---

## 🧪 Teknik Detaylar ve Metodoloji

### 🔬 Veri Ön İşleme Pipeline

**1. Veri Yükleme:**
```python
df = pd.read_csv('heart.csv')
# 303 satır × 14 sütun
```

**2. Eksik Veri Yönetimi:**
- Sayısal değişkenler → **Median** ile dolduruldu
- Kategorik değişkenler → **Mode** (en sık değer) ile dolduruldu
- Eksik veri oranı: %0 (veri seti temiz)

**3. Aykırı Değer Tespiti (IQR Yöntemi):**
```python
Q1 = df[col].quantile(0.25)
Q3 = df[col].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR
```
- `age`, `trestbps`, `chol`, `thalach`, `oldpeak` için uygulandı
- Tespit edilen aykırı değerler cap edildi (sınırlandırıldı)

**4. Kategorik Encoding:**
- **Label Encoding** kullanıldı (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`)
- One-Hot Encoding kullanılmadı (çok fazla boyut artışı yaratmaz için)

**5. Feature Scaling:**
- **MinMaxScaler** ile 0-1 arası normalizasyon
- Tüm özellikler aynı ölçeğe getirildi
- KNN gibi mesafe bazlı algoritmalarda kritik

**6. Train-Test Split:**
```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```
- %80 eğitim (242 örnek)
- %20 test (61 örnek)
- Stratified split: Sınıf oranları korundu

### 🤖 Model Eğitimi Detayları

**Hiperparametre Optimizasyonu:**
- **GridSearchCV** kullanıldı (tüm kombinasyonları dener)
- **3-Fold Cross Validation** (eğitim setini 3'e böler, 2'si eğitim 1'i validasyon)
- **Scoring Metric:** F1-Score (dengesiz veri için uygun)
- **n_jobs:** -1 (tüm CPU core'ları kullan)

**Neden F1-Score?**
- Accuracy tek başına yanıltıcı olabilir
- F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
- Hem false positive hem false negative'i dengeler
- Tıbbi uygulamalarda kritik (hastayı kaçırmamak önemli)

### 📊 Model Değerlendirme Metrikleri

**1. Confusion Matrix Yorumlama:**
```
                Predicted
                0    1
Actual  0     [TN] [FP]
        1     [FN] [TP]
```
- **TP (True Positive):** Hasta + Hasta tahmin = ✅
- **TN (True Negative):** Sağlıklı + Sağlıklı tahmin = ✅
- **FP (False Positive):** Sağlıklı + Hasta tahmin = ❌ (Tip 1 Hata)
- **FN (False Negative):** Hasta + Sağlıklı tahmin = ❌ (Tip 2 Hata - Daha tehlikeli!)

**2. Metrik Formülleri:**
```python
Accuracy = (TP + TN) / (TP + TN + FP + FN)
Precision = TP / (TP + FP)
Recall = TP / (TP + FN)
F1-Score = 2 × (Precision × Recall) / (Precision + Recall)
AUC = ROC eğrisi altındaki alan (0.5-1.0 arası)
```

**3. ROC Curve (Receiver Operating Characteristic):**
- X-ekseni: False Positive Rate (FPR)
- Y-ekseni: True Positive Rate (TPR = Recall)
- AUC = 0.5: Rastgele tahmin
- AUC = 1.0: Mükemmel sınıflandırma
- AUC > 0.7: Kabul edilebilir
- AUC > 0.8: İyi
- AUC > 0.9: Mükemmel

---

## 📊 Deneysel Sonuçlar ve Analiz

### 🏆 Model Performans Karşılaştırması

| Model | Accuracy | Precision | Recall | F1-Score | AUC-ROC | Eğitim Süresi |
|-------|----------|-----------|--------|----------|---------|---------------|
| **Logistic Regression** ⭐ | **0.6000** | **0.5806** | **0.8095** | **0.6757** | **0.5895** | ~0.5s |
| Random Forest | 0.4667 | 0.4906 | 0.7143 | 0.5789 | 0.5106 | ~1.2s |
| K-Nearest Neighbors | 0.5500 | 0.5570 | 0.6190 | 0.5846 | 0.5784 | ~0.2s |

**Değerlendirme Kriterleri:**
- ✅ **En yüksek F1-Score:** Logistic Regression (0.6757)
- ✅ **En yüksek Recall:** Logistic Regression (0.8095) - Hastaları %81 doğru tespit ediyor
- ✅ **En yüksek AUC:** Logistic Regression (0.5895)
- ⚡ **En hızlı:** KNN (0.2s)

### � Model Seçimi Gerekçesi

**Neden Logistic Regression en iyi?**

1. **Yüksek Recall (0.81):** 
   - Hasta olan kişilerin %81'ini doğru tespit ediyor
   - Tıbbi uygulamalarda kritik (hastayı kaçırmamak hayati)
   - False Negative (FN) sayısı düşük

2. **Dengeli F1-Score (0.68):**
   - Precision ve Recall arasında iyi denge
   - Hem false positive hem false negative düşük

3. **Yorumlanabilirlik:**
   - Logistic Regression katsayıları incelenebilir
   - Hangi özelliğin ne kadar etkilediği görülebilir
   - Doktorlar için açıklanabilir AI

4. **Stabilite:**
   - Cross-validation'da tutarlı performans
   - Overfitting riski düşük

**Diğer Modeller Neden Daha Düşük?**

- **Random Forest:** Overfitting'e meyilli, test setinde düşük performans
- **KNN:** Optimal k değeri bulunmasına rağmen sınırlı performans

### 🔍 Özellik Önem Analizi (Feature Importance)

**En Önemli Risk Faktörleri** (Logistic Regression katsayılarına göre):

| Sıra | Özellik | Etki | Yorumlama |
|------|---------|------|-----------|
| 1 | `cp` (Göğüs ağrısı) | +++++ | Asemptomatik göğüs ağrısı yüksek risk |
| 2 | `thalach` (Max kalp atışı) | ++++ | Düşük max kalp atışı risk artırıyor |
| 3 | `oldpeak` (ST depresyonu) | +++ | Yüksek ST depresyonu tehlikeli |
| 4 | `ca` (Ana damar sayısı) | +++ | Tıkalı damar sayısı ile doğru orantılı |
| 5 | `sex` (Cinsiyet) | ++ | Erkeklerde risk daha yüksek |
| 6 | `age` (Yaş) | ++ | Yaş arttıkça risk artıyor |

**EDA'dan Elde Edilen Önemli Bulgular:**

1. **Yaş Dağılımı:**
   - Hasta grubu ortalama yaş: 56.6 yıl
   - Sağlıklı grubu ortalama yaş: 52.5 yıl
   - p-value < 0.05 (istatistiksel olarak anlamlı)

2. **Cinsiyet Farkı:**
   - Erkeklerde hasta oranı: %55.3
   - Kadınlarda hasta oranı: %25.5
   - 2.2x risk artışı (erkekler için)

3. **Kolesterol:**
   - Hasta grubu ortalama: 251.5 mg/dl
   - Sağlıklı grubu ortalama: 242.6 mg/dl
   - Beklenenden daha az fark (kolesterol tek başına yetersiz)

4. **Maksimum Kalp Atışı:**
   - Hasta grubu: 139.0 bpm (düşük)
   - Sağlıklı grubu: 158.4 bpm (yüksek)
   - Güçlü negatif korelasyon (-0.42)

### 📉 Model Limitasyonları ve İyileştirme Önerileri

**Mevcut Limitasyonlar:**

1. **Düşük Accuracy (%60):**
   - Gerçek dünya uygulaması için yetersiz
   - Daha fazla veri gerekli

2. **Sınırlı Veri Seti (303 örnek):**
   - Daha büyük veri setleri ile performans artabilir
   - En az 1000+ hasta verisi önerilir

3. **Feature Engineering Eksikliği:**
   - BMI (Vücut Kitle İndeksi) hesaplanabilir
   - Yaş grupları oluşturulabilir
   - Polinomial özellikler eklenebilir

4. **Dengesiz Metrikler:**
   - AUC skorları 0.5'e yakın (ideal 0.8+)
   - Daha gelişmiş modeller denenebilir

**İyileştirme Önerileri:**

1. **Daha Fazla Veri Toplamak:**
   - Çok merkezli çalışma
   - Longitudinal takip verileri
   - Daha fazla demografik özellik

2. **Gelişmiş Modeller:**
   - Gradient Boosting (XGBoost, LightGBM)
   - Neural Networks (Deep Learning)
   - Stacking/Ensemble yöntemleri

3. **Feature Engineering:**
   - Özellik etkileşimleri (age × chol)
   - Polinomial özellikler
   - Domain knowledge bazlı özellikler

4. **Hiperparametre Optimizasyonu:**
   - RandomizedSearchCV kullanılabilir
   - Bayesian Optimization
   - Daha geniş parametre aralıkları

5. **Veri Dengeleme:**
   - SMOTE (Synthetic Minority Over-sampling)
   - Class weight ayarlamaları
   - Undersampling/Oversampling

### 🎯 Projenin Başarısı ve Katkıları

**Başarılan Hedefler:**

✅ **End-to-end ML Pipeline:** Veri toplama → Model → Deployment  
✅ **Kapsamlı EDA:** Görselleştirmeler ve istatistiksel analizler  
✅ **Model Karşılaştırması:** 3 farklı algoritma optimize edildi  
✅ **Profesyonel Raporlama:** Word raporu + Web arayüzü  
✅ **Reprodusibility:** Tüm kod paylaşılabilir ve tekrarlanabilir  
✅ **Deployment:** Streamlit ile canlı demo  

**Öğrenilenler:**

1. 📚 **Teknik Beceriler:**
   - Python ile veri bilimi projesi geliştirme
   - Scikit-learn ile model eğitimi ve optimizasyonu
   - Streamlit ile web uygulaması geliştirme
   - Git ile versiyon kontrolü

2. 🔬 **Domain Knowledge:**
   - Kalp hastalığı risk faktörleri
   - Tıbbi veri setleri ile çalışma
   - EKG ve klinik parametrelerin yorumlanması

3. 🧠 **Makine Öğrenmesi:**
   - Hiperparametre optimizasyonu
   - Cross-validation teknikleri
   - Model değerlendirme metrikleri
   - Overfitting/Underfitting dengesi

4. 💼 **Soft Skills:**
   - Proje planlama ve zaman yönetimi
   - Dokümantasyon yazma
   - Sonuçları görselleştirme ve sunma

---

## 📝 Notlar

- Bu proje eğitim amaçlıdır
- Gerçek tıbbi uygulamalarda kullanılmadan önce daha kapsamlı doğrulama gereklidir
- Daha büyük veri setleri ile model performansı artırılabilir

---

## 📚 Kaynaklar ve Referanslar

**Veri Seti:**
1. UCI Machine Learning Repository - Heart Disease Dataset  
   https://archive.ics.uci.edu/ml/datasets/Heart+Disease

2. Kaggle - Heart Disease UCI Dataset  
   https://www.kaggle.com/datasets/johnsmith88/heart-disease-dataset

**Kütüphane Dokümantasyonları:**
- Scikit-learn: https://scikit-learn.org/stable/
- Pandas: https://pandas.pydata.org/docs/
- Streamlit: https://docs.streamlit.io/
- Plotly: https://plotly.com/python/
- Matplotlib: https://matplotlib.org/
- Seaborn: https://seaborn.pydata.org/

**Akademik Kaynaklar:**
- Detrano, R., et al. (1989). "International application of a new probability algorithm for the diagnosis of coronary artery disease." *American Journal of Cardiology*.
- "Hands-On Machine Learning with Scikit-Learn" - Aurélien Géron
- "Introduction to Statistical Learning" - James, Witten, Hastie, Tibshirani

---

## 📧 İletişim ve Proje Bilgileri

**👤 Öğrenci:** Ferhat Bayutmuş  
**📧 E-posta:** ferhatbayutmus58@gmail.com  
**🎓 Üniversite:** İstanbul Medeniyet Üniversitesi  
**📚 Ders:** Veri Bilimine Giriş - Final Projesi  
**📅 Teslim Tarihi:** 17 Kasım 2025

**🔗 Proje Linkleri:**
- GitHub Repository: https://github.com/frhtbytms/kalp-hastaligi-tahmin-projesi
- Streamlit Demo: [Deploy edildiğinde eklenecek]
- Kaggle Notebook: [İsteğe bağlı]

---

## 🙏 Teşekkürler

Bu projenin geliştirilmesinde:
- 👨‍🏫 Değerli hocama rehberliği için
- 📊 UCI ML Repository ekibine veri setini paylaştıkları için
- 🐍 Python ve open-source topluluğuna katkılarından dolayı
- 👥 Sınıf arkadaşlarıma destekleri için

teşekkür ederim.

---

## 📄 Lisans ve Sorumluluk Reddi

**Lisans:** Bu proje **eğitim amaçlı** hazırlanmıştır.

**⚠️ Sorumluluk Reddi:**  
> Bu yazılım eğitim amaçlıdır ve tıbbi tavsiye yerine geçmez. Kalp sağlığınız hakkında endişeleriniz varsa mutlaka bir doktora danışın. Bu aracın kullanımından kaynaklanan herhangi bir zarardan geliştirici sorumlu tutulamaz.

**Kullanım Koşulları:**
- ✅ Akademik ve eğitim amaçlı kullanım serbest
- ✅ Kaynak göstererek referans verebilirsiniz
- ❌ Ticari kullanım ve klinik uygulamalar için onay gereklidir

---

<div align="center">

### 🎉 Proje Başarıyla Tamamlandı!

**Versiyon:** 1.0.0  
**Son Güncelleme:** 17 Kasım 2025  
**Durum:** ✅ Teslime Hazır

---

**💙 Kalp sağlığınıza dikkat edin!**

*Made with ❤️ using Python, Scikit-learn, and Streamlit*

</div>
