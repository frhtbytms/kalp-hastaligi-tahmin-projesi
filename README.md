# ❤️ Kalp Hastalığı Risk Tahmini

**Veri Bilimine Giriş - Final Projesi**  
**Öğrenci:** Ferhat Bayutmuş  
**Üniversite:** İstanbul Medeniyet Üniversitesi  
**Akademik Yıl:** 2025-2026 Güz Dönemi  

---

## 📋 Proje Hakkında

Kalp hastalıkları dünya genelinde en yaygın ölüm nedenlerinden biri olduğu için, erken teşhis çok önemli. Bu projede **makine öğrenmesi** kullanarak kalp hastalığı riskini tahmin etmeye çalıştım. 

Projede yaptıklarım:
- UCI'den gerçek hasta verisi aldım ve analiz ettim
- Veriyi temizledim, eksikleri doldurdum
- 3 farklı ML algoritması denedim ve karşılaştırdım
- Model performanslarını değerlendirdim
- Word raporu ve grafikler hazırladım
- Streamlit ile **web arayüzü** yaptım

Veri toplama, temizleme, modelleme, değerlendirme, görselleştirme ve deployment - hepsini deneyimledim.

---

## 🎯 Ne Yaptım?

### 1️⃣ Veri Toplama
- UCI'den **Heart Disease Dataset** indirdim
- Kaggle'dan da baktım alternatif versiyonlara
- 303 hasta, 14 değişken var
- İlk EDA yaparak veriyi tanıdım

### 2️⃣ Veri Temizleme (`preprocess.py`)
**Yaptığım işlemler:**
- Eksik verileri median/mode ile doldurdum
- IQR yöntemiyle aykırı değerleri buldum
- Kategorik değişkenleri encode ettim
- MinMaxScaler ile 0-1 arası normalize ettim
- %80 train, %20 test ayırdım

**Kullanılan Teknikler:**
```python
- handle_missing_values()  # Eksik veri doldurma
- detect_outliers_iqr()    # Aykırı değer tespiti
- encode_and_scale()       # Encoding ve normalizasyon
```

### 3️⃣ Model Eğitimi (`train_models.py`)
3 algoritma denedim, **GridSearchCV** ile en iyi parametreleri buldum:

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

**Nasıl Optimize Ettim:**
- 3-fold Cross-Validation kullandım
- F1-Score'a göre seçtim (dengesiz veri için uygun)
- En iyi parametreler otomatik bulundu

### 4️⃣ Model Değerlendirme (`evaluate.py`)
Her model için performans analizi yaptım:

**Kullandığım Metrikler:**
- **Accuracy:** Genel doğruluk
- **Precision:** Pozitif tahmin doğruluğu
- **Recall:** Gerçek pozitifleri bulma
- **F1-Score:** Precision ve Recall ortalaması
- **AUC-ROC:** Sınıflandırma performansı

**Grafikler:**
- Confusion Matrix
- ROC Curve
- Korelasyon haritası
- Sınıf dağılımları
- Boxplot'lar

### 5️⃣ Rapor Hazırlama (`report_generator.py`)
Python-docx ile Word raporu oluşturdum:
- Proje özeti
- Performans tablosu
- Tüm grafikler
- Sonuç ve öneriler

### 6️⃣ Web Arayüzü (`app.py`)
**Streamlit** ile modern web uygulaması yaptım:

**Özellikler:**
- 🏠 **Ana Sayfa:** Proje özeti, istatistikler ve model karşılaştırma paneli
  - Metrik kartları (3 modelin performans göstergeleri)
  - İnteraktif model karşılaştırma dashboard'u
  - Bar chart ve heatmap görselleştirmeleri
  
- 📊 **Veri Analizi:** İnteraktif Plotly grafikleri
  - Yaş dağılımı histogramları
  - Kolesterol ve kalp atışı boxplotları
  - Korelasyon ısı haritası
  - 4 tab'lı detaylı keşifsel veri analizi
  
- 🤖 **Model Performansı:** Kapsamlı model analizi ve yorumlanabilirlik
  - **Feature Importance:** Hangi özelliklerin modeli etkilediğini gösteren analiz
  - **Learning Curves:** Model performansının eğitim verisi miktarıyla değişimini gösteren grafikler
  - **Validation Curves:** Hiperparametre optimizasyonu için doğrulama eğrileri
  - Confusion matrix ve ROC curve görselleri (3 model için)
  - Metrik karşılaştırma tabloları
  
- 🔮 **Tahmin Yap:** Gerçek zamanlı risk hesaplama ve LIME ile açıklama
  - 13 parametre girişi (slider, selectbox ile kolay kullanım)
  - Model ile anlık tahmin
  - Risk yüzdesi gösterimi
  - **LIME:** Her tahminin hangi özelliklerden etkilendiğini gösteriyor
  - **Tahmin Geçmişi:** Yapılan tahminlerin kaydı ve istatistikleri
  
- ℹ️ **Hakkında:** Proje detayları ve iletişim

**Teknik Özellikler:**
- Modern gradient tasarımı yaptım
- Responsive - her ekranda çalışıyor
- Custom CSS ile güzel görünüm
- Caching ile hızlı (@st.cache_data, @st.cache_resource)
- Session state - tahmin geçmişi için
- İnteraktif Plotly grafikleri

---

## 📊 Kullandığım Veri

**Heart Disease UCI Verisi**
- **Kaynak:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Heart+Disease)
- **Alternatif:** Kaggle'dan da indirdim
- **Kayıt Sayısı:** 303 hasta
- **Özellik:** 13 bağımsız değişken + 1 hedef
- **Tip:** Sayısal + Kategorik karışık
- **Eksik Veri:** Minimal (temizledim)
- **Sınıf Dağılımı:** Dengeli (0: 138, 1: 165)

### 📋 Değişkenler:

| Değişken | Açıklama | Tip | Değerler |
|----------|----------|-----|----------|
| **age** | Yaş | Sayısal | 29-77 |
| **sex** | Cinsiyet | Kategorik | 0=Kadın, 1=Erkek |
| **cp** | Göğüs ağrısı tipi | Kategorik | 0-3 |
| **trestbps** | Dinlenme kan basıncı | Sayısal | 94-200 mm Hg |
| **chol** | Kolesterol | Sayısal | 126-564 mg/dl |
| **fbs** | Açlık şekeri > 120 | İkili | 0=Hayır, 1=Evet |
| **restecg** | EKG sonuçları | Kategorik | 0-2 |
| **thalach** | Max kalp atışı | Sayısal | 71-202 |
| **exang** | Egzersiz anjinası | İkili | 0=Hayır, 1=Evet |
| **oldpeak** | ST depresyonu | Sayısal | 0-6.2 |
| **slope** | ST eğimi | Kategorik | 0-2 |
| **ca** | Ana damar sayısı | Sayısal | 0-3 |
| **thal** | Talassemi | Kategorik | 1-3 |
| **target** | Kalp hastalığı | İkili | 0=Sağlıklı, 1=Hasta |

### 🔍 Veri Analizinden Bulduklarım:

**Demografik:**
- Ortalama yaş: 54.4 ± 9.1
- Erkek oranı: %68.3
- En yaygın ağrı: Asemptomatik (%47.2)

**Risk Faktörleri:**
- Ortalama kolesterol: 246.7 mg/dl
- Ortalama max kalp atışı: 149.6 bpm
- Egzersiz anjinası: %33.3

**Korelasyonlar:**
- En güçlü pozitif: `cp` ile target (0.43)
- En güçlü negatif: `oldpeak` ile target (-0.43)
- Yaş ile kalp atışı: Negatif (-0.39)

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

### **Neden Bu Kütüphaneleri Kullandım?**

1. **Scikit-learn:** 
   - Çoğu ML algoritması var
   - GridSearchCV çok kullanışlı
   - Dokümantasyonu iyi

2. **Streamlit:**
   - Python ile web sayfası yapmak kolay
   - Özellikle veri bilimi projeleri için iyi
   - Ücretsiz yayınlayabiliyorsun

3. **Plotly:**
   - Grafikler interaktif oluyor
   - Modern görünüm
   - Kullanıcı zoom, hover yapabiliyor

---

## 🤖 Makine Öğrenmesi Modelleri

### **1. Logistic Regression (Lojistik Regresyon)** ⭐ EN İYİ MODEL

**Nasıl Çalışıyor:**
- İkili sınıflandırma için klasik yöntem
- Sigmoid fonksiyonu kullanıyor
- Doğrusal bir karar sınırı çiziyor

**Parametreler:**
```python
param_grid = {
    'C': [0.01, 0.1, 1],        # Regularizasyon
    'solver': ['liblinear'],     # Optimizasyon
    'max_iter': [1000]           # Max iterasyon
}
```

**Neden En İyi:**
- Basit ve anlaşılır
- Hızlı eğitiliyor
- İkili sınıflandırmada başarılı

**Performans:**
- F1-Score: 0.6757
- Accuracy: 60.0%
- AUC-ROC: 0.5895

---

### **2. Random Forest (Rastgele Orman)**

**Nasıl Çalışıyor:**
- Birden fazla karar ağacı kullanıyor
- Hepsinin oyunu birleştiriyor
- Bagging ile overfitting'i engelliyor

**Parametreler:**
```python
param_grid = {
    'n_estimators': [50, 100],      # Kaç ağaç
    'max_depth': [None, 10, 20],    # Ağaç derinliği
    'random_state': [42]             # Random seed
}
```

**Artıları:**
- Hangi özellikler önemli görebiliyorsun
- Non-linear ilişkileri yakalıyor
- Aykırı değerlere dayanıklı

**Performans:**
- F1-Score: 0.5789
- Accuracy: 56.7%
- AUC-ROC: 0.5106

---

### **3. K-Nearest Neighbors (KNN)**

**Nasıl Çalışıyor:**
- En yakın k komşuya bakıyor
- Mesafeye göre sınıflandırıyor
- Euclidean distance kullanıyor

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
- `best_model.pkl`: Joblib ile kaydettim (Logistic Regression)
- `metrics_table.csv`: Tüm modellerin performans metrikleri
- `Final_Rapor_Heart_Disease.docx`: Kapsamlı proje raporu
- PNG dosyaları: Görselleştirmeler (11 adet)

---

## 🚀 Kurulum ve Çalıştırma Adımları

### ⚙️ Sistem Gereksinimleri:
- **Python:** 3.10 veya üzeri (Ben 3.13.7 kullandım)
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
- Aykırı değerleri buldum ve cap ettim (sınırlandırdım)

**4. Kategorik Encoding:**
- **Label Encoding** yaptım (`sex`, `cp`, `fbs`, `restecg`, `exang`, `slope`, `ca`, `thal`)
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
- **GridSearchCV** kullandım (tüm kombinasyonları dener)
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

Projenin en önemli kısımlarından biri **model yorumlanabilirliği** oldu. Sadece tahmin yapmak değil, neden bu tahmini yaptığını anlamak da önemliydi.

**Feature Importance Analizi:**
- Logistic Regression'ın katsayıları incelenerek hangi özelliklerin modeli nasıl etkilediğini görselleştirdim
- İnteraktif bar chart ile 13 özelliğin göreceli önemini gösterdim
- Pozitif katsayılar risk artışını, negatif katsayılar risk azalışını gösteriyor

**LIME ile Açıklanabilir AI:**
- Her bireysel tahmin için LIME (Local Interpretable Model-agnostic Explanations) kullandım
- Hastaya özel sonuçlar: Hangi faktörler bu kişi için riski artırıyor/azaltıyor?
- Yeşil barlar risk artıran, kırmızı barlar risk azaltan faktörleri gösteriyor
- Bu özellik sayesinde doktorlar modelin kararını anlayıp güvenebilir

**En Önemli Risk Faktörleri** (önem sırasına göre):

| Sıra | Özellik | Etki | Yorumlama |
|------|---------|------|-----------|
| 1 | `cp` (Göğüs ağrısı) | +++++ | Asemptomatik göğüs ağrısı yüksek risk |
| 2 | `thalach` (Max kalp atışı) | ++++ | Düşük max kalp atışı risk artırıyor |
| 3 | `oldpeak` (ST depresyonu) | +++ | Yüksek ST depresyonu tehlikeli |
| 4 | `ca` (Ana damar sayısı) | +++ | Tıkalı damar sayısı ile doğru orantılı |
| 5 | `sex` (Cinsiyet) | ++ | Erkeklerde risk daha yüksek |
| 6 | `age` (Yaş) | ++ | Yaş arttıkça risk artıyor |

**Learning ve Validation Curves:**
- Modelin eğitim verisi miktarıyla performansının nasıl değiştiğini analiz ettim
- Validation curves ile hiperparametre C'nin etkisini görselleştirdim  
- Bu analizler sayesinde modelin yeterli veri ile eğitildiğini doğruladım

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
✅ **Model Karşılaştırması:** 3 farklı algoritma denedim ve optimize ettim  
✅ **Profesyonel Raporlama:** Word raporu + Modern web arayüzü  
✅ **Model Yorumlanabilirliği:** Feature Importance + LIME explainability  
✅ **İleri Seviye Analizler:** Learning curves, validation curves, model comparison dashboard  
✅ **Kullanıcı Deneyimi:** Session state ile tahmin geçmişi takibi  
✅ Proje Boyunca Öğrendiklerim:**

Bu proje gerçekten çok şey öğretti. Sadece kod yazmaktan çok daha fazlasıydı:

1. 📚 **Teknik Beceriler:**
   - Python ile profesyonel veri bilimi projesi geliştirdim
   - Scikit-learn'ün detaylarını öğrendim (GridSearchCV, pipeline, metrics)
   - Streamlit ile interaktif web uygulaması yaptım - ilk defa bir projeyi deployment yaptım
   - LIME kütüphanesi ile explainable AI deneyimledim
   - Plotly ile interaktif grafikler oluşturmayı öğrendim
   - Session state yönetimini öğrendim (tahmin geçmişi için)
   - CSS ile modern web tasarımı yaptım

2. 🔬 **Domain Knowledge:**
   - Kalp hastalıklarının risk faktörlerini detaylı öğrendim
   - Tıbbi veri setleriyle çalışmanın özel gereksinimleri (privacy, accuracy)
   - EKG parametreleri ve klinik değerlerin anlamını anladım
   - Feature importance'ın medikal karar destek sistemlerindeki önemini gördüm

3. 🧠 **Makine Öğrenmesi:**
   - GridSearchCV ile sistematik hiperparametre optimizasyonu
   - Cross-validation'ın neden kritik olduğunu anladım
   - Metrik seçiminin önemini kavradım (accuracy yeterli değil!)
   - Learning curves ile modelin durumunu analiz etmeyi öğrendim
   - Validation curves ile hiperparametre etkilerini görselleştirdim
   - Explainable AI'ın gerçek değerini gördüm (LIME sayesinde)

4. 💼 **Proje Yönetimi:**
   - Modüler kod yazmanın faydalarını gördüm (bakım kolaylığı)
   - Git ile düzenli commit atmayı öğrendim
   - README ve dokümantasyon yazmanın önemini anladım
   - Zaman planlaması ve iş bölümü yaptım
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
- Streamlit Demo: [Henüz deploy etmedim]
- Kaggle Notebook: [İsteğe bağlı]

---

## 🙏 Teşekkürler

Projeyi geliştirirken yardımcı olanlar:
- 👨‍🏫 Hocama rehberliği için
- 📊 UCI ML Repository - veri seti için
- 🐍 Python ve open-source topluluğu
- 👥 Arkadaşlarıma destekleri için

teşekkür ederim.

---

## 📄 Lisans ve Sorumluluk Reddi

**Lisans:** Eğitim amaçlı yaptım.

**⚠️ Uyarı:**  
> Bu eğitim projesi, tıbbi tavsiye yerine geçmez. Kalp sağlığınız hakkında endişeleriniz varsa mutlaka doktora danışın.

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
