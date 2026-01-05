"""
Kalp Hastalığı Tahmin Projesi - Ana Program
Veri Bilimine Giriş Final Projesi

UCI veri seti ile 3 ML algoritması kullanıyorum:
Logistic Regression, Random Forest, KNN

Ferhat Bayutmuş - İMÜ - 2025
"""

import os
import sys
import logging
import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Kendi modüllerimizi import et
from preprocess import (
    load_data, inspect_missing_values, handle_missing_values,
    detect_outliers_iqr, encode_and_scale
)
from train_models import (
    split_data, train_all_models, save_best_model
)
from evaluate import (
    evaluate_all_models, create_all_visualizations
)
from report_generator import (
    generate_report, create_summary_report
)

# Logging ayarlarını yapılandır
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('main.log'),
        logging.StreamHandler()
    ]
)

def setup_directories():
    """
    Gerekli klasörleri oluşturur.
    """
    directories = ['results', 'logs']
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
            logging.info(f"Klasör oluşturuldu: {directory}")

def check_data_file(data_path="heart.csv"):
    """
    Veri dosyasının varlığını kontrol eder.
    
    Args:
        data_path (str): Veri dosyası yolu
        
    Returns:
        bool: Dosya var mı?
    """
    if os.path.exists(data_path):
        logging.info(f"Veri dosyası bulundu: {data_path}")
        return True
    else:
        print(f"⚠ UYARI: '{data_path}' dosyası bulunamadı!")
        print("\nLütfen aşağıdaki adımları takip edin:")
        print("1. Kaggle'dan 'Heart Disease UCI' veri setini indirin")
        print("2. CSV dosyasını 'heart.csv' olarak kaydedin")
        print("3. Dosyayı proje klasörüne koyun")
        print("4. Programı yeniden çalıştırın")
        print("\nVeri seti linki: https://www.kaggle.com/datasets/ronitf/heart-disease-uci")
        
        # Örnek veri seti oluştur (test amaçlı)
        create_sample_dataset(data_path)
        return True

def create_sample_dataset(file_path="heart.csv"):
    """
    Test amaçlı örnek veri seti oluşturur.
    
    Args:
        file_path (str): Oluşturulacak dosya yolu
    """
    print("\n🔄 Test amaçlı örnek veri seti oluşturuluyor...")
    
    # Örnek kalp hastalığı verisi oluştur
    np.random.seed(42)
    n_samples = 300
    
    data = {
        'age': np.random.randint(25, 80, n_samples),
        'sex': np.random.randint(0, 2, n_samples),
        'cp': np.random.randint(0, 4, n_samples),
        'trestbps': np.random.randint(90, 200, n_samples),
        'chol': np.random.randint(120, 400, n_samples),
        'fbs': np.random.randint(0, 2, n_samples),
        'restecg': np.random.randint(0, 3, n_samples),
        'thalach': np.random.randint(70, 200, n_samples),
        'exang': np.random.randint(0, 2, n_samples),
        'oldpeak': np.random.uniform(0, 6, n_samples),
        'slope': np.random.randint(0, 3, n_samples),
        'ca': np.random.randint(0, 4, n_samples),
        'thal': np.random.choice([1, 2, 3], n_samples),
        'target': np.random.randint(0, 2, n_samples)
    }
    
    df = pd.DataFrame(data)
    df.to_csv(file_path, index=False)
    
    print(f"✓ Örnek veri seti oluşturuldu: {file_path}")
    print(f"  - Satır sayısı: {n_samples}")
    print(f"  - Sütun sayısı: {len(data)}")
    logging.info(f"Örnek veri seti oluşturuldu: {file_path}")

def main():
    """
    Ana çalıştırma fonksiyonu - tüm pipeline'ı çalıştırır.
    """
    start_time = datetime.now()
    
    print("=" * 70)
    print(" KALP HASTALIĞI TAHMİNİ PROJESİ")
    print(" Veri Bilimine Giriş - Final Projesi")  
    print("=" * 70)
    print(f" Başlangıç Zamanı: {start_time.strftime('%d.%m.%Y %H:%M:%S')}")
    print()
    
    try:
        # 1. Klasörleri hazırla
        print("📁 1. Klasör yapısı kontrol ediliyor...")
        setup_directories()
        
        # 2. Veri dosyasını kontrol et
        print("\n📋 2. Veri dosyası kontrol ediliyor...")
        data_path = "heart.csv"
        if not check_data_file(data_path):
            return
        
        # 3. Veri yükleme ve ön işleme
        print("\n🔄 3. Veri yükleme ve ön işleme...")
        print("-" * 40)
        
        # Veriyi yükle
        df_raw = load_data(data_path)
        print(f"✓ Veri yüklendi: {df_raw.shape}")
        
        # Eksik değer analizi
        inspect_missing_values(df_raw)
        
        # Eksik değerleri temizle
        df_clean = handle_missing_values(df_raw)
        
        # Sayısal sütunlar için aykırı değer analizi
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        outlier_info = detect_outliers_iqr(df_clean, numeric_cols)
        
        # Görselleştirmeleri oluştur (ham veri için)
        print("\n🎨 Veri görselleştirmeleri oluşturuluyor...")
        create_all_visualizations(df_clean)
        
        # Kodlama ve ölçeklendirme
        X_scaled, y, feature_names = encode_and_scale(df_clean)
        print(f"✓ Veri ön işleme tamamlandı: {X_scaled.shape}")
        
        # 4. Veriyi böl
        print("\n📊 4. Veri bölme (Train/Test Split)...")
        print("-" * 40)
        X_train, X_test, y_train, y_test = split_data(X_scaled, y, test_size=0.2, random_state=42)
        
        # 5. Model eğitimi
        print("\n🤖 5. Model eğitimi ve hiperparametre optimizasyonu...")
        print("-" * 40)
        models, results = train_all_models(X_train, y_train)
        
        if not models:
            print("❌ Hiçbir model eğitilemedi!")
            return
        
        # 6. Model değerlendirme
        print("\n📈 6. Model değerlendirme...")
        print("-" * 40)
        metrics_df = evaluate_all_models(models, X_test, y_test)
        
        # 7. En iyi modeli seç ve kaydet
        print("\n🏆 7. En iyi model seçimi...")
        print("-" * 40)
        best_model_name, best_model, best_score = save_best_model(models, results)
        
        # 8. Rapor oluşturma
        print("\n📄 8. Rapor oluşturuluyor...")
        print("-" * 40)
        
        # Grafik dosyalarının listesi
        figures_paths = [
            "results/corr_heatmap.png",
            "results/class_distribution.png", 
            "results/boxplots.png",
            "results/confusion_logistic_regression.png",
            "results/confusion_random_forest.png",
            "results/confusion_k-nearest_neighbors.png",
            "results/roc_logistic_regression.png",
            "results/roc_random_forest.png",
            "results/roc_k-nearest_neighbors.png"
        ]
        
        # Word raporu oluştur
        report_success = generate_report(
            metrics_df=metrics_df,
            best_model_name=best_model_name,
            figures_paths=figures_paths,
            output_path="results/Final_Rapor_Heart_Disease.docx"
        )
        
        # 9. Özet rapor
        create_summary_report(metrics_df, best_model_name)
        
        # Süre hesaplama
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n⏱ Toplam Süre: {duration}")
        print(f"🏁 Bitiş Zamanı: {end_time.strftime('%d.%m.%Y %H:%M:%S')}")
        
        # Final mesajları
        print("\n" + "="*60)
        print("🎉 PROJE BAŞARIYLA TAMAMLANDI!")
        print("="*60)
        print("📁 Oluşturulan dosyalar:")
        
        output_files = [
            "results/metrics_table.csv",
            "results/best_model.pkl", 
            "results/Final_Rapor_Heart_Disease.docx",
            "results/corr_heatmap.png",
            "results/class_distribution.png",
            "results/boxplots.png"
        ]
        
        for file_path in output_files:
            if os.path.exists(file_path):
                print(f"  ✓ {file_path}")
            else:
                print(f"  ⚠ {file_path} (oluşturulamadı)")
        
        print("\n📋 SONRAKI ADIMLAR:")
        print("1. 'results/Final_Rapor_Heart_Disease.docx' dosyasını açın")
        print("2. Raporun başındaki öğrenci bilgilerini doldurun")
        print("3. Gerekirse ek yorumlar ekleyin")
        print("4. Projeyi hocaya teslim edin")
        print("\n🔬 Model kullanımı için 'best_model.pkl' dosyasını joblib ile yükleyebilirsiniz.")
        
    except Exception as e:
        logging.error(f"Ana program hatası: {e}", exc_info=True)
        print(f"\n❌ HATA: {e}")
        print("Detaylı hata bilgileri 'main.log' dosyasında bulunabilir.")
        
    finally:
        print(f"\nLog dosyaları: main.log, preprocess.log, train_models.log, evaluate.log")

if __name__ == "__main__":
    main()