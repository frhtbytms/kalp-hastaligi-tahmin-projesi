"""
Veri Ön İşleme Modülü
Kalp Hastalığı Tahmini Projesi

Bu modül, ham veri setini makine öğrenmesi modellerine hazır hale getirir.
Eksik veri doldurma, aykırı değer tespiti, encoding ve normalizasyon işlemlerini içerir.

Fonksiyonlar:
- load_data(): CSV dosyasını okur
- handle_missing_values(): Eksik verileri doldurur
- detect_outliers_iqr(): IQR yöntemi ile aykırı değerleri tespit eder
- encode_and_scale(): Kategorik kodlama ve ölçeklendirme yapar

Geliştirici: Ferhat Bayutmuş
Tarih: 17 Kasım 2025
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import LabelEncoder

# Logging ayarlarını yapılandır
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('preprocess.log'),
        logging.StreamHandler()
    ]
)

def load_data(path: str) -> pd.DataFrame:
    """
    Verilen dosya yolundan heart.csv dosyasını okur.
    
    Args:
        path (str): CSV dosyasının yolu
        
    Returns:
        pd.DataFrame: Yüklenen veri seti
    """
    try:
        df = pd.read_csv(path)
        logging.info(f"Veri seti başarıyla yüklendi. Shape: {df.shape}")
        logging.info(f"Sütun isimleri: {list(df.columns)}")
        logging.info("İlk 5 satır:")
        logging.info(f"\n{df.head()}")
        return df
    except Exception as e:
        logging.error(f"Veri yükleme hatası: {e}")
        raise

def inspect_missing_values(df: pd.DataFrame) -> None:
    """
    Eksik değer sayısını her sütun için hesaplar ve raporlar.
    
    Args:
        df (pd.DataFrame): İncelenecek veri seti
    """
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    
    missing_report = pd.DataFrame({
        'Sütun': missing_values.index,
        'Eksik_Değer_Sayısı': missing_values.values,
        'Eksik_Değer_Yüzdesi': missing_percentage.values
    })
    
    print("\n=== EKSİK DEĞER ANALİZİ ===")
    print(missing_report)
    logging.info(f"Eksik değer raporu:\n{missing_report}")
    
    total_missing = missing_values.sum()
    if total_missing == 0:
        print("✓ Veri setinde eksik değer bulunmamaktadır.")
        logging.info("Veri setinde eksik değer bulunamadı.")
    else:
        print(f"⚠ Toplam eksik değer sayısı: {total_missing}")
        logging.warning(f"Toplam eksik değer sayısı: {total_missing}")

def handle_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """
    Eksik değerleri uygun yöntemlerle doldurur.
    
    Args:
        df (pd.DataFrame): Temizlenecek veri seti
        
    Returns:
        pd.DataFrame: Temizlenmiş veri seti
    """
    df_clean = df.copy()
    
    # Sayısal sütunları belirle
    numeric_cols = df_clean.select_dtypes(include=[np.number]).columns
    # Kategorik sütunları belirle
    categorical_cols = df_clean.select_dtypes(include=['object']).columns
    
    # Sayısal sütunlarda eksik değerleri median ile doldur
    for col in numeric_cols:
        if df_clean[col].isnull().sum() > 0:
            median_value = df_clean[col].median()
            df_clean[col].fillna(median_value, inplace=True)
            logging.info(f"'{col}' sütunundaki eksik değerler median ({median_value}) ile dolduruldu.")
    
    # Kategorik sütunlarda eksik değerleri mode ile doldur
    for col in categorical_cols:
        if df_clean[col].isnull().sum() > 0:
            mode_value = df_clean[col].mode()[0]
            df_clean[col].fillna(mode_value, inplace=True)
            logging.info(f"'{col}' sütunundaki eksik değerler mode ({mode_value}) ile dolduruldu.")
    
    logging.info("Eksik değer temizleme işlemi tamamlandı.")
    return df_clean

def detect_outliers_iqr(df: pd.DataFrame, numeric_cols: list) -> dict:
    """
    IQR yöntemini kullanarak aykırı değerleri tespit eder.
    
    Args:
        df (pd.DataFrame): İncelenecek veri seti
        numeric_cols (list): Sayısal sütun isimleri
        
    Returns:
        dict: Her sütun için aykırı değer bilgileri
    """
    outlier_info = {}
    
    print("\n=== AYKIRI DEĞER ANALİZİ (IQR Yöntemi) ===")
    logging.info("Aykırı değer analizi başlatıldı.")
    
    for col in numeric_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)][col]
        outlier_count = len(outliers)
        
        outlier_info[col] = {
            'Q1': Q1,
            'Q3': Q3,
            'IQR': IQR,
            'alt_sınır': lower_bound,
            'üst_sınır': upper_bound,
            'aykırı_değer_sayısı': outlier_count,
            'aykırı_değerler': outliers.tolist()
        }
        
        print(f"{col}: {outlier_count} aykırı değer [Alt: {lower_bound:.2f}, Üst: {upper_bound:.2f}]")
        logging.info(f"{col} sütunu için {outlier_count} aykırı değer tespit edildi.")
    
    return outlier_info

def encode_and_scale(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, list]:
    """
    Kategorik değişkenleri kodlar ve özellikleri ölçeklendirir.
    
    Args:
        df (pd.DataFrame): İşlenecek veri seti
        
    Returns:
        Tuple: (X_scaled, y, feature_names)
    """
    df_processed = df.copy()
    
    # Hedef değişkeni ayır
    if 'target' in df_processed.columns:
        y = df_processed['target'].values
        X = df_processed.drop('target', axis=1)
        logging.info("Hedef değişken 'target' olarak belirlendi.")
    else:
        logging.error("Hedef değişken 'target' bulunamadı!")
        raise ValueError("Hedef değişken 'target' bulunamadı!")
    
    # Kategorik değişkenleri encode et
    categorical_cols = X.select_dtypes(include=['object']).columns
    le = LabelEncoder()
    
    for col in categorical_cols:
        X[col] = le.fit_transform(X[col].astype(str))
        logging.info(f"'{col}' sütunu Label Encoding ile dönüştürüldü.")
    
    # Özellik isimlerini kaydet
    feature_names = list(X.columns)
    
    # Özellikleri ölçeklendir
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)
    
    logging.info(f"Özellikler MinMaxScaler ile ölçeklendirildi. Final shape: {X_scaled.shape}")
    logging.info(f"Hedef değişken dağılımı: {np.bincount(y)}")
    
    return X_scaled, y, feature_names

if __name__ == "__main__":
    # Test için örnek kullanım
    try:
        # Veri setini yükle (bu dosya projenizde olmalı)
        df = load_data("heart.csv")
        
        # Eksik değer analizi
        inspect_missing_values(df)
        
        # Eksik değerleri temizle
        df_clean = handle_missing_values(df)
        
        # Sayısal sütunları belirle
        numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in numeric_cols:
            numeric_cols.remove('target')
        
        # Aykırı değer analizi
        outlier_info = detect_outliers_iqr(df_clean, numeric_cols)
        
        # Kodlama ve ölçeklendirme
        X_scaled, y, feature_names = encode_and_scale(df_clean)
        
        print(f"\n✓ Veri ön işleme tamamlandı!")
        print(f"  - Özellik sayısı: {X_scaled.shape[1]}")
        print(f"  - Örnek sayısı: {X_scaled.shape[0]}")
        print(f"  - Hedef sınıf dağılımı: {np.bincount(y)}")
        
    except FileNotFoundError:
        print("⚠ heart.csv dosyası bulunamadı. Lütfen veri setini projeye ekleyin.")
    except Exception as e:
        print(f"❌ Hata: {e}")