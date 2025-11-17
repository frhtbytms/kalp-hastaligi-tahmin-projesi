"""
Makine Öğrenmesi Model Eğitimi Modülü
Kalp Hastalığı Tahmini Projesi

Bu modül, 3 farklı makine öğrenmesi algoritmasını eğitir:
1. Logistic Regression - İkili sınıflandırma için klasik yöntem
2. Random Forest - Ensemble learning yaklaşımı
3. K-Nearest Neighbors - Mesafe tabanlı sınıflandırma

Her model için GridSearchCV ile hiperparametre optimizasyonu yapılır.
En iyi performans gösteren model kaydedilir.

Geliştirici: Ferhat Bayutmuş
Tarih: 17 Kasım 2025
"""

import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
import joblib
import warnings
warnings.filterwarnings('ignore')

# Logging ayarlarını yapılandır
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('train_models.log'),
        logging.StreamHandler()
    ]
)

def split_data(X, y, test_size=0.2, random_state=42):
    """
    Veriyi eğitim ve test setlerine böler.
    
    Args:
        X: Özellik matrisi
        y: Hedef değişken
        test_size (float): Test seti boyutu (varsayılan 0.2)
        random_state (int): Rastgele tohum değeri
        
    Returns:
        tuple: X_train, X_test, y_train, y_test
    """
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    logging.info(f"Veri bölündü: Eğitim {X_train.shape[0]} örneklem, Test {X_test.shape[0]} örneklem")
    logging.info(f"Eğitim seti sınıf dağılımı: {np.bincount(y_train)}")
    logging.info(f"Test seti sınıf dağılımı: {np.bincount(y_test)}")
    
    return X_train, X_test, y_train, y_test

def train_logistic_regression(X_train, y_train):
    """
    Logistic Regression modelini eğitir ve hiperparametre optimizasyonu yapar.
    
    Args:
        X_train: Eğitim özellikleri
        y_train: Eğitim hedef değişkeni
        
    Returns:
        tuple: (en_iyi_model, en_iyi_parametreler, cv_skorları)
    """
    logging.info("Logistic Regression eğitimi başlatıldı...")
    
    # Hiperparametre grid'i (basitleştirilmiş)
    param_grid = {
        'C': [0.1, 1, 10],
        'penalty': ['l2'],
        'solver': ['liblinear']
    }
    
    # GridSearchCV ile en iyi parametreleri bul
    lr = LogisticRegression(random_state=42, max_iter=1000)
    grid_search = GridSearchCV(
        lr, param_grid, cv=3, scoring='f1', n_jobs=1
    )
    grid_search.fit(X_train, y_train)
    
    # En iyi modeli al
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Cross-validation skorlarını hesapla
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')
    
    logging.info(f"Logistic Regression - En iyi parametreler: {best_params}")
    logging.info(f"Logistic Regression - CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return best_model, best_params, cv_scores

def train_random_forest(X_train, y_train):
    """
    Random Forest modelini eğitir ve hiperparametre optimizasyonu yapar.
    
    Args:
        X_train: Eğitim özellikleri
        y_train: Eğitim hedef değişkeni
        
    Returns:
        tuple: (en_iyi_model, en_iyi_parametreler, cv_skorları)
    """
    logging.info("Random Forest eğitimi başlatıldı...")
    
    # Hiperparametre grid'i (basitleştirilmiş)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [3, 5, 10],
        'min_samples_split': [2, 5]
    }
    
    # GridSearchCV ile en iyi parametreleri bul
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(
        rf, param_grid, cv=3, scoring='f1', n_jobs=1
    )
    grid_search.fit(X_train, y_train)
    
    # En iyi modeli al
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Cross-validation skorlarını hesapla
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')
    
    logging.info(f"Random Forest - En iyi parametreler: {best_params}")
    logging.info(f"Random Forest - CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return best_model, best_params, cv_scores

def train_knn(X_train, y_train):
    """
    K-Nearest Neighbors modelini eğitir ve hiperparametre optimizasyonu yapar.
    
    Args:
        X_train: Eğitim özellikleri
        y_train: Eğitim hedef değişkeni
        
    Returns:
        tuple: (en_iyi_model, en_iyi_parametreler, cv_skorları)
    """
    logging.info("K-Nearest Neighbors eğitimi başlatıldı...")
    
    # Hiperparametre grid'i (basitleştirilmiş)
    param_grid = {
        'n_neighbors': [3, 5, 7],
        'weights': ['uniform', 'distance']
    }
    
    # GridSearchCV ile en iyi parametreleri bul
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(
        knn, param_grid, cv=3, scoring='f1', n_jobs=1
    )
    grid_search.fit(X_train, y_train)
    
    # En iyi modeli al
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    
    # Cross-validation skorlarını hesapla
    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring='f1')
    
    logging.info(f"KNN - En iyi parametreler: {best_params}")
    logging.info(f"KNN - CV F1-Score: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return best_model, best_params, cv_scores

def train_all_models(X_train, y_train):
    """
    Tüm modelleri eğitir ve sonuçları döndürür.
    
    Args:
        X_train: Eğitim özellikleri
        y_train: Eğitim hedef değişkeni
        
    Returns:
        dict: Eğitilmiş modeller, parametreler ve skorlar
    """
    print("\n🤖 Model eğitimi başlatıldı...")
    print("=" * 50)
    
    models = {}
    results = {}
    
    # Logistic Regression
    try:
        lr_model, lr_params, lr_cv = train_logistic_regression(X_train, y_train)
        models['logreg'] = lr_model
        results['logreg'] = {
            'model': lr_model,
            'params': lr_params,
            'cv_scores': lr_cv,
            'cv_mean': lr_cv.mean(),
            'cv_std': lr_cv.std()
        }
        print(f"✓ Logistic Regression: CV F1-Score = {lr_cv.mean():.4f}")
    except Exception as e:
        logging.error(f"Logistic Regression eğitimi başarısız: {e}")
        print(f"❌ Logistic Regression eğitimi başarısız!")
    
    # Random Forest
    try:
        rf_model, rf_params, rf_cv = train_random_forest(X_train, y_train)
        models['rf'] = rf_model
        results['rf'] = {
            'model': rf_model,
            'params': rf_params,
            'cv_scores': rf_cv,
            'cv_mean': rf_cv.mean(),
            'cv_std': rf_cv.std()
        }
        print(f"✓ Random Forest: CV F1-Score = {rf_cv.mean():.4f}")
    except Exception as e:
        logging.error(f"Random Forest eğitimi başarısız: {e}")
        print(f"❌ Random Forest eğitimi başarısız!")
    
    # K-Nearest Neighbors
    try:
        knn_model, knn_params, knn_cv = train_knn(X_train, y_train)
        models['knn'] = knn_model
        results['knn'] = {
            'model': knn_model,
            'params': knn_params,
            'cv_scores': knn_cv,
            'cv_mean': knn_cv.mean(),
            'cv_std': knn_cv.std()
        }
        print(f"✓ K-Nearest Neighbors: CV F1-Score = {knn_cv.mean():.4f}")
    except Exception as e:
        logging.error(f"KNN eğitimi başarısız: {e}")
        print(f"❌ KNN eğitimi başarısız!")
    
    print("=" * 50)
    logging.info(f"Toplam {len(models)} model başarıyla eğitildi.")
    
    return models, results

def save_best_model(models, results, save_path="results/best_model.pkl"):
    """
    En iyi performanslı modeli seçer ve kaydeder.
    
    Args:
        models (dict): Eğitilmiş modeller
        results (dict): Model sonuçları
        save_path (str): Kayıt dosya yolu
        
    Returns:
        tuple: (en_iyi_model_adı, en_iyi_model, en_iyi_skor)
    """
    if not models:
        logging.error("Kaydedilecek model bulunamadı!")
        return None, None, None
    
    # En yüksek CV F1-Score'a sahip modeli bul
    best_model_name = max(results.keys(), key=lambda k: results[k]['cv_mean'])
    best_model = models[best_model_name]
    best_score = results[best_model_name]['cv_mean']
    
    # Modeli kaydet
    try:
        joblib.dump(best_model, save_path)
        logging.info(f"En iyi model ({best_model_name}) {save_path} olarak kaydedildi.")
        print(f"🏆 En iyi model: {best_model_name} (F1-Score: {best_score:.4f})")
        print(f"💾 Model kaydedildi: {save_path}")
    except Exception as e:
        logging.error(f"Model kaydetme hatası: {e}")
        print(f"❌ Model kaydetme başarısız!")
    
    return best_model_name, best_model, best_score

if __name__ == "__main__":
    # Test için örnek kullanım
    print("Model eğitim modülü test edildi.")
    print("Bu modül main.py tarafından çağrılmalıdır.")