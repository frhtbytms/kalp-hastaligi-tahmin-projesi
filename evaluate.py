"""
Model Değerlendirme ve Görselleştirme Modülü
Kalp Hastalığı Tahmini Projesi

Bu modül, eğitilmiş modellerin performansını değerlendirir ve görselleştirir.

Özellikler:
- Performans metrikleri hesaplama (Accuracy, Precision, Recall, F1, AUC)
- Confusion Matrix oluşturma ve görselleştirme
- ROC Curve çizimi
- Korelasyon analizi
- Veri dağılımı grafikleri

Çıktılar:
- metrics_table.csv: Model karşılaştırma tablosu
- *.png: Görselleştirme dosyaları

Geliştirici: Ferhat Bayutmuş
Tarih: 17 Kasım 2025
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc
)
import logging
import os

# Matplotlib ayarları
plt.style.use('seaborn-v0_8')
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

# Logging ayarları
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluate.log'),
        logging.StreamHandler()
    ]
)

def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    """
    Tek bir modelin performansını değerlendirir ve görselleştirmeleri oluşturur.
    
    Args:
        model: Eğitilmiş model
        X_test: Test özellikleri
        y_test: Test hedef değişkeni
        model_name (str): Model adı
        
    Returns:
        dict: Model performans metrikleri
    """
    logging.info(f"{model_name} modeli değerlendiriliyor...")
    
    # Tahminleri al
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Metrikleri hesapla
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    
    # AUC hesapla (eğer predict_proba mevcutsa)
    roc_auc = None
    if y_pred_proba is not None:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
        roc_auc = auc(fpr, tpr)
    
    # Classification report
    class_report = classification_report(y_test, y_pred)
    logging.info(f"{model_name} Classification Report:\n{class_report}")
    
    # Confusion Matrix görselleştirmesi
    plot_confusion_matrix(y_test, y_pred, model_name)
    
    # ROC Curve görselleştirmesi
    if y_pred_proba is not None:
        plot_roc_curve(y_test, y_pred_proba, model_name)
    
    # Sonuçları döndür
    results = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc': roc_auc if roc_auc else 0.0
    }
    
    logging.info(f"{model_name} Sonuçları: {results}")
    return results

def plot_confusion_matrix(y_true, y_pred, model_name: str, save_dir: str = "results"):
    """
    Confusion Matrix görselleştirmesi oluşturur.
    
    Args:
        y_true: Gerçek değerler
        y_pred: Tahmin değerleri  
        model_name (str): Model adı
        save_dir (str): Kayıt klasörü
    """
    # Confusion matrix hesapla
    cm = confusion_matrix(y_true, y_pred)
    
    # Görselleştir
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Sağlıklı', 'Hasta'],
                yticklabels=['Sağlıklı', 'Hasta'])
    plt.title(f'{model_name} - Confusion Matrix')
    plt.ylabel('Gerçek Değer')
    plt.xlabel('Tahmin Değeri')
    
    # Kaydet
    save_path = os.path.join(save_dir, f'confusion_{model_name.lower().replace(" ", "_")}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Confusion matrix kaydedildi: {save_path}")

def plot_roc_curve(y_true, y_proba, model_name: str, save_dir: str = "results"):
    """
    ROC eğrisi görselleştirmesi oluşturur.
    
    Args:
        y_true: Gerçek değerler
        y_proba: Tahmin olasılıkları
        model_name (str): Model adı
        save_dir (str): Kayıt klasörü
    """
    # ROC eğrisi hesapla
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Görselleştir
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} - ROC Curve')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    
    # Kaydet
    save_path = os.path.join(save_dir, f'roc_{model_name.lower().replace(" ", "_")}.png')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"ROC curve kaydedildi: {save_path}")

def evaluate_all_models(models: dict, X_test, y_test) -> pd.DataFrame:
    """
    Tüm modellerin performansını değerlendirir ve karşılaştırır.
    
    Args:
        models (dict): Eğitilmiş modeller sözlüğü
        X_test: Test özellikleri
        y_test: Test hedef değişkeni
        
    Returns:
        pd.DataFrame: Tüm modellerin performans metrikleri
    """
    print("\n📊 Model değerlendirmesi başlatıldı...")
    print("=" * 50)
    
    all_results = {}
    
    # Model ismi eşleştirmeleri
    model_names = {
        'logreg': 'Logistic Regression',
        'rf': 'Random Forest', 
        'knn': 'K-Nearest Neighbors'
    }
    
    for model_key, model in models.items():
        model_name = model_names.get(model_key, model_key)
        try:
            results = evaluate_model(model, X_test, y_test, model_name)
            all_results[model_name] = results
            print(f"✓ {model_name}: F1-Score = {results['f1']:.4f}, AUC = {results['auc']:.4f}")
        except Exception as e:
            logging.error(f"{model_name} değerlendirmesi başarısız: {e}")
            print(f"❌ {model_name} değerlendirmesi başarısız!")
    
    # DataFrame oluştur
    metrics_df = pd.DataFrame.from_dict(all_results, orient='index')
    metrics_df = metrics_df.round(4)
    
    # Sonuçları kaydet
    save_path = "results/metrics_table.csv"
    try:
        metrics_df.to_csv(save_path)
        logging.info(f"Metrik tablosu kaydedildi: {save_path}")
        print(f"💾 Metrik tablosu kaydedildi: {save_path}")
    except Exception as e:
        logging.error(f"Metrik tablosu kaydetme hatası: {e}")
    
    print("=" * 50)
    print("\n📈 PERFORMANS ÖZETİ:")
    print(metrics_df.to_string())
    
    return metrics_df

def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str = "results"):
    """
    Korelasyon ısı haritası oluşturur.
    
    Args:
        df (pd.DataFrame): Veri seti
        save_dir (str): Kayıt klasörü
    """
    # Sadece sayısal sütunları al
    numeric_df = df.select_dtypes(include=[np.number])
    
    # Korelasyon matrisi hesapla
    corr_matrix = numeric_df.corr()
    
    # Görselleştir
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
                square=True, fmt='.2f', cbar_kws={"shrink": .8})
    plt.title('Özellikler Arası Korelasyon Haritası')
    plt.tight_layout()
    
    # Kaydet
    save_path = os.path.join(save_dir, 'corr_heatmap.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Korelasyon haritası kaydedildi: {save_path}")
    print(f"✓ Korelasyon haritası oluşturuldu: {save_path}")

def plot_class_distribution(df: pd.DataFrame, target_col: str = 'target', save_dir: str = "results"):
    """
    Hedef değişken dağılım grafiği oluşturur.
    
    Args:
        df (pd.DataFrame): Veri seti
        target_col (str): Hedef değişken sütun adı
        save_dir (str): Kayıt klasörü
    """
    plt.figure(figsize=(8, 6))
    
    # Sınıf sayılarını hesapla
    class_counts = df[target_col].value_counts()
    
    # Bar plot
    bars = plt.bar(['Sağlıklı (0)', 'Hasta (1)'], class_counts.values, 
                   color=['lightblue', 'lightcoral'], edgecolor='black', linewidth=1.2)
    
    # Değerleri bar üzerine yaz
    for bar, count in zip(bars, class_counts.values):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(count), ha='center', va='bottom', fontweight='bold')
    
    plt.title('Hedef Değişken Dağılımı (Kalp Hastalığı)')
    plt.ylabel('Hasta Sayısı')
    plt.xlabel('Sınıf')
    plt.grid(axis='y', alpha=0.3)
    
    # Yüzdelik hesapla ve ekle
    total = class_counts.sum()
    percentages = (class_counts / total * 100).round(1)
    plt.text(0.5, max(class_counts.values) * 0.8, 
             f'Sağlıklı: %{percentages[0]}\nHasta: %{percentages[1]}',
             ha='center', va='center', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    
    # Kaydet
    save_path = os.path.join(save_dir, 'class_distribution.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Sınıf dağılımı grafiği kaydedildi: {save_path}")
    print(f"✓ Sınıf dağılımı grafiği oluşturuldu: {save_path}")

def plot_boxplots(df: pd.DataFrame, target_col: str = 'target', save_dir: str = "results"):
    """
    Sayısal değişkenler için boxplot oluşturur.
    
    Args:
        df (pd.DataFrame): Veri seti
        target_col (str): Hedef değişken sütun adı
        save_dir (str): Kayıt klasörü
    """
    # Sayısal sütunları al (hedef değişken hariç)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Alt grafik sayısını hesapla
    n_cols = len(numeric_cols)
    n_rows = (n_cols + 2) // 3  # 3 sütunlu düzen
    
    fig, axes = plt.subplots(n_rows, 3, figsize=(15, 5 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # Her sayısal sütun için boxplot
    for i, col in enumerate(numeric_cols):
        row = i // 3
        col_idx = i % 3
        
        if n_rows > 1:
            ax = axes[row, col_idx]
        else:
            ax = axes[col_idx]
        
        # Boxplot oluştur
        df.boxplot(column=col, by=target_col, ax=ax)
        ax.set_title(f'{col} - Sınıflara Göre Dağılım')
        ax.set_xlabel('Kalp Hastalığı')
        ax.set_ylabel(col)
        ax.grid(True, alpha=0.3)
    
    # Boş subplot'ları gizle
    for i in range(n_cols, n_rows * 3):
        row = i // 3
        col_idx = i % 3
        if n_rows > 1:
            axes[row, col_idx].set_visible(False)
        else:
            if col_idx < len(axes):
                axes[col_idx].set_visible(False)
    
    plt.suptitle('Sayısal Değişkenlerin Sınıflara Göre Box Plot Analizi')
    plt.tight_layout()
    
    # Kaydet
    save_path = os.path.join(save_dir, 'boxplots.png')
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Box plot grafiği kaydedildi: {save_path}")
    print(f"✓ Box plot grafiği oluşturuldu: {save_path}")

def create_all_visualizations(df: pd.DataFrame):
    """
    Tüm görselleştirmeleri oluşturur.
    
    Args:
        df (pd.DataFrame): Ham veri seti
    """
    print("\n🎨 Görselleştirmeler oluşturuluyor...")
    print("=" * 50)
    
    try:
        plot_correlation_heatmap(df)
        plot_class_distribution(df)
        plot_boxplots(df)
        print("=" * 50)
        print("✓ Tüm görselleştirmeler başarıyla oluşturuldu!")
    except Exception as e:
        logging.error(f"Görselleştirme hatası: {e}")
        print(f"❌ Görselleştirme hatası: {e}")

if __name__ == "__main__":
    # Test için örnek kullanım
    print("Model değerlendirme modülü test edildi.")
    print("Bu modül main.py tarafından çağrılmalıdır.")