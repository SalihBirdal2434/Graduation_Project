class ModelEvaluator:
    """Eğitilen modellerin performans metriklerini ve grafiklerini üreten sınıf."""

    def __init__(self, model_name: str, y_true, y_pred, y_prob=None):
        """
        Args:
            model_name (str): Değerlendirilen modelin adı (Örn: 'XGBoost').
            y_true: Gerçek etiketler (Test setindeki y_test).
            y_pred: Modelin tahmin ettiği etiketler.
            y_prob: Sınıflandırma problemleri için tahmin olasılıkları (ROC-AUC için).
        """
        self.model_name = model_name
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob

    def calculate_metrics(self) -> dict:
        """Accuracy, Precision, Recall, F1-Score (veya MSE, RMSE) gibi metrikleri hesaplar."""
        pass

    def plot_confusion_matrix(self) -> None:
        """Karmaşıklık matrisini (Confusion Matrix) çizer."""
        pass

    def plot_roc_curve(self) -> None:
        """ROC eğrisini çizer."""
        pass

    def generate_report(self) -> None:
        """Tüm metrikleri hesaplayıp, grafikleri çizdiren 'ana' raporlama metodu."""
        pass