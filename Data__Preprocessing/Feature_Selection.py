import pandas as pd
from typing import List


class EnsembleFeatureSelector:
    """Farklı özellik seçimi testlerini uygulayan ve en iyi özellikleri belirleyen sınıf."""

    def __init__(self, X_train: pd.DataFrame, y_train: pd.Series):
        self.X_train = X_train
        self.y_train = y_train
        self.feature_names = X_train.columns.tolist()

    def variance_threshold_test(self, threshold=0.01) -> List[str]:
        """Varyansı düşük (hemen hemen hep aynı değeri alan) kolonları eler."""
        pass

    def chi_squared_test(self, k=10) -> List[str]:
        """Kategorik özellikler için Ki-Kare testi uygular."""
        pass

    def mutual_info_test(self, k=10) -> List[str]:
        """Sürekli/Kategorik özellikler arasındaki bilgi kazancını ölçer."""
        pass

    def extra_classifier_test(self) -> List[str]:
        """Ağaç tabanlı bir model kurarak özellik önemlerini (feature importance) çıkarır."""
        pass

    def get_best_features(self) -> List[str]:
        """
        Bütün testleri çalıştırır ve ortak kararla en güçlü özellikleri seçer.
        """
        pass

