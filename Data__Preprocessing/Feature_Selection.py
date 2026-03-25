import pandas as pd
import os
from typing import List
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from Data__Preprocessing.Exploratory_Data_Analysis import ExploratoryDataAnalysis


class EnsembleFeatureSelector(ExploratoryDataAnalysis):
    """Veriyi böler, farklı özellik seçimi testleri uygular ve en iyi özellikleri belirler."""

    def __init__(self, df: pd.DataFrame, target_col: str = "Class"):
        """
        :param df: Tam veri seti (hedef sütun dahil).
        :param target_col: Hedef (bağımlı) değişkenin sütun adı.
        """
        super().__init__(df, target_col)
        self._ensure_info()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = [col for col in self.numeric_cols if col != target_col]
        self.cat_cols = [col for col in self.string_cols if col != target_col]

    def split_data(self, test_size=0.3, random_state=42) -> None:
        """
        Veriyi eğitim ve test setlerine böler (stratified).
        :param test_size: Test setinin oranı.
        :param random_state: Tekrarlanabilirlik için seed değeri.
        """
        all_features = self.feature_names + self.cat_cols
        X = self.df[all_features]
        y = self.df[self.target_col]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Sayısal sütunları ölçeklendir: fit_transform train'e, transform test'e
        self.scaler = StandardScaler()
        self.X_train[self.feature_names] = self.scaler.fit_transform(self.X_train[self.feature_names])
        self.X_test[self.feature_names] = self.scaler.transform(self.X_test[self.feature_names])

        # Kategorik sütunları encode et: fit_transform train'e, transform test'e
        self.label_encoders = {}
        for col in self.cat_cols:
            if col in self.X_train.columns:
                le = LabelEncoder()
                self.X_train[col] = le.fit_transform(self.X_train[col].astype(str))
                self.X_test[col] = le.transform(self.X_test[col].astype(str))
                self.label_encoders[col] = le

    def variance_threshold_test(self, threshold=0.01) -> List[str]:
        """
        Varyansı düşük (hemen hemen hep aynı değeri alan) kolonları eler.
        :param threshold: Minimum varyans eşik değeri.
        :return: Eşiği geçen sütun adları listesi.
        """
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(self.X_train)
        mask = selector.get_support()
        return [col for col, selected in zip(self.feature_names, mask) if selected]

    def chi_squared_test(self, k=10) -> List[str]:
        """
        Kategorik özellikler için Ki-Kare testi uygular.
        Negatif değerler varsa min-shift ile sıfırın üzerine taşınır.
        :param k: Seçilecek en iyi özellik sayısı.
        :return: En yüksek Ki-Kare skoruna sahip k sütun adı.
        """
        X_positive = self.X_train - self.X_train.min()
        k = min(k, len(self.feature_names))
        selector = SelectKBest(chi2, k=k)
        selector.fit(X_positive, self.y_train)
        mask = selector.get_support()
        return [col for col, selected in zip(self.feature_names, mask) if selected]

    def mutual_info_test(self, k=10) -> List[str]:
        """
        Sürekli/Kategorik özellikler arasındaki karşılıklı bilgi kazancını ölçer.
        :param k: Seçilecek en iyi özellik sayısı.
        :return: En yüksek mutual information skoruna sahip k sütun adı.
        """
        k = min(k, len(self.feature_names))
        selector = SelectKBest(mutual_info_classif, k=k)
        selector.fit(self.X_train, self.y_train)
        mask = selector.get_support()
        return [col for col, selected in zip(self.feature_names, mask) if selected]

    def extra_classifier_test(self) -> List[str]:
        """
        ExtraTreesClassifier ile özellik önemlerini hesaplar.
        Ortalama önemin üzerinde kalan özellikler seçilir.
        :return: Önem değeri ortalamanın üzerinde olan sütun adları.
        """
        model = ExtraTreesClassifier(n_estimators=100, random_state=42)
        model.fit(self.X_train, self.y_train)
        importances = model.feature_importances_
        mean_importance = importances.mean()
        return [col for col, imp in zip(self.feature_names, importances) if imp >= mean_importance]

    def get_best_features(self) -> List[str]:
        """
        Tüm testleri çalıştırır, en az 2 testte seçilen özellikleri döndürür.
        :return: Çoğunluk oyuyla seçilen sütun adları.
        """
        results = [
            set(self.variance_threshold_test()),
            set(self.chi_squared_test()),
            set(self.mutual_info_test()),
            set(self.extra_classifier_test())
        ]

        vote_count = {}
        for feature_set in results:
            for feature in feature_set:
                vote_count[feature] = vote_count.get(feature, 0) + 1

        min_votes = 2
        best_features = [f for f, count in vote_count.items() if count >= min_votes]

        # Orijinal sütun sırasını koru
        return [col for col in self.feature_names if col in best_features]
