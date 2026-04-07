import pandas as pd
import numpy as np
import os
from typing import List
from sklearn.feature_selection import VarianceThreshold, SelectKBest, chi2, mutual_info_classif
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from Data__Preprocessing.Exploratory_Data_Analysis import ExploratoryDataAnalysis


class EnsembleFeatureSelector(ExploratoryDataAnalysis):
    """Veriyi boler, farkli ozellik secimi testleri uygular ve en iyi ozellikleri belirler."""

    def __init__(self, df: pd.DataFrame, target_col: str = "Class"):
        super().__init__(df, target_col)
        self._ensure_info()
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_names = [col for col in self.numeric_cols if col != target_col]
        self.cat_cols = [col for col in self.string_cols if col != target_col]
        self.all_feature_cols = self.feature_names + self.cat_cols

    def split_data(self, test_size=0.3, random_state=42) -> None:
        """Veriyi egitim ve test setlerine boler (stratified)."""
        X = self.df[self.all_feature_cols].copy()
        y = self.df[self.target_col].copy()

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )

        # Kategorik sutunlari encode et
        self.label_encoders = {}
        for col in self.cat_cols:
            if col in self.X_train.columns:
                le = LabelEncoder()
                self.X_train[col] = le.fit_transform(self.X_train[col].astype(str))
                self.X_test[col] = le.transform(self.X_test[col].astype(str))
                self.label_encoders[col] = le

        # Sayisal sutunlari olceklendir
        self.scaler = StandardScaler()
        self.X_train[self.feature_names] = self.scaler.fit_transform(self.X_train[self.feature_names])
        self.X_test[self.feature_names] = self.scaler.transform(self.X_test[self.feature_names])

    def apply_smote(self, random_state=42) -> None:
        """Egitim setine SMOTE uygulayarak sinif dengesizligini giderir."""
        print(f"  SMOTE oncesi sinif dagilimi: {dict(self.y_train.value_counts())}")
        smote = SMOTE(random_state=random_state)
        X_resampled, y_resampled = smote.fit_resample(self.X_train, self.y_train)
        self.X_train = pd.DataFrame(X_resampled, columns=self.X_train.columns)
        self.y_train = pd.Series(y_resampled, name=self.target_col)
        print(f"  SMOTE sonrasi sinif dagilimi: {dict(self.y_train.value_counts())}")

    def variance_threshold_test(self, threshold=0.01) -> List[str]:
        """Varyansi dusuk kolonlari eler."""
        selector = VarianceThreshold(threshold=threshold)
        selector.fit(self.X_train[self.all_feature_cols])
        mask = selector.get_support()
        return [col for col, selected in zip(self.all_feature_cols, mask) if selected]

    def chi_squared_test(self, k=10) -> List[str]:
        """Ki-Kare testi ile en iyi k ozellik secer."""
        X_positive = self.X_train[self.all_feature_cols] - self.X_train[self.all_feature_cols].min()
        k = min(k, len(self.all_feature_cols))
        selector = SelectKBest(chi2, k=k)
        selector.fit(X_positive, self.y_train)
        mask = selector.get_support()
        return [col for col, selected in zip(self.all_feature_cols, mask) if selected]

    def mutual_info_test(self, k=10) -> List[str]:
        """Karsilikli bilgi kazanci ile en iyi k ozellik secer."""
        k = min(k, len(self.all_feature_cols))
        selector = SelectKBest(mutual_info_classif, k=k)
        selector.fit(self.X_train[self.all_feature_cols], self.y_train)
        mask = selector.get_support()
        return [col for col, selected in zip(self.all_feature_cols, mask) if selected]

    def extra_classifier_test(self) -> List[str]:
        """ExtraTreesClassifier ile ortalama uzerinde onem tasiyan ozellikleri secer."""
        model = ExtraTreesClassifier(n_estimators=100, random_state=42)
        model.fit(self.X_train[self.all_feature_cols], self.y_train)
        importances = model.feature_importances_
        mean_importance = importances.mean()
        return [col for col, imp in zip(self.all_feature_cols, importances) if imp >= mean_importance]

    def get_best_features(self) -> List[str]:
        """Tum testleri calistirir, en az 2 testte secilen ozellikleri dondurur."""
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

        return [col for col in self.all_feature_cols if col in best_features]
