from abc import ABC, abstractmethod
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from Data__Preprocessing.Feature_Selection import EnsembleFeatureSelector


class BaseModel(EnsembleFeatureSelector, ABC):
    """Tüm makine öğrenmesi modelleri için temel soyut sınıf."""

    def __init__(self, df, target_col="Class", **kwargs):
        EnsembleFeatureSelector.__init__(self, df, target_col)
        self.model = None
        self.params = kwargs

    @abstractmethod
    def train(self) -> None:
        """Modeli eğitir."""
        pass

    @abstractmethod
    def predict(self):
        """Tahmin yapar."""
        pass

    @abstractmethod
    def tune_hyperparameters(self):
        """Bu modele özel hiperparametre optimizasyonu yapar."""
        pass


class XGBoostModel(BaseModel):
    """XGBoost algoritmasının implementasyonu."""

    DEFAULT_PARAMS = {
        "n_estimators": 300,
        "max_depth": 6,
        "learning_rate": 0.1,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "scale_pos_weight": 1,
        "enable_categorical": True,
        "random_state": 42,
    }

    def __init__(self, df, target_col="Class", **kwargs):
        super().__init__(df, target_col, **kwargs)
        params = {**self.DEFAULT_PARAMS, **self.params}
        self.model = XGBClassifier(**params)

    def train(self) -> None:
        """X_train ve y_train ile modeli eğitir. cat_cols varsa category tipine çevirir."""
        for col in self.cat_cols:
            if col in self.X_train.columns:
                self.X_train[col] = self.X_train[col].astype("category")
                self.X_test[col] = self.X_test[col].astype("category")
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        """X_test üzerinde tahmin yapar ve y_test ile karşılaştırır."""
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def tune_hyperparameters(self):
        """XGBoost için hiperparametre optimizasyonu. (Henüz implement edilmedi)"""
        pass


class LightGBM(BaseModel):
    """LightGBM algoritmasının implementasyonu."""

    DEFAULT_PARAMS = {
        "n_estimators": 300,
        "max_depth": -1,
        "learning_rate": 0.1,
        "num_leaves": 31,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "random_state": 42,
        "verbose": -1,
    }

    def __init__(self, df, target_col="Class", **kwargs):
        super().__init__(df, target_col, **kwargs)
        params = {**self.DEFAULT_PARAMS, **self.params}
        self.model = LGBMClassifier(**params)

    def train(self) -> None:
        """X_train ve y_train ile modeli eğitir. cat_cols varsa LightGBM'e bildirir."""
        cat_indices = [self.X_train.columns.tolist().index(c) for c in self.cat_cols if c in self.X_train.columns]
        self.model.fit(
            self.X_train, self.y_train,
            categorical_feature=cat_indices if cat_indices else "auto"
        )

    def predict(self):
        """X_test üzerinde tahmin yapar ve y_test ile karşılaştırır."""
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def tune_hyperparameters(self):
        """LightGBM için hiperparametre optimizasyonu. (Henüz implement edilmedi)"""
        pass


class Catboost(BaseModel):
    """CatBoost algoritmasının implementasyonu."""

    DEFAULT_PARAMS = {
        "iterations": 500,
        "depth": 6,
        "learning_rate": 0.1,
        "l2_leaf_reg": 3,
        "random_seed": 42,
        "verbose": 0,
    }

    def __init__(self, df, target_col="Class", **kwargs):
        super().__init__(df, target_col, **kwargs)
        params = {**self.DEFAULT_PARAMS, **self.params}
        self.model = CatBoostClassifier(**params)

    def train(self) -> None:
        """X_train ve y_train ile modeli eğitir. cat_cols varsa CatBoost'a native olarak verir."""
        cat_indices = [self.X_train.columns.tolist().index(c) for c in self.cat_cols if c in self.X_train.columns]
        self.model.fit(
            self.X_train, self.y_train,
            cat_features=cat_indices if cat_indices else None
        )

    def predict(self):
        """X_test üzerinde tahmin yapar ve y_test ile karşılaştırır."""
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def tune_hyperparameters(self):
        """CatBoost için hiperparametre optimizasyonu. (Henüz implement edilmedi)"""
        pass




