from abc import ABC, abstractmethod
import numpy as np
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from Data__Preprocessing.Feature_Selection import EnsembleFeatureSelector


class BaseModel(EnsembleFeatureSelector, ABC):
    """Tum makine ogrenmesi modelleri icin temel soyut sinif."""

    def __init__(self, df, target_col="Class", **kwargs):
        EnsembleFeatureSelector.__init__(self, df, target_col)
        self.model = None
        self.params = kwargs
        self.best_params = None

    def _calc_scale_pos_weight(self) -> float:
        """Imbalanced veri icin negatif/pozitif sinif oranini hesaplar."""
        if self.y_train is None:
            return 1.0
        counts = self.y_train.value_counts()
        if len(counts) < 2:
            return 1.0
        return counts[0] / counts[1]

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def predict(self):
        pass

    @abstractmethod
    def tune_hyperparameters(self):
        pass


class XGBoostModel(BaseModel):
    """XGBoost algoritmasi."""

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
        for col in self.cat_cols:
            if col in self.X_train.columns:
                self.X_train[col] = self.X_train[col].astype("category")
                self.X_test[col] = self.X_test[col].astype("category")
        self.model.fit(self.X_train, self.y_train)

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def tune_hyperparameters(self, n_iter=30, cv=3, scoring="f1"):
        """XGBoost icin RandomizedSearchCV ile hiperparametre optimizasyonu."""
        spw = self._calc_scale_pos_weight()
        print(f"    scale_pos_weight = {spw:.2f}")

        param_distributions = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [4, 5, 6, 7, 8],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_weight": [1, 3, 5, 7],
            "gamma": [0, 0.1, 0.2, 0.3],
            "scale_pos_weight": [spw],
        }

        base_model = XGBClassifier(
            enable_categorical=True,
            random_state=42,
            eval_metric="logloss",
        )

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )

        for col in self.cat_cols:
            if col in self.X_train.columns:
                self.X_train[col] = self.X_train[col].astype("category")
                self.X_test[col] = self.X_test[col].astype("category")

        search.fit(self.X_train, self.y_train)

        self.best_params = search.best_params_
        self.model = search.best_estimator_
        print(f"    Best F1 (CV): {search.best_score_:.4f}")
        print(f"    Best params: {self.best_params}")
        return self.best_params


class LightGBM(BaseModel):
    """LightGBM algoritmasi."""

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
        cat_indices = [self.X_train.columns.tolist().index(c) for c in self.cat_cols if c in self.X_train.columns]
        self.model.fit(
            self.X_train, self.y_train,
            categorical_feature=cat_indices if cat_indices else "auto"
        )

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def tune_hyperparameters(self, n_iter=30, cv=3, scoring="f1"):
        """LightGBM icin RandomizedSearchCV ile hiperparametre optimizasyonu."""
        spw = self._calc_scale_pos_weight()
        print(f"    scale_pos_weight = {spw:.2f}")

        param_distributions = {
            "n_estimators": [100, 200, 300, 500],
            "max_depth": [3, 5, 7, -1],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "num_leaves": [15, 31, 50, 80],
            "subsample": [0.7, 0.8, 0.9, 1.0],
            "colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
            "min_child_samples": [5, 10, 20, 30],
            "reg_alpha": [0, 0.01, 0.1],
            "reg_lambda": [0, 0.01, 0.1, 1.0],
            "is_unbalance": [True],
        }

        base_model = LGBMClassifier(
            random_state=42,
            verbose=-1,
        )

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )

        search.fit(self.X_train, self.y_train)

        self.best_params = search.best_params_
        self.model = search.best_estimator_
        print(f"    Best F1 (CV): {search.best_score_:.4f}")
        print(f"    Best params: {self.best_params}")
        return self.best_params


class Catboost(BaseModel):
    """CatBoost algoritmasi."""

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
        cat_indices = [self.X_train.columns.tolist().index(c) for c in self.cat_cols if c in self.X_train.columns]
        self.model.fit(
            self.X_train, self.y_train,
            cat_features=cat_indices if cat_indices else None
        )

    def predict(self):
        self.y_pred = self.model.predict(self.X_test)
        return self.y_pred

    def tune_hyperparameters(self, n_iter=20, cv=3, scoring="f1"):
        """CatBoost icin RandomizedSearchCV ile hiperparametre optimizasyonu."""
        print(f"    auto_class_weights = Balanced")

        param_distributions = {
            "iterations": [200, 300, 500, 700],
            "depth": [4, 5, 6, 7, 8],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "l2_leaf_reg": [1, 3, 5, 7, 9],
            "border_count": [32, 64, 128],
            "auto_class_weights": ["Balanced"],
        }

        base_model = CatBoostClassifier(
            random_seed=42,
            verbose=0,
        )

        search = RandomizedSearchCV(
            estimator=base_model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            random_state=42,
            n_jobs=-1,
            verbose=0,
        )

        search.fit(self.X_train, self.y_train)

        self.best_params = search.best_params_
        self.model = search.best_estimator_
        print(f"    Best F1 (CV): {search.best_score_:.4f}")
        print(f"    Best params: {self.best_params}")
        return self.best_params
