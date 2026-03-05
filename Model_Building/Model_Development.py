from abc import ABC, abstractmethod
from xgboost import XGBClassifier
from catboost import CatboostClassifier
from lightgbm import LightGBMClassifier

class BaseModel(ABC):
    """Tüm makine öğrenmesi modelleri için temel soyut sınıf."""

    def __init__(self, **kwargs):
        self.model = None
        self.params = kwargs

    @abstractmethod
    def train(self, X_train, y_train) -> None:
        """Modeli eğitir."""
        pass

    @abstractmethod
    def predict(self, X_test):
        """Tahmin yapar."""
        pass

    @abstractmethod
    def tune_hyperparameters(self, X_train, y_train):
        """Bu modele özel hiperparametre optimizasyonu yapar."""
        pass


class XGBoostModel(BaseModel):
    """XGBoost algoritmasının implementasyonu."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Model nesnesini burada oluşturuyoruz
        self.model = XGBClassifier(**self.params)

    def train(self, X_train, y_train) -> None:
        print("XGBoost modeli eğitiliyor...")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def tune_hyperparameters(self, X_train, y_train):
        print("XGBoost için hiperparametreler aranıyor...")
        # Optuna veya GridSearchCV kodları buraya gelecek
        pass

class LightGBM(BaseModel):
    """LightGBM algoritmasının implementasyonu."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Model nesnesini burada oluşturuyoruz
        self.model = XGBClassifier(**self.params)

    def train(self, X_train, y_train) -> None:
        print("LightGBM modeli eğitiliyor...")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def tune_hyperparameters(self, X_train, y_train):
        print("LightGBM için hiperparametreler aranıyor...")
        # Optuna veya GridSearchCV kodları buraya gelecek
        pass

class Catboost(BaseModel):
    """Catboost algoritmasının implementasyonu."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Model nesnesini burada oluşturuyoruz
        self.model = XGBClassifier(**self.params)

    def train(self, X_train, y_train) -> None:
        print("Catboost modeli eğitiliyor...")
        self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)

    def tune_hyperparameters(self, X_train, y_train):
        print("Catboost için hiperparametreler aranıyor...")
        # Optuna veya GridSearchCV kodları buraya gelecek
        pass




