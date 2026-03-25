import pandas as pd
from Data__Preprocessing.Data_Overview import DataOverview


class DataCleaning(DataOverview):
    """Eksik değerleri farklı istatistiksel yöntemlerle dolduran sınıf."""

    def __init__(self, df: pd.DataFrame, target_col: str = "Class"):
        """
        :param df: Temizlenecek pandas DataFrame.
        :param target_col: Hedef (bağımlı) değişkenin sütun adı.
        """
        super().__init__(df, target_col)

    def median(self, cols: list = None) -> pd.DataFrame:
        """
        Eksik değerleri ilgili sütunun medyan değeriyle doldurur.
        :param cols: Doldurulacak sütun listesi. None ise tüm sayısal sütunlar.
        :return: Eksik değerleri doldurulmuş DataFrame.
        """
        numeric_cols = self.df.select_dtypes(include=["int64", "int32", "float64", "float32"]).columns.tolist()
        target_cols = cols if cols else numeric_cols

        self.df[target_cols] = self.df[target_cols].fillna(self.df[target_cols].median())
        return self.df

    def mode(self, cols: list = None) -> pd.DataFrame:
        """
        Eksik değerleri ilgili sütunun modu (en sık tekrar eden değer) ile doldurur.
        Hem sayısal hem kategorik sütunlarda kullanılabilir.
        :param cols: Doldurulacak sütun listesi. None ise tüm sütunlar.
        :return: Eksik değerleri doldurulmuş DataFrame.
        """
        target_cols = cols if cols else self.df.columns.tolist()

        for col in target_cols:
            if col in self.df.columns:
                col_mode = self.df[col].mode()
                if not col_mode.empty:
                    self.df[col] = self.df[col].fillna(col_mode[0])

        return self.df

    def knn(self, cols: list = None, k: int = 5) -> pd.DataFrame:
        """
        K-En Yakın Komşu (KNN) algoritmasıyla eksik değerleri doldurur.
        Her eksik değer, öklid uzaklığına göre en yakın k komşunun ortalamasıyla hesaplanır.
        :param cols: Doldurulacak sütun listesi. None ise tüm sayısal sütunlar.
        :param k: Komşu sayısı.
        :return: Eksik değerleri doldurulmuş DataFrame.
        """
        from sklearn.impute import KNNImputer

        numeric_cols = self.df.select_dtypes(include=["int64", "int32", "float64", "float32"]).columns.tolist()
        target_cols = cols if cols else numeric_cols
        target_cols = [c for c in target_cols if c in self.df.columns]

        imputer = KNNImputer(n_neighbors=k)
        self.df[target_cols] = imputer.fit_transform(self.df[target_cols])
        return self.df
