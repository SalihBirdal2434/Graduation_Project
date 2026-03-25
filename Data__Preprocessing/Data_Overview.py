import pandas as pd


class DataOverview:
    """Veri setinin genel özetini çıkarır ve .txt dosyasına kaydeder."""

    def __init__(self, df: pd.DataFrame, target_col: str = "Class", output_file: str = "data_overview.txt"):
        """
        :param df: Analiz edilecek pandas DataFrame.
        :param target_col: Hedef (bağımlı) değişkenin sütun adı.
        :param output_file: Sonuçların yazılacağı .txt dosyasının yolu.
        """
        self.df = df
        self.target_col = target_col
        self.output_file = output_file

    def _write(self, f, text: str = ""):
        """Verilen metni dosyaya yazar ve satır sonu ekler."""
        f.write(text + "\n")

    def info(self, f) -> None:
        """
        Sütun tiplerini tespit eder.
        string_cols ve numeric_cols alt sınıflar tarafından kullanılacağı için self'te saklanır.
        """
        self.string_cols = self.df.select_dtypes(include=["object"]).columns.tolist()
        int_cols = self.df.select_dtypes(include=["int64", "int32"]).columns.tolist()
        float_cols = self.df.select_dtypes(include=["float64", "float32"]).columns.tolist()
        self.numeric_cols = int_cols + float_cols

        self._write(f, "=" * 50)
        self._write(f, "SÜTUN TİPLERİ")
        self._write(f, "=" * 50)
        self._write(f, f"String (object) [{len(self.string_cols)} adet]: {self.string_cols or 'Yok'}")
        self._write(f, f"Integer         [{len(int_cols)} adet]: {int_cols or 'Yok'}")
        self._write(f, f"Float           [{len(float_cols)} adet]: {float_cols or 'Yok'}")

    def describe(self, f) -> None:
        """Sayısal sütunların istatistiksel özetini dosyaya yazar."""
        self._write(f, "\n" + "=" * 50)
        self._write(f, "İSTATİSTİKSEL ÖZET")
        self._write(f, "=" * 50)
        self._write(f, self.df.describe().to_string())

    def columns_object_value_counts(self, f) -> None:
        """Her string (object) sütunun benzersiz değer dağılımını dosyaya yazar."""
        self._write(f, "\n" + "=" * 50)
        self._write(f, "OBJECT SÜTUN DEĞER DAĞILIMI")
        self._write(f, "=" * 50)
        if not self.string_cols:
            self._write(f, "Object tipinde sütun yok.")
            return
        for col in self.string_cols:
            self._write(f, f"\n[{col}]:")
            self._write(f, self.df[col].value_counts().to_string())

    def missing_values(self, f) -> None:
        """Eksik değerlerin sayısını ve yüzdesini hesaplayıp dosyaya yazar."""
        missing_count = self.df.isnull().sum()
        missing_pct = (missing_count / len(self.df)) * 100
        missing_df = pd.DataFrame({"Eksik Sayı": missing_count, "Yüzde (%)": missing_pct.round(2)})
        missing_df = missing_df[missing_df["Eksik Sayı"] > 0].sort_values("Eksik Sayı", ascending=False)

        self._write(f, "\n" + "=" * 50)
        self._write(f, "EKSİK DEĞERLER")
        self._write(f, "=" * 50)
        self._write(f, missing_df.to_string() if not missing_df.empty else "Eksik değer yok.")

    def outliers_for_int_value(self, f) -> None:
        """IQR yöntemiyle sayısal sütunlardaki aykırı değerleri tespit eder."""
        results = []
        for col in self.numeric_cols:
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            outlier_count = ((self.df[col] < q1 - 1.5 * iqr) | (self.df[col] > q3 + 1.5 * iqr)).sum()
            results.append({"Sütun": col, "Q1": round(q1, 4), "Q3": round(q3, 4),
                            "IQR": round(iqr, 4), "Aykırı Sayısı": outlier_count})

        self._write(f, "\n" + "=" * 50)
        self._write(f, "AYKIRI DEĞER ANALİZİ (IQR)")
        self._write(f, "=" * 50)
        self._write(f, pd.DataFrame(results).sort_values("Aykırı Sayısı", ascending=False).to_string(index=False))

    def target_distribution(self, f) -> None:
        """Hedef sütunun sınıf dağılımını (adet ve yüzde) dosyaya yazar."""
        self._write(f, "\n" + "=" * 50)
        self._write(f, f"HEDEF SÜTUN DAĞILIMI: '{self.target_col}'")
        self._write(f, "=" * 50)
        if self.target_col not in self.df.columns:
            self._write(f, f"'{self.target_col}' sütunu bulunamadı.")
            return
        dist = self.df[self.target_col].value_counts()
        pct = self.df[self.target_col].value_counts(normalize=True) * 100
        for label in dist.index:
            self._write(f, f"  {label}: {dist[label]} adet  ({pct[label]:.2f}%)")

    def run(self) -> None:
        """Tüm analiz adımlarını sırayla çalıştırır ve output_file'a kaydeder."""
        with open(self.output_file, "w", encoding="utf-8") as f:
            self._write(f, f"BOYUT: {self.df.shape[0]} satır, {self.df.shape[1]} sütun\n")
            self.info(f)
            self.describe(f)
            self.columns_object_value_counts(f)
            self.missing_values(f)
            self.outliers_for_int_value(f)
            self.target_distribution(f)