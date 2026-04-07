import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
from Data__Preprocessing.Data_Cleaning import DataCleaning


class ExploratoryDataAnalysis(DataCleaning):
    """Sayısal sütunlar üzerinde keşifsel veri analizi yapar ve grafikleri dosyaya kaydeder."""

    def __init__(self, df: pd.DataFrame, target_col: str = "Class", output_dir: str = "eda_outputs"):
        """
        :param df: Analiz edilecek pandas DataFrame.
        :param target_col: Hedef (bağımlı) değişkenin sütun adı.
        :param output_dir: Grafiklerin kaydedileceği klasör yolu.
        """
        super().__init__(df, target_col)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _ensure_info(self):
        """info() henüz çağrılmadıysa çalıştırarak numeric_cols attribute'unu oluşturur."""
        if not hasattr(self, "numeric_cols") or self.numeric_cols is None:
            with open(os.devnull, "w") as devnull:
                self.info(devnull)

    def histograms(self) -> None:
        """
        Her sayısal sütun için histogram çizer.
        Tüm histogramlar tek bir PNG dosyasında grid olarak kaydedilir.
        """
        self._ensure_info()
        n = len(self.numeric_cols)
        cols_per_row = 4
        rows = (n + cols_per_row - 1) // cols_per_row

        fig, axes = plt.subplots(rows, cols_per_row, figsize=(cols_per_row * 4, rows * 3))
        axes = axes.flatten()

        for i, col in enumerate(self.numeric_cols):
            axes[i].hist(self.df[col].dropna(), bins=30, edgecolor="black")
            axes[i].set_title(col)
            axes[i].set_xlabel("Değer")
            axes[i].set_ylabel("Frekans")

        # Kullanılmayan subplot'ları kaldır
        for j in range(i + 1, len(axes)):
            fig.delaxes(axes[j])

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "histograms.png"))
        plt.close()

    def control_chart(self) -> None:
        """
        Her sayısal sütun için kontrol grafiği oluşturur.
        Ortalama, UCL (mean + 3σ) ve LCL (mean - 3σ) çizgileri çizilir.
        Her sütun ayrı bir PNG dosyasına kaydedilir.
        """
        self._ensure_info()

        for col in self.numeric_cols:
            series = self.df[col].dropna().reset_index(drop=True)
            mean = series.mean()
            std = series.std()
            ucl = mean + 3 * std
            lcl = mean - 3 * std

            fig, ax = plt.subplots(figsize=(12, 4))
            ax.plot(series.values, linewidth=0.8, color="steelblue", label="Değer")
            ax.axhline(mean, color="green", linestyle="--", label=f"Ortalama ({mean:.2f})")
            ax.axhline(ucl, color="red", linestyle="--", label=f"UCL ({ucl:.2f})")
            ax.axhline(lcl, color="orange", linestyle="--", label=f"LCL ({lcl:.2f})")
            ax.set_title(f"Kontrol Grafiği: {col}")
            ax.set_xlabel("Gözlem")
            ax.set_ylabel("Değer")
            ax.legend(fontsize=8)

            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, f"control_chart_{col}.png"))
            plt.close()

    def correlation(self) -> None:
        """
        Sayısal sütunlar arasındaki Pearson korelasyon matrisini hesaplar
        ve seaborn ısı haritası (heatmap) olarak kaydeder.
        """
        self._ensure_info()
        corr_matrix = self.df[self.numeric_cols].corr()

        size = len(self.numeric_cols) * 0.8 + 2
        fig, ax = plt.subplots(figsize=(size, size * 0.75))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                    linewidths=0.5, ax=ax, annot_kws={"size": 7})
        ax.set_title("Korelasyon Matrisi")

        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "correlation.png"))
        plt.close()

    def drop_high_missing(self, threshold: float = 0.10) -> pd.DataFrame:
        """
        Eksik deger orani threshold'un uzerinde olan sutunlari veri setinden cikarir.
        :param threshold: Maksimum eksik deger orani (varsayilan %10).
        :return: Sutunlari elenmis DataFrame.
        """
        missing_pct = self.df.isnull().mean()
        cols_to_drop = missing_pct[missing_pct > threshold].index.tolist()
        if cols_to_drop:
            print(f"  Eksik orani >{threshold*100:.0f}% olan sutunlar silindi: {cols_to_drop}")
            self.df = self.df.drop(columns=cols_to_drop)
        return self.df

    def handle_outliers(self) -> pd.DataFrame:
        """
        IQR yontemiyle aykiri degerleri tespit eder ve sinir degerlerine clip eder.
        :return: Aykiri degerleri islenmis DataFrame.
        """
        self._ensure_info()
        for col in self.numeric_cols:
            if col == self.target_col:
                continue
            q1 = self.df[col].quantile(0.25)
            q3 = self.df[col].quantile(0.75)
            iqr = q3 - q1
            lower = q1 - 1.5 * iqr
            upper = q3 + 1.5 * iqr
            self.df[col] = self.df[col].clip(lower=lower, upper=upper)
        return self.df
