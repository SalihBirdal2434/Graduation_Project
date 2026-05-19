import os
import numpy as np
import pandas as pd
import shap
import lime
import lime.lime_tabular
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


class XAI:
    """SHAP ve LIME ile model aciklanabilirligi saglayan sinif."""

    # Oz nitelik isimlerinin Turkce karsiliklari
    FEATURE_TR = {
        "Time": "Islem Zamani",
        "Amount": "Islem Tutari",
        "V1": "PCA Bileseni 1", "V2": "PCA Bileseni 2", "V3": "PCA Bileseni 3",
        "V4": "PCA Bileseni 4", "V5": "PCA Bileseni 5", "V6": "PCA Bileseni 6",
        "V7": "PCA Bileseni 7", "V8": "PCA Bileseni 8", "V9": "PCA Bileseni 9",
        "V10": "PCA Bileseni 10", "V11": "PCA Bileseni 11", "V12": "PCA Bileseni 12",
        "V13": "PCA Bileseni 13", "V14": "PCA Bileseni 14", "V15": "PCA Bileseni 15",
        "V16": "PCA Bileseni 16", "V17": "PCA Bileseni 17", "V18": "PCA Bileseni 18",
        "V19": "PCA Bileseni 19", "V20": "PCA Bileseni 20", "V21": "PCA Bileseni 21",
        "V22": "PCA Bileseni 22", "V23": "PCA Bileseni 23", "V24": "PCA Bileseni 24",
        "V25": "PCA Bileseni 25", "V26": "PCA Bileseni 26", "V27": "PCA Bileseni 27",
        "V28": "PCA Bileseni 28",
        "amt": "Islem Tutari", "category": "Islem Kategorisi",
        "gender": "Cinsiyet", "city_pop": "Sehir Nufusu",
        "lat": "Enlem", "long": "Boylam",
        "merch_lat": "Magaza Enlem", "merch_long": "Magaza Boylam",
        "unix_time": "Unix Zamani", "zip": "Posta Kodu",
        "MCC": "Isyeri Kategori Kodu", "Zip": "Posta Kodu",
        "Year": "Yil", "Month": "Ay", "Day": "Gun",
        "User": "Kullanici", "Card": "Kart",
    }

    def __init__(self, model, X_train: pd.DataFrame, X_test: pd.DataFrame,
                 feature_names: list, output_dir: str = "results"):
        self.model = model
        self.X_train = X_train
        self.X_test = X_test
        self.feature_names = feature_names
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

        self.global_shap_values = None
        self.shap_quartiles = None

    def _get_tr_name(self, col: str) -> str:
        return self.FEATURE_TR.get(col, col)

    # ------------------------------------------------------------------ #
    #  SHAP
    # ------------------------------------------------------------------ #
    def compute_shap(self, max_samples: int = 500) -> pd.DataFrame:
        """
        Global SHAP degerlerini hesaplar ve kaydeder.
        :param max_samples: Aciklama icin kullanilacak maksimum ornek sayisi.
        :return: SHAP degerlerini iceren DataFrame.
        """
        print("  SHAP degerleri hesaplaniyor...")
        try:
            explainer = shap.TreeExplainer(self.model)
        except Exception:
            background = shap.sample(self.X_train, min(100, len(self.X_train)))
            explainer = shap.Explainer(self.model, background)

        X_sample = self.X_test.iloc[:max_samples]
        shap_values = explainer.shap_values(X_sample)

        # Binary classification: shap_values listesi ise sinif 1'i al
        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        shap_df = pd.DataFrame(shap_values, columns=self.feature_names)
        self.global_shap_values = shap_df

        # Global SHAP summary plot
        plt.figure()
        shap.summary_plot(shap_values, X_sample, feature_names=self.feature_names, show=False)
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, "shap_summary_global.png"), dpi=150, bbox_inches="tight")
        plt.close()

        # SHAP degerlerini CSV'ye kaydet
        shap_df.to_csv(os.path.join(self.output_dir, "shap_values.csv"), index=False)
        print(f"  SHAP degerleri kaydedildi: {os.path.join(self.output_dir, 'shap_values.csv')}")
        return shap_df

    def compute_shap_quartiles(self) -> pd.DataFrame:
        """
        Global SHAP degerlerinin ceyrekliklerini (Q1, Q2/median, Q3) hesaplar.
        Her oz nitelik icin SHAP quartile sinirlarini belirler.
        """
        if self.global_shap_values is None:
            raise ValueError("Once compute_shap() calistirilmalidir.")

        quartiles = {}
        for col in self.feature_names:
            vals = self.global_shap_values[col].abs()
            quartiles[col] = {
                "feature_tr": self._get_tr_name(col),
                "Q1": vals.quantile(0.25),
                "Q2_median": vals.quantile(0.50),
                "Q3": vals.quantile(0.75),
                "mean_abs_shap": vals.mean(),
            }

        self.shap_quartiles = pd.DataFrame(quartiles).T
        self.shap_quartiles.index.name = "feature"
        self.shap_quartiles.to_csv(os.path.join(self.output_dir, "shap_quartiles.csv"))
        print(f"  SHAP quartile degerleri kaydedildi.")
        return self.shap_quartiles

    def classify_local_shap(self, sample_index: int = 0) -> pd.DataFrame:
        """
        Tek bir musteri icin lokal SHAP degerlerini quartile'lara gore siniflandirir.
        :param sample_index: X_test icerisindeki ornek indeksi.
        :return: Siniflandirilmis lokal SHAP DataFrame'i.
        """
        if self.global_shap_values is None or self.shap_quartiles is None:
            raise ValueError("Once compute_shap() ve compute_shap_quartiles() calistirilmalidir.")

        local_shap = self.global_shap_values.iloc[sample_index]
        rows = []
        for col in self.feature_names:
            abs_val = abs(local_shap[col])
            q1 = self.shap_quartiles.loc[col, "Q1"]
            q2 = self.shap_quartiles.loc[col, "Q2_median"]
            q3 = self.shap_quartiles.loc[col, "Q3"]

            if abs_val <= q1:
                quartile = "Krediye Çok Düşük Etki"
            elif abs_val <= q2:
                quartile = "Krediye Düşük Etki"
            elif abs_val <= q3:
                quartile = "Krediye Yüksek Etki"
            else:
                quartile = "Krediye Çok Yüksek Etki"


            rows.append({
                "feature_tr": self._get_tr_name(col),
                "quartile": quartile,
            })

        result = pd.DataFrame(rows).sort_values("abs_shap", ascending=False)
        result.to_csv(os.path.join(self.output_dir, f"local_shap_sample_{sample_index}.csv"), index=False)

        # Lokal SHAP bar plot
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ["#e74c3c" if v > 0 else "#3498db" for v in result["shap_value"]]
        ax.barh(result["feature_tr"], result["shap_value"], color=colors)
        ax.set_xlabel("SHAP Degeri")
        ax.set_title(f"Lokal SHAP - Ornek {sample_index}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"shap_musteri_{sample_index}.png"), dpi=150)
        plt.close()

        return result

    # ------------------------------------------------------------------ #
    #  LIME
    # ------------------------------------------------------------------ #
    def compute_lime(self, sample_index: int = 0, num_features: int = 10) -> pd.DataFrame:
        """
        Tek bir musteri icin LIME aciklamasi uretir.
        :param sample_index: X_test icerisindeki ornek indeksi.
        :param num_features: Gosterilecek ozellik sayisi.
        :return: LIME aciklamalarini iceren DataFrame.
        """
        print(f"  LIME aciklamasi hesaplaniyor (ornek {sample_index})...")
        explainer = lime.lime_tabular.LimeTabularExplainer(
            training_data=self.X_train.values,
            feature_names=self.feature_names,
            class_names=["Normal", "Fraud"],
            mode="classification",
        )

        instance = self.X_test.iloc[sample_index].values
        explanation = explainer.explain_instance(
            instance,
            self.model.predict_proba,
            num_features=num_features,
        )

        # LIME sonuclarini kaydet
        lime_list = explanation.as_list()
        lime_df = pd.DataFrame(lime_list, columns=["feature_rule", "weight"])
        lime_df.to_csv(os.path.join(self.output_dir, f"lime_sample_{sample_index}.csv"), index=False)

        # LIME gorsel
        fig = explanation.as_pyplot_figure()
        fig.tight_layout()
        fig.savefig(os.path.join(self.output_dir, f"lime_musteri_{sample_index}.png"), dpi=150)
        plt.close(fig)

        print(f"  LIME aciklamasi kaydedildi.")
        return lime_df
