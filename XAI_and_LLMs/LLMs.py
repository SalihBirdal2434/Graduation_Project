import os
import json
import pandas as pd
import requests


class LLM:
    """Ollama uzerinden yerel LLM ile XAI sonuclarindan dogal dil raporu ureten sinif."""

    OLLAMA_URL = "http://localhost:11434/api/generate"

    AVAILABLE_MODELS = [
        "qwen3:0.6b",
        "alibayram/turkish-gemma-9b-v0.1:latest",
    ]

    def __init__(self, model_name: str = "qwen3:0.6b", output_dir: str = "results"):
        self.model_name = model_name
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def _call_ollama(self, prompt: str) -> str:
        """Ollama API'sine istek gonderir ve yaniti dondurur."""
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 1024,
            },
        }
        try:
            resp = requests.post(self.OLLAMA_URL, json=payload, timeout=120)
            resp.raise_for_status()
            return resp.json().get("response", "")
        except requests.ConnectionError:
            return "[HATA] Ollama sunucusuna baglanilamadi. 'ollama serve' calistirdiginizdan emin olun."
        except Exception as e:
            return f"[HATA] LLM cagri hatasi: {e}"

    def generate_report_from_shap(self, local_shap_df: pd.DataFrame,
                                  credit_ok: bool, customer_id: str = "Musteri") -> str:
        """
        Lokal SHAP siniflandirma sonuclarindan LLM ile Turkce rapor uretir.
        :param local_shap_df: classify_local_shap() ciktisi DataFrame.
        :param credit_ok: Kredi onaylandi mi?
        :param customer_id: Musteri numarasi/adi.
        :return: LLM tarafindan uretilen Turkce rapor metni.
        """
        decision = "ONAYLANDI" if credit_ok else "REDDEDILDI"

        # En etkili ozellikleri sec (top 8)
        top_features = local_shap_df.head(8)
        feature_lines = []
        for _, row in top_features.iterrows():
            feature_lines.append(
                f"- {row['feature_tr']}: SHAP degeri={row['shap_value']:.4f}, "
                f"Etki={row['quartile']}, Yon={row['direction']}"
            )
        features_text = "\n".join(feature_lines)

        prompt = f"""Sen bir banka kredi analisti olarak gorev yapiyorsun.
Asagida bir musterinin kredi basvurusu icin yapay zeka modelinin aciklanabilirlik (XAI) analiz sonuclari verilmistir.

Musteri: {customer_id}
Karar: Kredi basvurusu {decision}

Onemli Ozellikler ve Etki Analizi:
{features_text}

Lutfen bu bilgilere dayanarak:
1. Kredi kararinin nedenlerini musteriye anlasilir bir dilde acikla.
2. Hangi finansal ozelliklerin karar uzerinde en cok etkili oldugunu belirt.
3. "Artiran" yondeki ozellikler dolandiricilik riskini artiran, "Azaltan" yondeki ozellikler riski azaltan ozelliklerdir.
4. Musteriye oneriler sun.

Raporu Turkce olarak yaz. Resmi ve profesyonel bir dil kullan. /no_think"""

        print(f"  LLM rapor uretiliyor ({self.model_name})...")
        report = self._call_ollama(prompt)

        # Raporu kaydet
        report_path = os.path.join(self.output_dir, f"llm_report_{customer_id}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"=== LLM Raporu - {customer_id} ===\n")
            f.write(f"Model: {self.model_name}\n")
            f.write(f"Karar: {decision}\n\n")
            f.write(report)

        print(f"  LLM raporu kaydedildi: {report_path}")
        return report

    def generate_analyst_summary(self, metrics_df: pd.DataFrame,
                                 shap_quartiles_df: pd.DataFrame) -> str:
        """
        Analist icin tum model performansi ve SHAP ozetini LLM ile ozetler.
        :param metrics_df: Tum modellerin metriklerini iceren DataFrame.
        :param shap_quartiles_df: Global SHAP quartile degerleri.
        :return: LLM tarafindan uretilen Turkce analiz ozeti.
        """
        metrics_text = metrics_df.to_string(index=False)
        top_features = shap_quartiles_df.sort_values("mean_abs_shap", ascending=False).head(10)
        features_text = top_features[["feature_tr", "mean_abs_shap"]].to_string(index=True)

        prompt = f"""Sen bir banka veri bilimci/analistisin.
Asagida kredi dolandiricilik tespiti icin egitilen makine ogrenmesi modellerinin performans metrikleri ve
SHAP (SHapley Additive exPlanations) ozellik onem analizi sonuclari verilmistir.

Model Performans Metrikleri:
{metrics_text}

En Etkili 10 Ozellik (Ortalama |SHAP| degerine gore):
{features_text}

Lutfen:
1. Modellerin performansini karsilastirmali olarak degerlendir.
2. En etkili ozelliklerin dolandiricilik tespitindeki rolunu acikla.
3. Genel bir degerlendirme ve oneri sun.

Raporu Turkce olarak, profesyonel bir dilde yaz. /no_think"""

        print(f"  Analist ozet raporu uretiliyor ({self.model_name})...")
        report = self._call_ollama(prompt)

        report_path = os.path.join(self.output_dir, "llm_analyst_summary.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("=== Analist Ozet Raporu ===\n")
            f.write(f"Model: {self.model_name}\n\n")
            f.write(report)

        print(f"  Analist ozet raporu kaydedildi: {report_path}")
        return report
