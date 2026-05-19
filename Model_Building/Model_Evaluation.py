import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report,
)


class ModelEvaluator:
    """Egitilen modellerin performans metriklerini ve grafiklerini ureten sinif."""

    def __init__(self, model_name: str, y_true, y_pred, y_prob=None, output_dir: str = "results"):
        self.model_name = model_name
        self.y_true = y_true
        self.y_pred = y_pred
        self.y_prob = y_prob
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def calculate_metrics(self) -> dict:
        """Accuracy, Precision, Recall, F1-Score ve AUC-ROC hesaplar."""
        metrics = {
            "Model": self.model_name,
            "Accuracy": accuracy_score(self.y_true, self.y_pred),
            "Precision": precision_score(self.y_true, self.y_pred, zero_division=0),
            "Recall": recall_score(self.y_true, self.y_pred, zero_division=0),
            "F1_Score": f1_score(self.y_true, self.y_pred, zero_division=0),
        }
        if self.y_prob is not None:
            metrics["AUC_ROC"] = roc_auc_score(self.y_true, self.y_prob)
        return metrics

    def plot_confusion_matrix(self) -> None:
        """Confusion Matrix cizer ve kaydeder."""
        cm = confusion_matrix(self.y_true, self.y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                    xticklabels=["Normal", "Fraud"], yticklabels=["Normal", "Fraud"])
        ax.set_xlabel("Tahmin")
        ax.set_ylabel("Gercek")
        ax.set_title(f"Confusion Matrix - {self.model_name}")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"confusion_matrix_{self.model_name}.png"))
        plt.close()

    def plot_roc_curve(self) -> None:
        """ROC egrisini cizer ve kaydeder."""
        if self.y_prob is None:
            print(f"  {self.model_name}: y_prob yok, ROC cizilemiyor.")
            return
        fpr, tpr, _ = roc_curve(self.y_true, self.y_prob)
        auc_val = roc_auc_score(self.y_true, self.y_prob)

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.plot(fpr, tpr, label=f"AUC = {auc_val:.4f}", color="steelblue", linewidth=2)
        ax.plot([0, 1], [0, 1], "k--", linewidth=0.8)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title(f"ROC Curve - {self.model_name}")
        ax.legend(loc="lower right")
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, f"roc_curve_{self.model_name}.png"))
        plt.close()

    def generate_report(self) -> dict:
        """Tum metrikleri hesaplar, grafikleri cizer ve sonuclari dondurur."""
        metrics = self.calculate_metrics()
        self.plot_confusion_matrix()
        self.plot_roc_curve()

        report_text = classification_report(
            self.y_true, self.y_pred,
            target_names=["Normal", "Fraud"], zero_division=0
        )
        report_path = os.path.join(self.output_dir, f"report_{self.model_name}.txt")
        with open(report_path, "w", encoding="utf-8") as f:
            f.write(f"=== {self.model_name} Classification Report ===\n\n")
            f.write(report_text)
            f.write(f"\nMetrics: {metrics}\n")

        print(f"  {self.model_name} => Acc: {metrics['Accuracy']:.4f}, "
              f"F1: {metrics['F1_Score']:.4f}, "
              f"AUC: {metrics.get('AUC_ROC', 'N/A')}")
        return metrics
