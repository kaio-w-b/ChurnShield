"""ChurnShield AI — Visualization Module"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns
from loguru import logger
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    RocCurveDisplay,
    PrecisionRecallDisplay,
    roc_auc_score,
)

# Estilo global
sns.set_theme(style="whitegrid", palette="husl")
COLORS = {
    "primary": "#2E4057",
    "accent": "#E63946",
    "safe": "#2A9D8F",
    "warning": "#E9C46A",
    "critical": "#E76F51",
    "neutral": "#A8DADC",
}


def plot_churn_distribution(df: pd.DataFrame, save_path: Optional[str] = None):
    """Plota distribuição de churn com estatísticas."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Distribuição de Churn", fontsize=16, fontweight="bold", y=1.02)

    # Gráfico de pizza
    churn_counts = df["churn"].value_counts()
    axes[0].pie(
        churn_counts,
        labels=["Retido", "Churn"],
        colors=[COLORS["safe"], COLORS["critical"]],
        autopct="%1.1f%%",
        startangle=90,
        explode=(0, 0.05),
    )
    axes[0].set_title("Proporção Geral")

    # Distribuição por contrato
    contract_churn = (
        df.groupby("contract_type")["churn"]
        .mean()
        .sort_values(ascending=False)
        .reset_index()
    )
    bars = axes[1].bar(
        contract_churn["contract_type"],
        contract_churn["churn"] * 100,
        color=[COLORS["critical"], COLORS["warning"], COLORS["safe"]],
        edgecolor="white",
        linewidth=1.5,
    )
    axes[1].set_title("Taxa de Churn por Tipo de Contrato")
    axes[1].set_ylabel("Taxa de Churn (%)")
    axes[1].yaxis.set_major_formatter(mtick.PercentFormatter())

    for bar in bars:
        h = bar.get_height()
        axes[1].text(
            bar.get_x() + bar.get_width() / 2, h + 0.5,
            f"{h:.1f}%", ha="center", va="bottom", fontweight="bold"
        )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot salvo: {save_path}")

    return fig


def plot_roc_curves(
    models_results: Dict[str, Dict],
    y_test: np.ndarray,
    save_path: Optional[str] = None,
):
    """Plota curvas ROC de todos os modelos sobrepostas."""
    fig, ax = plt.subplots(figsize=(10, 7))

    palette = plt.cm.tab10(np.linspace(0, 1, len(models_results)))

    for (name, result), color in zip(models_results.items(), palette):
        auc = result["metrics"]["roc_auc"]
        label = f"{name} (AUC = {auc:.3f})"
        RocCurveDisplay.from_predictions(
            y_test,
            result.get("y_prob", np.zeros_like(y_test)),
            name=label,
            ax=ax,
            color=color,
        )

    # Linha de referência
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Classificador Aleatório")
    ax.set_title("Curvas ROC — Comparação de Modelos", fontsize=14, fontweight="bold")
    ax.set_xlabel("Taxa de Falsos Positivos", fontsize=12)
    ax.set_ylabel("Taxa de Verdadeiros Positivos", fontsize=12)
    ax.legend(loc="lower right")
    ax.fill_between([0, 1], [0, 1], alpha=0.05, color="gray")
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shap_summary(
    shap_importance_df: pd.DataFrame,
    top_n: int = 15,
    save_path: Optional[str] = None,
):
    """Plota importância global de features via SHAP."""
    df_plot = shap_importance_df.head(top_n).sort_values("mean_abs_shap")

    fig, ax = plt.subplots(figsize=(10, max(6, top_n * 0.5)))

    colors = [
        COLORS["critical"] if v > df_plot["mean_abs_shap"].median() else COLORS["neutral"]
        for v in df_plot["mean_abs_shap"]
    ]
    bars = ax.barh(df_plot["feature"], df_plot["mean_abs_shap"], color=colors, edgecolor="white")

    ax.set_xlabel("Mean |SHAP Value|", fontsize=12)
    ax.set_title(f"Top {top_n} Features — Importância Global (SHAP)", fontsize=14, fontweight="bold")
    ax.axvline(df_plot["mean_abs_shap"].median(), color="gray", linestyle="--", alpha=0.7, label="Mediana")
    ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_model_leaderboard(
    results: Dict[str, Dict],
    save_path: Optional[str] = None,
):
    """Plota comparação visual de performance dos modelos."""
    metrics = ["roc_auc", "f1_score", "precision", "recall"]
    metric_labels = ["AUC-ROC", "F1-Score", "Precisão", "Recall"]

    model_names = list(results.keys())
    data = {
        name: [results[name]["metrics"].get(m, 0) for m in metrics]
        for name in model_names
    }

    x = np.arange(len(metrics))
    width = 0.8 / len(model_names)
    palette = plt.cm.Set2(np.linspace(0, 1, len(model_names)))

    fig, ax = plt.subplots(figsize=(13, 6))

    for i, (name, values) in enumerate(data.items()):
        offset = (i - len(model_names) / 2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=name, color=palette[i], alpha=0.85)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Comparação de Métricas — Todos os Modelos", fontsize=14, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(metric_labels, fontsize=11)
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right")
    ax.yaxis.set_major_formatter(mtick.PercentFormatter(xmax=1.0))
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
