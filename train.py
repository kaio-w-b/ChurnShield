"""
ChurnShield AI — Model Training
================================
Pipeline de treinamento com múltiplos modelos, otimização bayesiana de
hiperparâmetros (Optuna), ensemble e tracking completo via MLflow.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import optuna
import pandas as pd
import yaml
from catboost import CatBoostClassifier
from lightgbm import LGBMClassifier
from loguru import logger
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from xgboost import XGBClassifier

# Silencia logs verbosos de bibliotecas
optuna.logging.set_verbosity(optuna.logging.WARNING)


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"
with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)


# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Calcula conjunto completo de métricas de classificação.
    
    Returns:
        Dict com todas as métricas relevantes para o problema de churn.
    """
    y_pred_thresh = (y_prob >= threshold).astype(int)

    return {
        "roc_auc": roc_auc_score(y_true, y_prob),
        "average_precision": average_precision_score(y_true, y_prob),
        "f1_score": f1_score(y_true, y_pred_thresh),
        "precision": precision_score(y_true, y_pred_thresh, zero_division=0),
        "recall": recall_score(y_true, y_pred_thresh, zero_division=0),
        "threshold": threshold,
    }


def find_optimal_threshold(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    metric: str = "f1",
) -> float:
    """
    Encontra o threshold ótimo para maximizar a métrica escolhida.
    
    Para problemas de churn, recall pode ser mais importante que precisão
    (melhor identificar clientes em risco, mesmo com falsos positivos).
    """
    thresholds = np.arange(0.1, 0.9, 0.01)
    best_threshold = 0.5
    best_score = 0.0

    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        if metric == "f1":
            score = f1_score(y_true, y_pred, zero_division=0)
        elif metric == "recall":
            score = recall_score(y_true, y_pred, zero_division=0)
        else:
            score = f1_score(y_true, y_pred, zero_division=0)

        if score > best_score:
            best_score = score
            best_threshold = t

    logger.info(f"Threshold ótimo ({metric}): {best_threshold:.2f} → score: {best_score:.4f}")
    return best_threshold


# ---------------------------------------------------------------------------
# Otimização bayesiana de hiperparâmetros
# ---------------------------------------------------------------------------

class HyperparameterOptimizer:
    """Otimização bayesiana via Optuna para cada modelo."""

    def __init__(self, X_train, y_train, cv_folds: int = 5, n_trials: int = 50):
        self.X_train = X_train
        self.y_train = y_train
        self.cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        self.n_trials = n_trials

    def optimize_xgboost(self) -> Dict[str, Any]:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "max_depth": trial.suggest_int("max_depth", 3, 8),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "subsample": trial.suggest_float("subsample", 0.6, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": 42,
                "eval_metric": "auc",
                "verbosity": 0,
            }
            model = XGBClassifier(**params)
            scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=self.cv, scoring="roc_auc", n_jobs=-1,
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        logger.info(f"XGBoost best AUC (CV): {study.best_value:.4f}")
        return study.best_params

    def optimize_lightgbm(self) -> Dict[str, Any]:
        def objective(trial):
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 200, 800),
                "num_leaves": trial.suggest_int("num_leaves", 20, 120),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
                "bagging_freq": trial.suggest_int("bagging_freq", 1, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
                "random_state": 42,
                "verbose": -1,
            }
            model = LGBMClassifier(**params)
            scores = cross_val_score(
                model, self.X_train, self.y_train,
                cv=self.cv, scoring="roc_auc", n_jobs=-1,
            )
            return scores.mean()

        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=self.n_trials, show_progress_bar=False)
        logger.info(f"LightGBM best AUC (CV): {study.best_value:.4f}")
        return study.best_params


# ---------------------------------------------------------------------------
# Treinador principal
# ---------------------------------------------------------------------------

class ChurnModelTrainer:
    """
    Orquestra o treinamento de todos os modelos com tracking MLflow.
    
    Usage:
        trainer = ChurnModelTrainer()
        results = trainer.train_all(X_train, X_test, y_train, y_test)
        best_model = trainer.get_best_model()
    """

    def __init__(self, config: dict = CONFIG):
        self.config = config
        self.models_cfg = config["models"]
        self.eval_cfg = config["evaluation"]
        self.mlflow_cfg = config["mlflow"]
        self.trained_models: Dict[str, Any] = {}
        self.results: Dict[str, Dict] = {}

        # Configura MLflow
        mlflow.set_tracking_uri(self.mlflow_cfg["tracking_uri"])
        mlflow.set_experiment(self.mlflow_cfg["experiment_name"])

    def _get_base_models(self) -> Dict[str, Any]:
        """Instancia todos os modelos base com configurações padrão."""
        cfg = self.models_cfg
        return {
            "baseline_lr": LogisticRegression(**cfg["baseline"]["params"]),
            "xgboost": XGBClassifier(**cfg["xgboost"]["params"]),
            "lightgbm": LGBMClassifier(**cfg["lightgbm"]["params"]),
            "catboost": CatBoostClassifier(**cfg["catboost"]["params"]),
        }

    def train_single_model(
        self,
        name: str,
        model: Any,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        extra_params: Optional[Dict] = None,
    ) -> Dict[str, Any]:
        """
        Treina um único modelo com logging completo no MLflow.
        
        Returns:
            Dict com métricas, modelo treinado e run_id do MLflow.
        """
        logger.info(f"Treinando modelo: {name}")
        start_time = time.time()

        with mlflow.start_run(run_name=name) as run:
            # Log de parâmetros
            mlflow.log_params({"model_name": name, **(extra_params or {})})

            # Treinamento
            model.fit(X_train, y_train)
            train_time = time.time() - start_time

            # Predições
            y_prob = model.predict_proba(X_test)[:, 1]
            y_pred = model.predict(X_test)

            # Threshold ótimo
            opt_threshold = find_optimal_threshold(y_test, y_prob, metric="f1")

            # Métricas
            metrics = compute_metrics(y_test, y_pred, y_prob, threshold=opt_threshold)
            metrics["train_time_seconds"] = train_time

            # CV score
            cv_scores = cross_val_score(
                model, X_train, y_train,
                cv=StratifiedKFold(5, shuffle=True, random_state=42),
                scoring="roc_auc",
                n_jobs=-1,
            )
            metrics["cv_auc_mean"] = cv_scores.mean()
            metrics["cv_auc_std"] = cv_scores.std()

            # Log MLflow
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
                model,
                artifact_path="model",
                registered_model_name=self.mlflow_cfg["model_registry_name"]
                if name == "ensemble" else None,
            )

            run_id = run.info.run_id

        logger.success(
            f"✅ {name}: AUC={metrics['roc_auc']:.4f} | "
            f"F1={metrics['f1_score']:.4f} | "
            f"CV={metrics['cv_auc_mean']:.4f}±{metrics['cv_auc_std']:.4f} | "
            f"Tempo={train_time:.1f}s"
        )

        return {
            "model": model,
            "metrics": metrics,
            "run_id": run_id,
            "optimal_threshold": opt_threshold,
        }

    def train_all(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        y_train: np.ndarray,
        y_test: np.ndarray,
        optimize_hyperparams: bool = False,
    ) -> Dict[str, Dict]:
        """
        Treina todos os modelos e o ensemble final.
        
        Args:
            X_train, X_test, y_train, y_test: Dados de treino/teste.
            optimize_hyperparams: Se True, executa Optuna antes do treino.
        
        Returns:
            Dict com resultados de cada modelo.
        """
        logger.info("=" * 60)
        logger.info("🚀 Iniciando pipeline de treinamento — ChurnShield AI")
        logger.info("=" * 60)

        base_models = self._get_base_models()

        # Otimização bayesiana (opcional)
        if optimize_hyperparams:
            logger.info("🔬 Executando otimização bayesiana de hiperparâmetros...")
            optimizer = HyperparameterOptimizer(
                X_train, y_train,
                n_trials=self.config["hyperopt"]["n_trials"],
            )
            xgb_params = optimizer.optimize_xgboost()
            lgbm_params = optimizer.optimize_lightgbm()
            base_models["xgboost"] = XGBClassifier(**xgb_params)
            base_models["lightgbm"] = LGBMClassifier(**lgbm_params)

        # Treina modelos individuais
        for name, model in base_models.items():
            result = self.train_single_model(
                name, model, X_train, X_test, y_train, y_test
            )
            self.trained_models[name] = result["model"]
            self.results[name] = result

        # Ensemble (Voting Classifier com calibração)
        logger.info("🎯 Treinando Ensemble (Voting Classifier)...")
        ensemble_cfg = self.models_cfg["ensemble"]
        ensemble = VotingClassifier(
            estimators=[
                ("xgboost", self.trained_models["xgboost"]),
                ("lightgbm", self.trained_models["lightgbm"]),
                ("catboost", self.trained_models["catboost"]),
            ],
            voting=ensemble_cfg["voting"],
            weights=ensemble_cfg["weights"],
        )
        # Calibração de probabilidades (Platt scaling)
        calibrated_ensemble = CalibratedClassifierCV(ensemble, method="sigmoid", cv=3)
        result = self.train_single_model(
            "ensemble", calibrated_ensemble, X_train, X_test, y_train, y_test
        )
        self.trained_models["ensemble"] = calibrated_ensemble
        self.results["ensemble"] = result

        self._print_leaderboard()
        return self.results

    def get_best_model(self) -> Tuple[str, Any, float]:
        """
        Retorna o modelo com maior AUC-ROC no conjunto de teste.
        
        Returns:
            (model_name, model_object, auc_score)
        """
        best_name = max(
            self.results,
            key=lambda k: self.results[k]["metrics"]["roc_auc"],
        )
        best_model = self.trained_models[best_name]
        best_auc = self.results[best_name]["metrics"]["roc_auc"]
        logger.info(f"🏆 Melhor modelo: {best_name} (AUC={best_auc:.4f})")
        return best_name, best_model, best_auc

    def _print_leaderboard(self) -> None:
        """Imprime leaderboard formatado dos modelos."""
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(title="🏆 ChurnShield — Leaderboard de Modelos", show_lines=True)
        table.add_column("Modelo", style="cyan", justify="left")
        table.add_column("AUC-ROC", style="green", justify="center")
        table.add_column("F1-Score", style="yellow", justify="center")
        table.add_column("Precisão", justify="center")
        table.add_column("Recall", justify="center")
        table.add_column("CV AUC", justify="center")

        sorted_results = sorted(
            self.results.items(),
            key=lambda x: x[1]["metrics"]["roc_auc"],
            reverse=True,
        )

        for name, result in sorted_results:
            m = result["metrics"]
            prefix = "🥇 " if name == sorted_results[0][0] else ""
            table.add_row(
                f"{prefix}{name}",
                f"{m['roc_auc']:.4f}",
                f"{m['f1_score']:.4f}",
                f"{m['precision']:.4f}",
                f"{m['recall']:.4f}",
                f"{m['cv_auc_mean']:.4f}±{m['cv_auc_std']:.4f}",
            )

        console.print(table)


if __name__ == "__main__":
    # Exemplo de uso standalone
    from src.data.data_loader import ChurnDataLoader
    from src.features.feature_engineering import build_feature_pipeline, apply_smote

    loader = ChurnDataLoader()
    df = loader.load(use_synthetic=True)
    X_train, X_test, y_train, y_test = loader.get_splits(df)

    # Feature engineering
    pipeline = build_feature_pipeline(CONFIG)
    X_train_fe = pipeline.fit_transform(X_train, y_train)
    X_test_fe = pipeline.transform(X_test)

    # Balancear classes
    X_train_res, y_train_res = apply_smote(X_train_fe, y_train)

    # Treinar
    trainer = ChurnModelTrainer()
    results = trainer.train_all(X_train_res, X_test_fe, y_train_res, y_test.values)
