"""
ChurnShield AI — Prediction & Explainability
=============================================
Módulo de inferência com predições interpretáveis via SHAP.
Gera explicações individuais e globais para cada predição.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import shap
from loguru import logger


# ---------------------------------------------------------------------------
# Estruturas de dados
# ---------------------------------------------------------------------------

@dataclass
class ChurnPrediction:
    """
    Resultado estruturado de uma predição de churn.
    
    Attributes:
        customer_id: Identificador do cliente.
        churn_probability: Probabilidade de churn (0-1).
        prediction: Classe predita ('CHURNED' ou 'RETAINED').
        risk_level: Nível de risco ('LOW', 'MEDIUM', 'HIGH', 'CRITICAL').
        top_risk_factors: Lista dos fatores de risco mais impactantes.
        protective_factors: Lista dos fatores que reduzem o risco.
        recommended_actions: Ações recomendadas para retenção.
        shap_values: Valores SHAP para explicação detalhada.
        model_version: Versão do modelo usado.
    """

    customer_id: str
    churn_probability: float
    prediction: str
    risk_level: str
    top_risk_factors: List[str] = field(default_factory=list)
    protective_factors: List[str] = field(default_factory=list)
    recommended_actions: List[str] = field(default_factory=list)
    shap_values: Optional[Dict[str, float]] = None
    model_version: str = "1.0.0"

    def to_dict(self) -> Dict:
        return {
            "customer_id": self.customer_id,
            "churn_probability": round(self.churn_probability, 4),
            "prediction": self.prediction,
            "risk_level": self.risk_level,
            "top_risk_factors": self.top_risk_factors,
            "protective_factors": self.protective_factors,
            "recommended_actions": self.recommended_actions,
            "model_version": self.model_version,
        }


# ---------------------------------------------------------------------------
# Motor de explicações
# ---------------------------------------------------------------------------

class ShapExplainer:
    """
    Gera explicações SHAP para predições individuais e globais.
    
    Suporta:
    - TreeExplainer para modelos baseados em árvore (XGBoost, LightGBM, CatBoost)
    - KernelExplainer como fallback para outros modelos
    - Análise global com summary plots
    - Força plots para explicações individuais
    """

    def __init__(self, model: Any, feature_names: List[str]):
        self.model = model
        self.feature_names = feature_names
        self.explainer: Optional[shap.Explainer] = None
        self._background_data: Optional[np.ndarray] = None

    def fit(self, X_background: np.ndarray) -> "ShapExplainer":
        """
        Inicializa o explainer com dados de background.
        
        Args:
            X_background: Amostra dos dados de treino para KernelExplainer.
        """
        self._background_data = shap.sample(X_background, min(100, len(X_background)))

        try:
            # TreeExplainer é mais eficiente para árvores
            self.explainer = shap.TreeExplainer(self.model)
            logger.info("ShapExplainer: usando TreeExplainer")
        except Exception:
            # Fallback para modelos não-tree
            self.explainer = shap.KernelExplainer(
                self.model.predict_proba,
                self._background_data,
            )
            logger.info("ShapExplainer: usando KernelExplainer (fallback)")

        return self

    def explain_single(
        self,
        X: np.ndarray,
        top_n: int = 5,
    ) -> Dict[str, float]:
        """
        Gera explicação SHAP para uma única amostra.
        
        Returns:
            Dict {feature_name: shap_value} ordenado por magnitude.
        """
        if self.explainer is None:
            raise RuntimeError("Explainer não inicializado. Chame .fit() primeiro.")

        shap_values = self.explainer.shap_values(X)

        # Para classificação binária, pega valores da classe positiva
        if isinstance(shap_values, list):
            sv = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            sv = shap_values[0]

        # Monta dict feature → valor SHAP
        shap_dict = dict(zip(self.feature_names, sv))

        # Ordena por magnitude
        sorted_shap = dict(
            sorted(shap_dict.items(), key=lambda x: abs(x[1]), reverse=True)[:top_n]
        )
        return sorted_shap

    def explain_global(self, X: np.ndarray) -> pd.DataFrame:
        """
        Gera importância global de features via SHAP mean(|value|).
        
        Returns:
            DataFrame com feature e mean_abs_shap, ordenado por importância.
        """
        if self.explainer is None:
            raise RuntimeError("Explainer não inicializado.")

        shap_values = self.explainer.shap_values(X)
        if isinstance(shap_values, list):
            sv = shap_values[1] if len(shap_values) > 1 else shap_values[0]
        else:
            sv = shap_values

        mean_abs = np.abs(sv).mean(axis=0)
        importance_df = pd.DataFrame({
            "feature": self.feature_names,
            "mean_abs_shap": mean_abs,
        }).sort_values("mean_abs_shap", ascending=False)

        return importance_df


# ---------------------------------------------------------------------------
# Sistema de recomendações
# ---------------------------------------------------------------------------

RETENTION_ACTIONS = {
    "contract_risk": {
        "high": [
            "Oferecer desconto de 20% para migração para contrato anual",
            "Apresentar benefícios exclusivos do contrato bianual",
        ],
        "medium": ["Comunicar vantagens de contratos de longo prazo"],
    },
    "internet_risk": {
        "high": [
            "Verificar qualidade da conexão de fibra ótica do cliente",
            "Oferecer upgrade de velocidade sem custo por 3 meses",
            "Agendar visita técnica preventiva",
        ],
    },
    "payment_risk": {
        "high": [
            "Migrar para débito automático com desconto de 5%",
            "Oferecer parcelamento sem juros de faturas em atraso",
        ],
    },
    "online_security": {
        "low": ["Oferecer 3 meses grátis do pacote de segurança online"],
    },
    "tech_support": {
        "low": ["Incluir suporte técnico prioritário sem custo adicional por 6 meses"],
    },
    "tenure_group": {
        "0-12m": [
            "Ativar programa de onboarding personalizado",
            "Designar gerente de conta dedicado",
        ],
    },
    "is_high_value": {
        "high": [
            "Contato proativo do time de sucesso do cliente",
            "Oferecer programa de fidelidade VIP",
            "Desconto personalizado para renovação",
        ],
    },
}


def generate_retention_actions(shap_dict: Dict[str, float]) -> List[str]:
    """
    Gera lista de ações de retenção baseadas nos fatores de risco SHAP.
    
    Args:
        shap_dict: Dict {feature: shap_value} com principais fatores.
    
    Returns:
        Lista de ações recomendadas (máx. 4).
    """
    actions = []
    for feature, shap_val in shap_dict.items():
        if shap_val > 0:  # aumenta risco de churn
            feature_actions = RETENTION_ACTIONS.get(feature, {})
            if "high" in feature_actions:
                actions.extend(feature_actions["high"][:1])
            elif "medium" in feature_actions:
                actions.extend(feature_actions["medium"][:1])
        else:  # fator protetor
            feature_actions = RETENTION_ACTIONS.get(feature, {})
            if "low" in feature_actions:
                actions.extend(feature_actions["low"][:1])

    # Remove duplicatas, mantém ordem
    seen = set()
    unique_actions = []
    for a in actions:
        if a not in seen:
            seen.add(a)
            unique_actions.append(a)

    return unique_actions[:4] if unique_actions else [
        "Contato proativo para pesquisa de satisfação",
        "Oferecer oferta personalizada de retenção",
    ]


# ---------------------------------------------------------------------------
# Predictor principal
# ---------------------------------------------------------------------------

class ChurnPredictor:
    """
    Interface de predição com explicabilidade integrada.
    
    Usage:
        predictor = ChurnPredictor.from_mlflow("runs:/abc123/model")
        result = predictor.predict(customer_data)
    """

    RISK_THRESHOLDS = {
        "LOW": 0.30,
        "MEDIUM": 0.60,
        "HIGH": 0.80,
        "CRITICAL": 1.01,
    }

    def __init__(
        self,
        model: Any,
        feature_pipeline: Any,
        feature_names: List[str],
        model_version: str = "1.0.0",
    ):
        self.model = model
        self.feature_pipeline = feature_pipeline
        self.feature_names = feature_names
        self.model_version = model_version
        self.explainer: Optional[ShapExplainer] = None

    @classmethod
    def from_files(
        cls,
        model_path: str,
        pipeline_path: str,
        feature_names_path: str,
    ) -> "ChurnPredictor":
        """Carrega predictor a partir de arquivos."""
        model = joblib.load(model_path)
        pipeline = joblib.load(pipeline_path)
        feature_names = joblib.load(feature_names_path)
        logger.info(f"Modelos carregados: {model_path}")
        return cls(model, pipeline, feature_names)

    def fit_explainer(self, X_background: np.ndarray) -> "ChurnPredictor":
        """Inicializa o explainer SHAP com dados de background."""
        self.explainer = ShapExplainer(self.model, self.feature_names)
        self.explainer.fit(X_background)
        return self

    def _get_risk_level(self, prob: float) -> str:
        for level, threshold in self.RISK_THRESHOLDS.items():
            if prob < threshold:
                return level
        return "CRITICAL"

    def predict(
        self,
        raw_input: Dict[str, Any],
        customer_id: str = "UNKNOWN",
        explain: bool = True,
    ) -> ChurnPrediction:
        """
        Realiza predição completa com explicação.
        
        Args:
            raw_input: Dict com features do cliente.
            customer_id: ID do cliente.
            explain: Se True, gera explicação SHAP.
        
        Returns:
            ChurnPrediction com todos os detalhes.
        """
        # Prepara input
        X = pd.DataFrame([raw_input])
        X_transformed = self.feature_pipeline.transform(X)

        # Predição
        churn_prob = float(self.model.predict_proba(X_transformed)[0, 1])
        prediction = "CHURNED" if churn_prob >= 0.5 else "RETAINED"
        risk_level = self._get_risk_level(churn_prob)

        # Explicação SHAP
        shap_dict = {}
        top_risk_factors = []
        protective_factors = []
        recommended_actions = []

        if explain and self.explainer is not None:
            shap_dict = self.explainer.explain_single(X_transformed, top_n=8)

            top_risk_factors = [
                f for f, v in shap_dict.items() if v > 0
            ][:4]

            protective_factors = [
                f for f, v in shap_dict.items() if v < 0
            ][:3]

            recommended_actions = generate_retention_actions(shap_dict)

        logger.info(
            f"Predição para {customer_id}: "
            f"p={churn_prob:.3f} | risco={risk_level} | {prediction}"
        )

        return ChurnPrediction(
            customer_id=customer_id,
            churn_probability=churn_prob,
            prediction=prediction,
            risk_level=risk_level,
            top_risk_factors=top_risk_factors,
            protective_factors=protective_factors,
            recommended_actions=recommended_actions,
            shap_values=shap_dict,
            model_version=self.model_version,
        )

    def predict_batch(
        self,
        df: pd.DataFrame,
        id_column: str = "customer_id",
    ) -> pd.DataFrame:
        """
        Predição em lote para DataFrame completo.
        
        Returns:
            DataFrame original com colunas adicionais de predição.
        """
        logger.info(f"Predição em lote: {len(df):,} registros")

        ids = df[id_column].tolist() if id_column in df.columns else [str(i) for i in range(len(df))]
        X = df.drop(columns=[id_column], errors="ignore")
        X_transformed = self.feature_pipeline.transform(X)

        probs = self.model.predict_proba(X_transformed)[:, 1]
        predictions = ["CHURNED" if p >= 0.5 else "RETAINED" for p in probs]
        risk_levels = [self._get_risk_level(p) for p in probs]

        result_df = df.copy()
        result_df["churn_probability"] = probs.round(4)
        result_df["prediction"] = predictions
        result_df["risk_level"] = risk_levels

        # Estatísticas
        churn_rate = (np.array(predictions) == "CHURNED").mean()
        logger.success(
            f"Batch concluído: {churn_rate:.1%} em risco de churn | "
            f"CRITICAL: {(np.array(risk_levels) == 'CRITICAL').sum()} clientes"
        )

        return result_df
