"""
ChurnShield AI — Feature Engineering
=====================================
Pipeline avançado de engenharia de features com transformações estatísticas,
criação de variáveis derivadas e seleção automática de features.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from loguru import logger
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectFromModel, mutual_info_classif
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from imblearn.over_sampling import SMOTE


# ---------------------------------------------------------------------------
# Transformadores customizados (seguem API do scikit-learn)
# ---------------------------------------------------------------------------

class NumericalFeatureCreator(BaseEstimator, TransformerMixin):
    """
    Cria features numéricas derivadas com base no conhecimento de domínio.
    
    Features criadas:
    - avg_monthly_charges: média das cobranças mensais ao longo do tenure
    - charge_per_product: custo por produto contratado  
    - tenure_group: agrupamento de tenure em faixas
    - charges_to_tenure_ratio: razão cobranças/tempo de contrato
    - high_value_customer: flag para clientes de alto valor
    """

    def fit(self, X: pd.DataFrame, y=None) -> "NumericalFeatureCreator":
        # Armazena medianas para evitar data leakage em produção
        self.monthly_median_ = X["monthly_charges"].median()
        self.total_median_ = X["total_charges"].median()
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Razão cobranças totais / tenure (proteção contra divisão por zero)
        X["avg_monthly_charges"] = X["total_charges"] / (X["tenure"] + 1)

        # Variação relativa em relação ao cliente médio
        X["charges_vs_avg"] = X["monthly_charges"] / self.monthly_median_

        # Agrupamento de tenure
        X["tenure_group"] = pd.cut(
            X["tenure"],
            bins=[0, 12, 24, 48, 72, np.inf],
            labels=["0-12m", "12-24m", "24-48m", "48-72m", "72m+"],
            right=True,
        ).astype(str)

        # Score de valor do cliente (normalizado)
        X["customer_value_score"] = (
            (X["total_charges"] - X["total_charges"].min())
            / (X["total_charges"].max() - X["total_charges"].min() + 1e-9)
        )

        # Flag cliente de alto valor
        X["is_high_value"] = (
            X["monthly_charges"] > X["monthly_charges"].quantile(0.75)
        ).astype(int)

        # Interação tenure × charges
        X["tenure_x_charges"] = X["tenure"] * X["monthly_charges"]

        # Feature quadrática de tenure (captura não-linearidade)
        X["tenure_squared"] = X["tenure"] ** 2

        logger.debug(f"NumericalFeatureCreator: {X.shape[1]} features após transformação")
        return X


class CategoricalEncoder(BaseEstimator, TransformerMixin):
    """
    Encoder inteligente para variáveis categóricas.
    - Binárias: LabelEncoder (0/1)
    - Alta cardinalidade: One-Hot Encoding com drop='first'
    - Ordinal: mapeamento manual quando aplicável
    """

    BINARY_MAP = {"Yes": 1, "No": 0}
    ORDINAL_RISK = {
        "Month-to-month": 2,   # maior risco de churn
        "One year": 1,
        "Two year": 0,
    }
    INTERNET_SERVICE_MAP = {
        "Fiber optic": 2,       # maior risco
        "DSL": 1,
        "No": 0,
    }
    PAYMENT_METHOD_MAP = {
        "Electronic check": 3,  # maior risco
        "Mailed check": 2,
        "Bank transfer (automatic)": 1,
        "Credit card (automatic)": 0,
    }

    BINARY_COLS = [
        "gender", "partner", "dependents", "phone_service",
        "paperless_billing", "senior_citizen",
    ]
    INTERNET_ADDON_COLS = [
        "online_security", "online_backup", "device_protection",
        "tech_support", "streaming_tv", "streaming_movies",
    ]
    OHE_COLS = ["multiple_lines"]

    def fit(self, X: pd.DataFrame, y=None) -> "CategoricalEncoder":
        self.ohe_ = OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore")
        ohe_present = [c for c in self.OHE_COLS if c in X.columns]
        if ohe_present:
            self.ohe_.fit(X[ohe_present])
            self.ohe_feature_names_ = self.ohe_.get_feature_names_out(ohe_present)
        else:
            self.ohe_feature_names_ = []
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Gênero → binário
        if "gender" in X.columns:
            X["gender"] = (X["gender"] == "Male").astype(int)

        # Binárias simples
        for col in self.BINARY_COLS:
            if col in X.columns and col != "gender" and col != "senior_citizen":
                X[col] = X[col].map(self.BINARY_MAP).fillna(0).astype(int)

        # senior_citizen já é 0/1
        if "senior_citizen" in X.columns:
            X["senior_citizen"] = X["senior_citizen"].astype(int)

        # Add-ons de internet: 3 categorias → ordinal de risco
        internet_addon_map = {
            "Yes": 1,
            "No": 0,
            "No internet service": -1,  # sem internet = sem exposição
        }
        for col in self.INTERNET_ADDON_COLS:
            if col in X.columns:
                X[col] = X[col].map(internet_addon_map).fillna(0).astype(int)

        # Features ordinais de risco (domain knowledge)
        if "contract_type" in X.columns:
            X["contract_risk"] = X["contract_type"].map(self.ORDINAL_RISK).fillna(1)
            X = X.drop(columns=["contract_type"])

        if "internet_service" in X.columns:
            X["internet_risk"] = X["internet_service"].map(self.INTERNET_SERVICE_MAP).fillna(1)
            X = X.drop(columns=["internet_service"])

        if "payment_method" in X.columns:
            X["payment_risk"] = X["payment_method"].map(self.PAYMENT_METHOD_MAP).fillna(1)
            X = X.drop(columns=["payment_method"])

        # OHE para colunas de alta cardinalidade
        ohe_present = [c for c in self.OHE_COLS if c in X.columns]
        if ohe_present and len(self.ohe_feature_names_) > 0:
            ohe_encoded = self.ohe_.transform(X[ohe_present])
            ohe_df = pd.DataFrame(
                ohe_encoded,
                columns=self.ohe_feature_names_,
                index=X.index,
            )
            X = pd.concat([X.drop(columns=ohe_present), ohe_df], axis=1)

        # Conta total de produtos contratados
        addon_cols_present = [c for c in self.INTERNET_ADDON_COLS if c in X.columns]
        if addon_cols_present:
            X["num_addons"] = (X[addon_cols_present] == 1).sum(axis=1)

        logger.debug(f"CategoricalEncoder: {X.shape[1]} features após encoding")
        return X


class InteractionFeatureCreator(BaseEstimator, TransformerMixin):
    """Cria features de interação entre pares de variáveis numéricas importantes."""

    INTERACTIONS = [
        ("contract_risk", "monthly_charges"),
        ("internet_risk", "monthly_charges"),
        ("payment_risk", "tenure"),
        ("contract_risk", "tenure"),
    ]

    def fit(self, X, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col_a, col_b in self.INTERACTIONS:
            if col_a in X.columns and col_b in X.columns:
                feat_name = f"{col_a}_x_{col_b}"
                X[feat_name] = X[col_a] * X[col_b]
        return X


class OutlierHandler(BaseEstimator, TransformerMixin):
    """Trata outliers usando método IQR com winsorização."""

    def __init__(self, columns: List[str], threshold: float = 1.5):
        self.columns = columns
        self.threshold = threshold

    def fit(self, X: pd.DataFrame, y=None) -> "OutlierHandler":
        self.bounds_ = {}
        for col in self.columns:
            if col in X.columns:
                Q1 = X[col].quantile(0.25)
                Q3 = X[col].quantile(0.75)
                IQR = Q3 - Q1
                self.bounds_[col] = (
                    Q1 - self.threshold * IQR,
                    Q3 + self.threshold * IQR,
                )
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        for col, (lower, upper) in self.bounds_.items():
            clipped = X[col].clip(lower, upper)
            n_clipped = (X[col] != clipped).sum()
            if n_clipped > 0:
                logger.debug(f"Outliers: {n_clipped} valores clipados em '{col}'")
            X[col] = clipped
        return X


# ---------------------------------------------------------------------------
# Pipeline completo de engenharia de features
# ---------------------------------------------------------------------------

def build_feature_pipeline(config: dict) -> Pipeline:
    """
    Constrói o pipeline sklearn completo de transformação de features.
    
    Ordem das etapas:
    1. Criação de features numéricas derivadas
    2. Encoding de variáveis categóricas
    3. Criação de features de interação
    4. Tratamento de outliers
    5. Normalização (StandardScaler)
    
    Args:
        config: Configuração do projeto.
    
    Returns:
        sklearn Pipeline pronto para fit/transform.
    """
    feat_cfg = config.get("features", {})
    numeric_cols = config["data"].get("numerical_features", [])

    steps = [
        ("num_creator", NumericalFeatureCreator()),
        ("cat_encoder", CategoricalEncoder()),
        ("interactions", InteractionFeatureCreator()),
        ("outlier_handler", OutlierHandler(columns=numeric_cols)),
        ("scaler", StandardScaler()),
    ]

    pipeline = Pipeline(steps=steps)
    logger.info(f"Pipeline de features construído com {len(steps)} etapas")
    return pipeline


def apply_smote(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Aplica SMOTE para balancear as classes no conjunto de treino.
    
    Args:
        X_train: Features de treino.
        y_train: Labels de treino.
        k_neighbors: Número de vizinhos para SMOTE.
        random_state: Semente.
    
    Returns:
        (X_resampled, y_resampled) como arrays numpy.
    """
    before = y_train.value_counts().to_dict()
    smote = SMOTE(k_neighbors=k_neighbors, random_state=random_state)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    after = pd.Series(y_res).value_counts().to_dict()

    logger.info(f"SMOTE: antes={before} → depois={after}")
    return X_res, y_res


def get_feature_importance_ranking(
    X: pd.DataFrame,
    y: pd.Series,
    n_top: int = 20,
) -> pd.DataFrame:
    """
    Calcula importância de features via Mutual Information.
    Útil para análise exploratória e seleção de features.
    
    Returns:
        DataFrame ordenado por importância.
    """
    mi_scores = mutual_info_classif(X, y, random_state=42)
    importance_df = pd.DataFrame({
        "feature": X.columns,
        "mutual_info_score": mi_scores,
    }).sort_values("mutual_info_score", ascending=False).head(n_top)

    logger.info(f"Top 5 features por Mutual Information:\n{importance_df.head()}")
    return importance_df
