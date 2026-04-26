"""
ChurnShield AI — Data Loader
============================
Módulo responsável por ingestão, validação e preparação dos dados brutos.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from loguru import logger
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

CONFIG_PATH = Path(__file__).parent.parent.parent / "config" / "config.yaml"

with open(CONFIG_PATH) as f:
    CONFIG = yaml.safe_load(f)

DATA_CFG = CONFIG["data"]
PATHS = CONFIG["paths"]


# ---------------------------------------------------------------------------
# Gerador de dataset sintético (para uso quando não há arquivo real)
# ---------------------------------------------------------------------------

def generate_synthetic_dataset(n_samples: int = 7_043, random_state: int = 42) -> pd.DataFrame:
    """
    Gera dataset sintético realista de churn de telecomunicações.
    Baseado nas distribuições do Telco Customer Churn (IBM Sample Data).
    
    Args:
        n_samples: Número de registros a gerar.
        random_state: Semente para reprodutibilidade.
    
    Returns:
        DataFrame com dados sintéticos.
    """
    rng = np.random.default_rng(random_state)
    logger.info(f"Gerando dataset sintético com {n_samples:,} amostras...")

    # --- Dados categóricos ---
    gender = rng.choice(["Male", "Female"], size=n_samples)
    senior_citizen = rng.choice([0, 1], size=n_samples, p=[0.84, 0.16])
    partner = rng.choice(["Yes", "No"], size=n_samples, p=[0.48, 0.52])
    dependents = rng.choice(["Yes", "No"], size=n_samples, p=[0.30, 0.70])

    tenure = rng.integers(1, 73, size=n_samples)

    phone_service = rng.choice(["Yes", "No"], size=n_samples, p=[0.91, 0.09])
    multiple_lines = np.where(
        phone_service == "No", "No phone service",
        rng.choice(["Yes", "No"], size=n_samples, p=[0.42, 0.58])
    )

    internet_service = rng.choice(
        ["DSL", "Fiber optic", "No"],
        size=n_samples,
        p=[0.34, 0.44, 0.22]
    )

    def internet_feature(has_internet_mask, p_yes=0.5):
        vals = rng.choice(["Yes", "No"], size=n_samples, p=[p_yes, 1 - p_yes])
        return np.where(has_internet_mask, vals, "No internet service")

    no_internet_mask = internet_service == "No"
    online_security = internet_feature(~no_internet_mask, p_yes=0.29)
    online_backup = internet_feature(~no_internet_mask, p_yes=0.34)
    device_protection = internet_feature(~no_internet_mask, p_yes=0.34)
    tech_support = internet_feature(~no_internet_mask, p_yes=0.29)
    streaming_tv = internet_feature(~no_internet_mask, p_yes=0.38)
    streaming_movies = internet_feature(~no_internet_mask, p_yes=0.39)

    contract = rng.choice(
        ["Month-to-month", "One year", "Two year"],
        size=n_samples,
        p=[0.55, 0.21, 0.24]
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n_samples, p=[0.59, 0.41])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        size=n_samples,
        p=[0.34, 0.23, 0.22, 0.21]
    )

    # --- Charges com correlações realistas ---
    base_charge = 20.0
    internet_charge = np.where(internet_service == "Fiber optic", 35.0,
                       np.where(internet_service == "DSL", 20.0, 0.0))
    phone_charge = np.where(phone_service == "Yes", 20.0, 0.0)
    addon_charge = (
        (online_security != "No internet service") * (online_security == "Yes") * 5 +
        (online_backup != "No internet service") * (online_backup == "Yes") * 5 +
        (device_protection != "No internet service") * (device_protection == "Yes") * 5 +
        (tech_support != "No internet service") * (tech_support == "Yes") * 5 +
        (streaming_tv != "No internet service") * (streaming_tv == "Yes") * 8 +
        (streaming_movies != "No internet service") * (streaming_movies == "Yes") * 8
    )
    monthly_charges = (
        base_charge + internet_charge + phone_charge + addon_charge
        + rng.normal(0, 2, size=n_samples)
    ).clip(18.0, 120.0).round(2)

    total_charges = (monthly_charges * tenure + rng.normal(0, 10, n_samples)).clip(18.0).round(2)

    # --- Churn com lógica de negócio realista ---
    churn_prob = (
        0.05
        + 0.25 * (contract == "Month-to-month")
        - 0.10 * (contract == "Two year")
        + 0.15 * (internet_service == "Fiber optic")
        + 0.12 * (payment_method == "Electronic check")
        - 0.08 * (online_security == "Yes")
        - 0.06 * (tech_support == "Yes")
        + 0.10 * (senior_citizen == 1)
        - 0.15 * (tenure / 72)
        + 0.08 * (monthly_charges / 120)
        - 0.05 * (partner == "Yes")
    ).clip(0.02, 0.95)

    churn = (rng.random(n_samples) < churn_prob).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"CUST-{i:06d}" for i in range(n_samples)],
        "gender": gender,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phone_service": phone_service,
        "multiple_lines": multiple_lines,
        "internet_service": internet_service,
        "online_security": online_security,
        "online_backup": online_backup,
        "device_protection": device_protection,
        "tech_support": tech_support,
        "streaming_tv": streaming_tv,
        "streaming_movies": streaming_movies,
        "contract_type": contract,
        "paperless_billing": paperless_billing,
        "payment_method": payment_method,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "churn": churn,
    })

    churn_rate = df["churn"].mean()
    logger.success(
        f"Dataset gerado: {n_samples:,} linhas | {df.shape[1]} colunas | "
        f"Churn rate: {churn_rate:.1%}"
    )
    return df


# ---------------------------------------------------------------------------
# Validação de dados
# ---------------------------------------------------------------------------

class DataValidator:
    """Valida a qualidade e integridade dos dados de entrada."""

    REQUIRED_COLUMNS = [
        "customer_id", "tenure", "monthly_charges", "total_charges",
        "contract_type", "payment_method", "internet_service", "churn",
    ]
    NUMERIC_COLUMNS = ["tenure", "monthly_charges", "total_charges"]
    VALUE_RANGES = {
        "tenure": (0, 100),
        "monthly_charges": (0, 200),
        "total_charges": (0, 15_000),
    }

    def validate(self, df: pd.DataFrame) -> dict:
        """
        Executa todas as validações e retorna relatório.
        
        Returns:
            Dict com resultados de cada validação.
        """
        results = {}

        # 1. Colunas obrigatórias
        missing_cols = set(self.REQUIRED_COLUMNS) - set(df.columns)
        results["missing_columns"] = list(missing_cols)

        # 2. Valores nulos
        null_report = df.isnull().sum()
        results["null_counts"] = null_report[null_report > 0].to_dict()

        # 3. Duplicatas
        results["duplicate_rows"] = int(df.duplicated(subset="customer_id").sum())

        # 4. Ranges de valores
        range_violations = {}
        for col, (min_val, max_val) in self.VALUE_RANGES.items():
            if col in df.columns:
                out_of_range = ((df[col] < min_val) | (df[col] > max_val)).sum()
                if out_of_range > 0:
                    range_violations[col] = int(out_of_range)
        results["range_violations"] = range_violations

        # 5. Distribuição de target
        if "churn" in df.columns:
            results["class_distribution"] = df["churn"].value_counts(normalize=True).to_dict()

        # Status geral
        has_issues = (
            bool(results["missing_columns"])
            or bool(results["null_counts"])
            or results["duplicate_rows"] > 0
            or bool(results["range_violations"])
        )
        results["is_valid"] = not has_issues

        if results["is_valid"]:
            logger.success("✅ Validação de dados: PASSOU em todas as verificações")
        else:
            logger.warning(f"⚠️  Validação de dados: encontrou problemas → {results}")

        return results


# ---------------------------------------------------------------------------
# Loader principal
# ---------------------------------------------------------------------------

class ChurnDataLoader:
    """
    Classe principal para carregamento e preparação dos dados de churn.
    
    Usage:
        loader = ChurnDataLoader()
        df = loader.load()
        X_train, X_test, y_train, y_test = loader.get_splits(df)
    """

    def __init__(self, config: dict = CONFIG):
        self.config = config
        self.data_cfg = config["data"]
        self.paths = config["paths"]
        self.validator = DataValidator()
        self._data_hash: Optional[str] = None

    def load(self, use_synthetic: bool = False) -> pd.DataFrame:
        """
        Carrega os dados brutos. Usa sintético se arquivo não encontrado.
        
        Args:
            use_synthetic: Força uso de dados sintéticos.
        
        Returns:
            DataFrame com dados brutos.
        """
        raw_path = Path(self.paths["data_raw"]) / "churn_data.csv"

        if use_synthetic or not raw_path.exists():
            logger.info("Arquivo não encontrado. Usando dataset sintético...")
            df = generate_synthetic_dataset()
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(raw_path, index=False)
            logger.info(f"Dataset salvo em: {raw_path}")
        else:
            logger.info(f"Carregando dados de: {raw_path}")
            df = pd.read_csv(raw_path)

        # Hash para rastreabilidade
        self._data_hash = hashlib.md5(
            pd.util.hash_pandas_object(df).values
        ).hexdigest()[:8]
        logger.info(f"Data hash: {self._data_hash}")

        # Validação
        validation_report = self.validator.validate(df)
        if not validation_report["is_valid"]:
            logger.warning("Dados com problemas — aplicando correções automáticas...")
            df = self._auto_fix(df, validation_report)

        return df

    def _auto_fix(self, df: pd.DataFrame, report: dict) -> pd.DataFrame:
        """Aplica correções automáticas nos dados."""
        # Remove duplicatas
        if report["duplicate_rows"] > 0:
            before = len(df)
            df = df.drop_duplicates(subset="customer_id", keep="first")
            logger.info(f"Removidas {before - len(df)} duplicatas")

        # Preenche nulos em numéricos com mediana
        for col in DataValidator.NUMERIC_COLUMNS:
            if col in df.columns and df[col].isnull().any():
                median_val = df[col].median()
                df[col] = df[col].fillna(median_val)
                logger.info(f"Nulos em '{col}' preenchidos com mediana: {median_val}")

        return df

    def get_splits(
        self,
        df: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Divide os dados em treino e teste de forma estratificada.
        
        Returns:
            (X_train, X_test, y_train, y_test)
        """
        target = self.data_cfg["target_column"]
        drop_cols = self.data_cfg.get("drop_columns", [])

        X = df.drop(columns=[target] + [c for c in drop_cols if c in df.columns])
        y = df[target]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=self.data_cfg["test_size"],
            random_state=self.data_cfg["random_state"],
            stratify=y if self.data_cfg.get("stratify") else None,
        )

        logger.info(
            f"Split: train={len(X_train):,} | test={len(X_test):,} | "
            f"churn_train={y_train.mean():.1%} | churn_test={y_test.mean():.1%}"
        )
        return X_train, X_test, y_train, y_test

    @property
    def data_hash(self) -> Optional[str]:
        return self._data_hash


if __name__ == "__main__":
    loader = ChurnDataLoader()
    df = loader.load(use_synthetic=True)
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nChurn rate: {df['churn'].mean():.1%}")
