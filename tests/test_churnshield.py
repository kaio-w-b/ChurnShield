"""
ChurnShield AI — Test Suite
============================
Testes unitários e de integração cobrindo:
- Geração e validação de dados
- Pipeline de feature engineering  
- Predições da API
- Consistência do modelo
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="module")
def sample_customer_payload():
    """Payload padrão para testes de predição."""
    return {
        "customer_id": "TEST-001",
        "tenure": 6,
        "monthly_charges": 95.50,
        "total_charges": 573.0,
        "contract_type": "Month-to-month",
        "payment_method": "Electronic check",
        "internet_service": "Fiber optic",
        "gender": "Male",
        "senior_citizen": 0,
        "partner": "No",
        "dependents": "No",
        "phone_service": "Yes",
        "multiple_lines": "No",
        "online_security": "No",
        "online_backup": "No",
        "device_protection": "No",
        "tech_support": "No",
        "streaming_tv": "Yes",
        "streaming_movies": "No",
        "paperless_billing": "Yes",
    }


@pytest.fixture(scope="module")
def low_risk_customer():
    """Cliente com perfil de baixo risco de churn."""
    return {
        "customer_id": "TEST-LOW-001",
        "tenure": 60,
        "monthly_charges": 45.0,
        "total_charges": 2700.0,
        "contract_type": "Two year",
        "payment_method": "Bank transfer (automatic)",
        "internet_service": "DSL",
        "gender": "Female",
        "senior_citizen": 0,
        "partner": "Yes",
        "dependents": "Yes",
        "phone_service": "Yes",
        "multiple_lines": "No",
        "online_security": "Yes",
        "online_backup": "Yes",
        "device_protection": "Yes",
        "tech_support": "Yes",
        "streaming_tv": "No",
        "streaming_movies": "No",
        "paperless_billing": "No",
    }


@pytest.fixture(scope="module")
def synthetic_dataset():
    """Dataset sintético pequeno para testes de ML."""
    from api.main import predict_single  # noqa
    np.random.seed(42)
    n = 200
    return pd.DataFrame({
        "tenure": np.random.randint(1, 73, n),
        "monthly_charges": np.random.uniform(18, 120, n).round(2),
        "total_charges": np.random.uniform(18, 8000, n).round(2),
        "churn": np.random.binomial(1, 0.27, n),
    })


@pytest.fixture(scope="module")
def api_client():
    """Cliente de testes para a FastAPI."""
    from api.main import app
    return TestClient(app)


# ---------------------------------------------------------------------------
# Testes de Dados
# ---------------------------------------------------------------------------

class TestDataLoader:
    """Testes do módulo de carregamento de dados."""

    def test_synthetic_dataset_shape(self):
        """Dataset sintético deve ter shape mínimo correto."""
        from src.data.data_loader import generate_synthetic_dataset
        df = generate_synthetic_dataset(n_samples=500)
        assert df.shape[0] == 500
        assert df.shape[1] >= 20  # pelo menos 20 colunas

    def test_synthetic_dataset_columns(self):
        """Dataset deve conter colunas obrigatórias."""
        from src.data.data_loader import generate_synthetic_dataset
        df = generate_synthetic_dataset(n_samples=100)
        required = {"customer_id", "tenure", "monthly_charges", "total_charges", "churn"}
        assert required.issubset(set(df.columns))

    def test_churn_rate_realistic(self):
        """Taxa de churn deve estar em faixa realista (15%-35%)."""
        from src.data.data_loader import generate_synthetic_dataset
        df = generate_synthetic_dataset(n_samples=5000)
        churn_rate = df["churn"].mean()
        assert 0.15 <= churn_rate <= 0.40, f"Churn rate fora do esperado: {churn_rate:.1%}"

    def test_no_negative_charges(self):
        """Cobranças não devem ser negativas."""
        from src.data.data_loader import generate_synthetic_dataset
        df = generate_synthetic_dataset(n_samples=500)
        assert (df["monthly_charges"] >= 0).all()
        assert (df["total_charges"] >= 0).all()

    def test_no_duplicate_customer_ids(self):
        """IDs de clientes devem ser únicos."""
        from src.data.data_loader import generate_synthetic_dataset
        df = generate_synthetic_dataset(n_samples=500)
        assert df["customer_id"].nunique() == len(df)

    def test_data_validator_passes_clean_data(self):
        """Validador deve aprovar dados limpos."""
        from src.data.data_loader import DataValidator, generate_synthetic_dataset
        df = generate_synthetic_dataset(n_samples=200)
        validator = DataValidator()
        result = validator.validate(df)
        assert result["is_valid"] is True
        assert len(result["missing_columns"]) == 0

    def test_train_test_split_proportions(self):
        """Split deve respeitar proporção configurada."""
        from src.data.data_loader import ChurnDataLoader
        loader = ChurnDataLoader()
        df = loader.load(use_synthetic=True)
        X_train, X_test, y_train, y_test = loader.get_splits(df)

        total = len(X_train) + len(X_test)
        test_ratio = len(X_test) / total
        assert abs(test_ratio - 0.20) < 0.02, f"Test ratio: {test_ratio:.2%}"

    def test_stratified_split_preserves_distribution(self):
        """Split estratificado deve preservar distribuição de classes."""
        from src.data.data_loader import ChurnDataLoader
        loader = ChurnDataLoader()
        df = loader.load(use_synthetic=True)
        X_train, X_test, y_train, y_test = loader.get_splits(df)

        train_churn_rate = y_train.mean()
        test_churn_rate = y_test.mean()
        # Taxa de churn deve ser similar (diferença < 2 pp)
        assert abs(train_churn_rate - test_churn_rate) < 0.02


# ---------------------------------------------------------------------------
# Testes de Feature Engineering
# ---------------------------------------------------------------------------

class TestFeatureEngineering:
    """Testes do pipeline de feature engineering."""

    @pytest.fixture
    def raw_df(self):
        from src.data.data_loader import generate_synthetic_dataset
        return generate_synthetic_dataset(n_samples=300)

    def test_numerical_creator_adds_features(self, raw_df):
        """NumericalFeatureCreator deve criar novas colunas."""
        from src.features.feature_engineering import NumericalFeatureCreator
        creator = NumericalFeatureCreator()
        creator.fit(raw_df)
        result = creator.transform(raw_df)
        new_cols = {"avg_monthly_charges", "customer_value_score", "tenure_squared"}
        assert new_cols.issubset(set(result.columns))

    def test_no_infinite_values_after_transform(self, raw_df):
        """Transformações não devem gerar valores infinitos."""
        from src.features.feature_engineering import NumericalFeatureCreator
        creator = NumericalFeatureCreator()
        creator.fit(raw_df)
        result = creator.transform(raw_df[["tenure", "monthly_charges", "total_charges"]])
        # Seleciona colunas numéricas e verifica infinitos
        numeric_cols = result.select_dtypes(include=[np.number]).columns
        assert not np.isinf(result[numeric_cols].values).any()

    def test_categorical_encoder_removes_raw_categories(self, raw_df):
        """Encoder deve remover colunas categóricas originais."""
        from src.features.feature_engineering import CategoricalEncoder
        encoder = CategoricalEncoder()
        encoder.fit(raw_df)
        result = encoder.transform(raw_df)
        assert "contract_type" not in result.columns
        assert "internet_service" not in result.columns
        assert "payment_method" not in result.columns

    def test_categorical_encoder_creates_risk_features(self, raw_df):
        """Encoder deve criar features de risco ordinais."""
        from src.features.feature_engineering import CategoricalEncoder
        encoder = CategoricalEncoder()
        encoder.fit(raw_df)
        result = encoder.transform(raw_df)
        assert "contract_risk" in result.columns
        assert "internet_risk" in result.columns
        assert "payment_risk" in result.columns

    def test_outlier_handler_clips_extreme_values(self):
        """OutlierHandler deve clipar valores extremos."""
        from src.features.feature_engineering import OutlierHandler
        df = pd.DataFrame({"monthly_charges": [20, 50, 70, 95, 1000, -500]})
        handler = OutlierHandler(columns=["monthly_charges"])
        handler.fit(df)
        result = handler.transform(df)
        # Valores extremos devem ser clipados
        assert result["monthly_charges"].max() < 1000
        assert result["monthly_charges"].min() > -500


# ---------------------------------------------------------------------------
# Testes da API
# ---------------------------------------------------------------------------

class TestAPI:
    """Testes de integração da FastAPI."""

    def test_health_check_returns_200(self, api_client):
        response = api_client.get("/health")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "model_version" in data

    def test_root_endpoint(self, api_client):
        response = api_client.get("/")
        assert response.status_code == 200
        data = response.json()
        assert "name" in data
        assert "version" in data

    def test_predict_returns_200(self, api_client, sample_customer_payload):
        response = api_client.post("/predict", json=sample_customer_payload)
        assert response.status_code == 200

    def test_predict_response_schema(self, api_client, sample_customer_payload):
        """Resposta deve conter todos os campos obrigatórios."""
        response = api_client.post("/predict", json=sample_customer_payload)
        data = response.json()
        required_fields = {
            "customer_id", "churn_probability", "prediction",
            "risk_level", "top_risk_factors", "recommended_actions", "model_version",
        }
        assert required_fields.issubset(set(data.keys()))

    def test_churn_probability_in_valid_range(self, api_client, sample_customer_payload):
        """Probabilidade deve estar entre 0 e 1."""
        response = api_client.post("/predict", json=sample_customer_payload)
        prob = response.json()["churn_probability"]
        assert 0.0 <= prob <= 1.0

    def test_high_risk_profile_returns_high_probability(
        self, api_client, sample_customer_payload
    ):
        """Perfil de alto risco deve retornar probabilidade elevada."""
        response = api_client.post("/predict", json=sample_customer_payload)
        data = response.json()
        # Fiber optic + month-to-month + electronic check = alto risco
        assert data["churn_probability"] > 0.4
        assert data["risk_level"] in ("MEDIUM", "HIGH", "CRITICAL")

    def test_low_risk_profile_returns_low_probability(
        self, api_client, low_risk_customer
    ):
        """Perfil de baixo risco deve retornar probabilidade baixa."""
        response = api_client.post("/predict", json=low_risk_customer)
        data = response.json()
        # Two year + bank transfer + tech support = baixo risco
        assert data["churn_probability"] < 0.5
        assert data["risk_level"] in ("LOW", "MEDIUM")

    def test_invalid_contract_type_returns_422(self, api_client, sample_customer_payload):
        """Contract type inválido deve retornar 422."""
        bad_payload = {**sample_customer_payload, "contract_type": "INVALID_CONTRACT"}
        response = api_client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_invalid_internet_service_returns_422(self, api_client, sample_customer_payload):
        """Internet service inválido deve retornar 422."""
        bad_payload = {**sample_customer_payload, "internet_service": "SATELLITE"}
        response = api_client.post("/predict", json=bad_payload)
        assert response.status_code == 422

    def test_missing_required_fields_returns_422(self, api_client):
        """Campos obrigatórios faltando devem retornar 422."""
        response = api_client.post("/predict", json={"customer_id": "TEST"})
        assert response.status_code == 422

    def test_batch_predict_returns_200(self, api_client, sample_customer_payload, low_risk_customer):
        """Predição em lote deve funcionar corretamente."""
        payload = {"customers": [sample_customer_payload, low_risk_customer]}
        response = api_client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert data["total_customers"] == 2
        assert len(data["predictions"]) == 2

    def test_model_info_endpoint(self, api_client):
        """Endpoint de informações do modelo deve retornar dados válidos."""
        response = api_client.get("/model/info")
        assert response.status_code == 200
        data = response.json()
        assert "performance" in data
        assert data["performance"]["roc_auc"] > 0.8

    def test_prediction_is_deterministic(self, api_client, sample_customer_payload):
        """Mesma entrada deve gerar mesma saída (determinismo)."""
        resp1 = api_client.post("/predict", json=sample_customer_payload).json()
        resp2 = api_client.post("/predict", json=sample_customer_payload).json()
        assert resp1["churn_probability"] == resp2["churn_probability"]
        assert resp1["prediction"] == resp2["prediction"]


# ---------------------------------------------------------------------------
# Testes de Robustez
# ---------------------------------------------------------------------------

class TestRobustness:
    """Testes de robustez e casos extremos."""

    def test_minimum_tenure(self, api_client, sample_customer_payload):
        """Tenure = 0 (cliente novo) deve funcionar."""
        payload = {**sample_customer_payload, "tenure": 0, "total_charges": 0.0}
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_maximum_tenure(self, api_client, sample_customer_payload):
        """Tenure máximo deve funcionar."""
        payload = {**sample_customer_payload, "tenure": 100, "total_charges": 12000.0}
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_senior_citizen_flag(self, api_client, sample_customer_payload):
        """Flag de idoso deve ser aceita."""
        payload = {**sample_customer_payload, "senior_citizen": 1}
        response = api_client.post("/predict", json=payload)
        assert response.status_code == 200

    def test_batch_size_limit(self, api_client, sample_customer_payload):
        """Lote de 1000 clientes deve ser aceito."""
        payload = {"customers": [sample_customer_payload] * 100}
        response = api_client.post("/predict/batch", json=payload)
        assert response.status_code == 200
        assert response.json()["total_customers"] == 100
