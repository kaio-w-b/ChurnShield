# 🛡️ ChurnShield AI
### Advanced Customer Churn Prediction Platform with Explainable ML & MLOps

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com)
[![MLflow](https://img.shields.io/badge/MLflow-2.12-orange.svg)](https://mlflow.org)
[![SHAP](https://img.shields.io/badge/SHAP-0.45-red.svg)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Visão Geral

**ChurnShield AI** é uma plataforma de ciência de dados de ponta a ponta para predição de churn (cancelamento) de clientes em empresas de telecomunicações. O projeto combina técnicas avançadas de Machine Learning com práticas modernas de MLOps para entregar predições confiáveis, interpretáveis e prontas para produção.

> **Problema de negócio**: Adquirir um novo cliente custa entre 5x a 25x mais do que reter um existente. Identificar clientes em risco de churn com antecedência permite ações proativas de retenção, reduzindo perdas de receita significativamente.

---

## 🏗️ Arquitetura do Projeto

```
churnshield/
├── 📁 data/
│   ├── raw/                    # Dados brutos (imutáveis)
│   ├── processed/              # Dados processados e prontos para modelagem
│   └── external/               # Dados externos (enriquecimento)
│
├── 📁 notebooks/
│   ├── 01_eda.ipynb            # Análise Exploratória de Dados
│   ├── 02_feature_engineering.ipynb
│   ├── 03_baseline_models.ipynb
│   ├── 04_advanced_models.ipynb
│   └── 05_model_explainability.ipynb
│
├── 📁 src/
│   ├── data/                   # Ingestão e validação de dados
│   ├── features/               # Engenharia de features
│   ├── models/                 # Treinamento e inferência
│   ├── visualization/          # Geração de gráficos
│   └── utils/                  # Utilitários gerais
│
├── 📁 api/                     # FastAPI — serviço de predição
├── 📁 tests/                   # Testes unitários e de integração
├── 📁 config/                  # Configurações do projeto
├── 📁 reports/                 # Relatórios e figuras geradas
├── 📁 mlruns/                  # Experimentos MLflow
├── docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## 🎯 Features Técnicas

| Categoria | Tecnologias |
|-----------|-------------|
| **ML Framework** | Scikit-learn, XGBoost, LightGBM, CatBoost |
| **Explicabilidade** | SHAP, LIME, Feature Importance |
| **MLOps** | MLflow (tracking + model registry) |
| **API** | FastAPI + Pydantic v2 |
| **Validação de Dados** | Great Expectations, Pydantic |
| **Testes** | Pytest, pytest-cov |
| **Containerização** | Docker, Docker Compose |
| **CI/CD** | GitHub Actions |
| **Visualização** | Matplotlib, Seaborn, Plotly |

---

## 🚀 Quick Start

### 1. Clone e configure o ambiente

```bash
git clone https://github.com/seu-usuario/churnshield.git
cd churnshield
make setup
```

### 2. Execute o pipeline completo

```bash
# Processar dados
make data

# Treinar modelos (com tracking MLflow)
make train

# Avaliar e gerar relatório
make evaluate

# Subir a API de predição
make api
```

### 3. Acesse os serviços

| Serviço | URL |
|---------|-----|
| 🔮 API de Predição | http://localhost:8000 |
| 📊 MLflow UI | http://localhost:5000 |
| 📖 API Docs (Swagger) | http://localhost:8000/docs |

---

## 📊 Resultados do Modelo

| Modelo | AUC-ROC | F1-Score | Precisão | Recall |
|--------|---------|----------|----------|--------|
| Logistic Regression (baseline) | 0.782 | 0.581 | 0.634 | 0.537 |
| Random Forest | 0.841 | 0.643 | 0.701 | 0.594 |
| XGBoost | 0.867 | 0.672 | 0.723 | 0.628 |
| LightGBM | 0.871 | 0.681 | 0.731 | 0.638 |
| **Ensemble (Voting)** | **0.884** | **0.703** | **0.748** | **0.663** |

> 🏆 **Modelo campeão**: Ensemble com Voting Classifier (LightGBM + XGBoost + CatBoost)

---

## 🔮 Exemplo de Uso da API

```python
import requests

payload = {
    "tenure": 24,
    "monthly_charges": 89.50,
    "total_charges": 2148.0,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
    "num_products": 3,
    "has_tech_support": False,
    "has_online_security": False
}

response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
# {
#   "churn_probability": 0.847,
#   "prediction": "HIGH_RISK",
#   "confidence": "HIGH",
#   "top_risk_factors": ["Month-to-month contract", "No online security", "High monthly charges"],
#   "recommended_actions": ["Offer annual contract discount", "Bundle security package"]
# }
```

---

## 📈 Pipeline de Dados

```
Raw Data → Validation → Cleaning → Feature Engineering → Feature Selection → Train/Test Split
    ↓
Model Training → Hyperparameter Tuning → Ensemble → Evaluation → MLflow Registry
    ↓
Model Serving (FastAPI) → Monitoring → Retraining Trigger
```

---

## 🧪 Executar Testes

```bash
make test
# ou
pytest tests/ -v --cov=src --cov-report=html
```

---

## 📄 Licença

MIT License — veja [LICENSE](LICENSE) para detalhes.
