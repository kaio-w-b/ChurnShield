# 🛡️ ChurnShield AI
### Predição de churn de clientes com Machine Learning explicável e API de serviço

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green.svg)](https://fastapi.tiangolo.com)
[![SHAP](https://img.shields.io/badge/SHAP-0.45-red.svg)](https://shap.readthedocs.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

---

## 📌 Visão Geral

**ChurnShield AI** é um projeto de ciência de dados de ponta a ponta para predição de churn (cancelamento) de clientes de telecomunicações: da análise exploratória ao treino de modelos de gradient boosting, explicabilidade com SHAP e uma API REST em FastAPI.

> **Problema de negócio**: adquirir um novo cliente custa entre 5x e 25x mais do que reter um existente. Identificar clientes em risco com antecedência permite ações proativas de retenção.

> **Sobre os dados**: os notebooks usam um dataset **sintético** de 7.043 clientes e 21 colunas, gerado por `src/data/data_loader.py` a partir das distribuições do *Telco Customer Churn* (IBM Sample Data), com taxa de churn de 22,1%. Os resultados abaixo valem para esse dataset sintético.

---

## 🏗️ Estrutura do Projeto

```
ChurnShield/
├── notebooks/
│   ├── 01_eda.ipynb                     # Análise exploratória
│   ├── 02_feature_engineering.ipynb     # Criação e seleção de features
│   ├── 03_baseline_models.ipynb         # Dummy, Regressão Logística, Árvore, Random Forest
│   ├── 04_advanced_models.ipynb         # XGBoost, LightGBM, CatBoost e Ensemble
│   └── 05_model_explainability.ipynb    # SHAP (global e por cliente)
│
├── src/
│   ├── data/data_loader.py              # Carga, geração sintética, split e hash dos dados
│   ├── features/feature_engineering.py  # Pipeline de features e SMOTE
│   ├── visualization/plots.py           # Gráficos reutilizáveis
│   └── utils/helpers.py                 # Utilitários
│
├── api/main.py                          # FastAPI: /predict, /predict/batch, /health, /model/info
├── tests/test_churnshield.py            # Testes com Pytest
├── config/config.yaml                   # Configuração central (caminhos, features, modelos)
├── ci.yml                               # Pipeline de CI (lint, testes, build Docker)
├── Dockerfile / docker-compose.yml
├── Makefile
└── requirements.txt
```

---

## 🎯 Stack

| Categoria | Tecnologias |
|-----------|-------------|
| **ML** | Scikit-learn, XGBoost, LightGBM, CatBoost, imbalanced-learn (SMOTE) |
| **Explicabilidade** | SHAP |
| **API** | FastAPI + Pydantic v2 |
| **Testes** | Pytest |
| **Containerização** | Docker, Docker Compose |
| **Visualização** | Matplotlib, Seaborn, Plotly |

---

## 📊 Resultados (conjunto de teste, dataset sintético)

Valores registrados nas saídas dos notebooks `03_baseline_models` e `04_advanced_models`:

| Modelo | AUC-ROC | F1-Score |
|--------|---------|----------|
| Dummy (estratificado) | 0.496 | 0.301 |
| **Regressão Logística** | **0.767** | **0.501** |
| XGBoost | 0.747 | — |
| CatBoost | 0.742 | — |
| Random Forest | 0.743 | 0.381 |
| LightGBM | 0.734 | — |
| Ensemble (Voting soft + calibração) | 0.731 | — |
| Decision Tree | 0.729 | 0.483 |

**Leitura dos resultados**

* Todos os modelos superam com folga a referência aleatória (AUC ≈ 0.50).
* A Regressão Logística teve o melhor AUC no teste. Os modelos de boosting e o ensemble não trouxeram ganho neste dataset, o que é coerente com dados sintéticos gerados a partir de relações majoritariamente lineares.
* Nos notebooks, a validação cruzada dos modelos de boosting (AUC ≈ 0.93) foi feita sobre os dados já balanceados com SMOTE, o que infla o resultado. A métrica confiável é a do conjunto de teste. A correção está no roadmap.

---

## 🔍 Explicabilidade

O notebook `05_model_explainability` usa SHAP `TreeExplainer` para:

* importância global das features (summary plot);
* explicação individual de um cliente de alto risco (force plot);
* conclusão principal: o tipo de contrato (`contract_risk`) é o preditor mais forte, e o tempo de casa (`tenure`) tem efeito protetor.

---

## 🔮 API de Predição

```bash
make api            # ou: uvicorn api.main:app --reload
# Swagger: http://localhost:8000/docs
```

```python
import requests

payload = {
    "tenure": 24,
    "monthly_charges": 89.50,
    "total_charges": 2148.0,
    "contract_type": "Month-to-month",
    "payment_method": "Electronic check",
    "internet_service": "Fiber optic",
    # ... demais campos do schema CustomerFeatures
}

response = requests.post("http://localhost:8000/predict", json=payload)
print(response.json())
# churn_probability, risk_level, top_risk_factors, recommended_actions
```

> ⚠️ **Status atual**: a API ainda não carrega o modelo treinado. O endpoint `/predict` usa uma regra de pontuação baseada nos fatores de risco encontrados na EDA e no SHAP, para demonstrar o contrato da API (schema, validação, níveis de risco e ações de retenção). Servir o modelo treinado é o próximo passo do roadmap.

---

## 🚀 Como Executar

```bash
git clone https://github.com/kaio-w-b/ChurnShield.git
cd ChurnShield
make setup          # cria a venv e instala as dependências

# Notebooks (gera o dataset sintético na primeira execução)
jupyter notebook notebooks/

# Testes
make test           # ou: pytest tests/ -v
```

---

## 🗺️ Roadmap

* [ ] Treinar com o dataset real *Telco Customer Churn* e comparar com o sintético
* [ ] Aplicar SMOTE dentro de cada fold (`imblearn.pipeline.Pipeline`) para uma validação cruzada sem vazamento
* [ ] Salvar o melhor modelo e servi-lo na API no lugar da regra de demonstração
* [ ] Rastrear experimentos e registrar o modelo com MLflow
* [ ] Ativar o CI movendo `ci.yml` para `.github/workflows/`
* [ ] Ajuste de hiperparâmetros com Optuna

---

## 👨‍💻 Autor

**Kaio Wellington Batista e Silva** · [GitHub](https://github.com/kaio-w-b) · [LinkedIn](https://www.linkedin.com/in/kaiowellinghton)

## 📄 Licença

MIT License — veja [LICENSE](LICENSE).
