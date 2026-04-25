# ============================================================
# ChurnShield AI — Makefile
# ============================================================

.PHONY: help setup data train evaluate api test lint clean docker-up docker-down mlflow

PYTHON := python3
VENV := .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------

help:  ## Mostra todos os comandos disponíveis
	@echo ""
	@echo "🛡️  ChurnShield AI — Comandos disponíveis"
	@echo "=========================================="
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST)
	@echo ""

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

setup:  ## Cria ambiente virtual e instala dependências
	@echo "🔧 Configurando ambiente..."
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip
	$(PIP) install -r requirements.txt
	$(VENV)/bin/pre-commit install
	@echo "✅ Ambiente configurado!"

install:  ## Instala apenas dependências (sem criar venv)
	pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Dados
# ---------------------------------------------------------------------------

data:  ## Gera e processa os dados
	@echo "📊 Processando dados..."
	$(PYTHON) -m src.data.data_loader
	@echo "✅ Dados processados e salvos em data/raw/"

data-validate:  ## Valida qualidade dos dados
	@echo "🔍 Validando dados..."
	$(PYTHON) -c "from src.data.data_loader import ChurnDataLoader, DataValidator; \
		loader = ChurnDataLoader(); df = loader.load(use_synthetic=True); \
		v = DataValidator(); print(v.validate(df))"

# ---------------------------------------------------------------------------
# Treinamento
# ---------------------------------------------------------------------------

train:  ## Treina todos os modelos (com MLflow tracking)
	@echo "🚀 Iniciando treinamento..."
	$(PYTHON) -m src.models.train
	@echo "✅ Modelos treinados e registrados no MLflow"

train-fast:  ## Treina apenas modelos rápidos (sem otimização)
	@echo "⚡ Treinamento rápido (sem Optuna)..."
	$(PYTHON) -m src.models.train --no-hyperopt

train-optimize:  ## Treina com otimização bayesiana completa
	@echo "🔬 Treinamento com otimização bayesiana (pode demorar)..."
	$(PYTHON) -m src.models.train --optimize

# ---------------------------------------------------------------------------
# Avaliação
# ---------------------------------------------------------------------------

evaluate:  ## Avalia o melhor modelo e gera relatório
	@echo "📈 Avaliando modelos..."
	$(PYTHON) -m src.models.evaluate
	@echo "📊 Relatório salvo em reports/"

# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------

api:  ## Sobe a API FastAPI em desenvolvimento
	@echo "🌐 Iniciando API em http://localhost:8000"
	@echo "📖 Docs: http://localhost:8000/docs"
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload

api-prod:  ## Sobe a API em modo produção (4 workers)
	@echo "🚀 API em modo produção..."
	uvicorn api.main:app --host 0.0.0.0 --port 8000 --workers 4

# ---------------------------------------------------------------------------
# MLflow
# ---------------------------------------------------------------------------

mlflow:  ## Sobe o servidor MLflow UI
	@echo "📊 MLflow UI em http://localhost:5000"
	mlflow ui --host 0.0.0.0 --port 5000 --backend-store-uri mlruns/

# ---------------------------------------------------------------------------
# Testes
# ---------------------------------------------------------------------------

test:  ## Executa todos os testes
	@echo "🧪 Executando testes..."
	$(PYTEST) tests/ -v --cov=src --cov=api --cov-report=html --cov-report=term-missing
	@echo "📊 Relatório de cobertura: htmlcov/index.html"

test-fast:  ## Executa testes sem cobertura (mais rápido)
	$(PYTEST) tests/ -v -x

test-api:  ## Executa apenas testes da API
	$(PYTEST) tests/test_churnshield.py::TestAPI -v

test-data:  ## Executa apenas testes de dados
	$(PYTEST) tests/test_churnshield.py::TestDataLoader -v

# ---------------------------------------------------------------------------
# Qualidade de código
# ---------------------------------------------------------------------------

lint:  ## Verifica qualidade do código (ruff)
	@echo "🔍 Verificando código..."
	ruff check src/ api/ tests/
	@echo "✅ Código OK"

format:  ## Formata o código (black + ruff)
	@echo "✨ Formatando código..."
	black src/ api/ tests/
	ruff check --fix src/ api/ tests/

type-check:  ## Verificação de tipos (mypy)
	mypy src/ api/ --ignore-missing-imports

# ---------------------------------------------------------------------------
# Docker
# ---------------------------------------------------------------------------

docker-build:  ## Constrói imagens Docker
	@echo "🐳 Construindo imagens..."
	docker-compose build

docker-up:  ## Sobe todos os serviços (API + MLflow)
	@echo "🐳 Subindo containers..."
	docker-compose up -d
	@echo "✅ API: http://localhost:8000 | MLflow: http://localhost:5000"

docker-down:  ## Para todos os containers
	docker-compose down

docker-logs:  ## Exibe logs dos containers
	docker-compose logs -f

# ---------------------------------------------------------------------------
# Pipeline completo
# ---------------------------------------------------------------------------

pipeline:  ## Executa pipeline completo (dados → treino → avaliação)
	@echo "🔄 Executando pipeline completo..."
	$(MAKE) data
	$(MAKE) train
	$(MAKE) evaluate
	@echo "🏁 Pipeline concluído!"

# ---------------------------------------------------------------------------
# Limpeza
# ---------------------------------------------------------------------------

clean:  ## Remove arquivos temporários e cache
	@echo "🧹 Limpando..."
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name "htmlcov" -exec rm -rf {} + 2>/dev/null || true
	rm -f .coverage
	@echo "✅ Limpo!"

clean-data:  ## Remove dados processados (mantém raw)
	rm -rf data/processed/*
	@echo "✅ Dados processados removidos"

clean-models:  ## Remove modelos treinados
	rm -rf models/*
	@echo "✅ Modelos removidos"

clean-all: clean clean-data clean-models  ## Remove tudo (cuidado!)
	rm -rf mlruns/
	@echo "✅ Limpeza total concluída"
