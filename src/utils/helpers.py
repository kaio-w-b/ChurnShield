"""
ChurnShield AI — Utilitários Gerais
=====================================
Funções auxiliares reutilizadas em todo o projeto.
"""

from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import joblib
import yaml
from loguru import logger


# ---------------------------------------------------------------------------
# Configuração
# ---------------------------------------------------------------------------

def load_config(config_path: str = "config/config.yaml") -> dict:
    """Carrega o arquivo de configuração YAML."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Arquivo de configuração não encontrado: {path}")
    with open(path) as f:
        config = yaml.safe_load(f)
    logger.info(f"Configuração carregada de: {path}")
    return config


def get_project_root() -> Path:
    """Retorna o diretório raiz do projeto."""
    return Path(__file__).parent.parent.parent


# ---------------------------------------------------------------------------
# Persistência de modelos
# ---------------------------------------------------------------------------

def save_model(model: Any, path: str, metadata: Optional[Dict] = None) -> str:
    """
    Salva um modelo e metadados opcionais em disco.

    Args:
        model: Objeto do modelo a salvar.
        path: Caminho do arquivo .joblib.
        metadata: Dicionário com informações extras (métricas, versão, etc).

    Returns:
        Caminho absoluto do arquivo salvo.
    """
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    logger.info(f"Modelo salvo em: {path}")

    if metadata:
        meta_path = path.replace(".joblib", "_metadata.json")
        metadata["saved_at"] = datetime.now().isoformat()
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)
        logger.info(f"Metadados salvos em: {meta_path}")

    return str(Path(path).resolve())


def load_model(path: str) -> Any:
    """Carrega um modelo do disco."""
    if not Path(path).exists():
        raise FileNotFoundError(f"Modelo não encontrado: {path}")
    model = joblib.load(path)
    logger.info(f"Modelo carregado de: {path}")
    return model


# ---------------------------------------------------------------------------
# Timer e profiling
# ---------------------------------------------------------------------------

class Timer:
    """Context manager para medir tempo de execução."""

    def __init__(self, name: str = ""):
        self.name = name
        self.elapsed: float = 0.0

    def __enter__(self) -> "Timer":
        self._start = time.perf_counter()
        return self

    def __exit__(self, *args) -> None:
        self.elapsed = time.perf_counter() - self._start
        label = f"[{self.name}] " if self.name else ""
        logger.info(f"⏱️  {label}Tempo: {self.elapsed:.2f}s")


# ---------------------------------------------------------------------------
# Formatação
# ---------------------------------------------------------------------------

def format_metrics(metrics: Dict[str, float], decimals: int = 4) -> str:
    """Formata dicionário de métricas para exibição."""
    lines = ["\n📊 Métricas:"]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"   {k:<25} {v:.{decimals}f}")
        else:
            lines.append(f"   {k:<25} {v}")
    return "\n".join(lines)


def pct(value: float, decimals: int = 1) -> str:
    """Formata valor como porcentagem."""
    return f"{value * 100:.{decimals}f}%"


# ---------------------------------------------------------------------------
# Variáveis de ambiente
# ---------------------------------------------------------------------------

def get_env(key: str, default: Optional[str] = None) -> str:
    """Lê variável de ambiente com fallback."""
    value = os.environ.get(key, default)
    if value is None:
        raise ValueError(f"Variável de ambiente obrigatória não definida: {key}")
    return value


# ---------------------------------------------------------------------------
# Seed global
# ---------------------------------------------------------------------------

def set_global_seed(seed: int = 42) -> None:
    """Define seed global para reprodutibilidade."""
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    logger.info(f"Seed global definida: {seed}")
