from __future__ import annotations

import importlib.util
import json
import shutil
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_config(path: str = "config.yaml") -> dict[str, Any]:
    cfg_path = PROJECT_ROOT / path
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return (PROJECT_ROOT / p).resolve()


def ensure_python_package(module_name: str, install_hint: str = "pip install -r requirements.txt") -> None:
    if importlib.util.find_spec(module_name) is None:
        raise RuntimeError(
            f"Missing Python package for module `{module_name}`. "
            f"Install project dependencies first with `{install_hint}`."
        )


def require_path(path: Path, label: str) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def ensure_command(command_name: str, install_hint: str) -> str:
    exe = shutil.which(command_name)
    if exe is None:
        raise RuntimeError(f"`{command_name}` was not found on PATH. {install_hint}")
    return exe


def fetch_ollama_models() -> set[str]:
    try:
        with urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=5) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise RuntimeError(
            "Ollama is not reachable at http://127.0.0.1:11434. "
            "Start it with `ollama serve` or ensure the Ollama service is running."
        ) from exc

    models = payload.get("models", [])
    return {str(model.get("name", "")).strip() for model in models if model.get("name")}


def _matches_installed_model(required_model: str, installed_models: set[str]) -> bool:
    if required_model in installed_models:
        return True
    if f"{required_model}:latest" in installed_models:
        return True
    if required_model.endswith(":latest") and required_model.removesuffix(":latest") in installed_models:
        return True
    return False


def ensure_ollama_models(required_models: list[str]) -> None:
    installed = fetch_ollama_models()
    missing = [model for model in required_models if not _matches_installed_model(model, installed)]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "Missing Ollama model(s): "
            f"{missing_str}. Pull them first with `ollama pull <model>`."
        )
