"""YAML configuration loading with explicit, non-silent overrides."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def load_yaml(path: str | Path) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:
        raise RuntimeError(
            "PyYAML is required to load configuration. Run `pip install -e .`."
        ) from exc
    config_path = Path(path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    with config_path.open("r", encoding="utf-8") as stream:
        value = yaml.safe_load(stream) or {}
    if not isinstance(value, dict):
        raise ValueError(f"configuration root must be a mapping: {config_path}")
    return value


def load_config(
    path: str | Path,
    *,
    base_path: str | Path | None = None,
) -> dict[str, Any]:
    config_path = Path(path)
    override = load_yaml(config_path)
    extends = override.pop("extends", None)
    config: dict[str, Any] = {}
    if base_path:
        config = load_config(base_path)
    elif extends:
        parent = Path(extends)
        if not parent.is_absolute():
            parent = config_path.parent / parent
        config = load_config(parent)
    config = _deep_merge(config, override)
    if "project" not in config or "seed" not in config["project"]:
        raise ValueError("configuration must define project.seed")
    return config
