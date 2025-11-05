"""
Module for working with configuration files.
"""

import copy
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


@dataclass
class Config:
    """Class for storing configuration."""

    raw: Dict[str, Any]

    @property
    def seed(self) -> int:
        return int(self.raw.get("seed", 42))

    @property
    def data(self) -> Dict[str, Any]:
        return self.raw.get("data", {})

    @property
    def model(self) -> Dict[str, Any]:
        return self.raw.get("model", {})

    @property
    def training(self) -> Dict[str, Any]:
        return self.raw.get("training", {})


def load_config(path: str, overrides: Optional[Dict[str, str]] = None) -> Config:
    """
    Load configuration from YAML file.

    Args:
        path: Path to configuration file
        overrides: Dictionary of overrides in format {"key.subkey": "value"}

    Returns:
        Config object with loaded configuration
    """
    with open(path, "r") as f:
        base = yaml.safe_load(f)

    cfg = copy.deepcopy(base) if base else {}

    # Apply overrides in "a.b.c" = value style
    for k, v in (overrides or {}).items():
        _apply_override(cfg, k, v)

    logger.info("Loaded config from %s", path)
    if overrides:
        logger.info("Applied overrides: %s", overrides)
    return Config(cfg)


def _apply_override(cfg: Dict[str, Any], dotted_key: str, value: str) -> None:
    """Apply config override by dotted key."""
    keys = dotted_key.split(".")
    cur = cfg
    for k in keys[:-1]:
        if k not in cur or not isinstance(cur[k], dict):
            cur[k] = {}
        cur = cur[k]
    # Try to cast string to number/bool
    casted: Any = _maybe_cast(value)
    cur[keys[-1]] = casted


def _maybe_cast(v: str):
    """Cast string value to appropriate type."""
    if v.lower() in {"true", "false"}:
        return v.lower() == "true"
    try:
        if "." in v or "e" in v.lower():
            return float(v)
        return int(v)
    except ValueError:
        return v
