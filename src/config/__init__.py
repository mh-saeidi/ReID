"""Configuration schema, loading and path resolution."""

from src.config.loader import config_from_dict, load_config, load_yaml
from src.config.paths import ProjectPaths
from src.config.schema import AppConfig

__all__ = ["AppConfig", "ProjectPaths", "load_config", "load_yaml", "config_from_dict"]
