"""Configuration loading utilities."""

import yaml
import os
from typing import Dict, Any


def load_config(config_path: str = None) -> Dict[str, Any]:
    """
    Load configuration from YAML file.
    
    Args:
        config_path: Path to config file. If None, uses default config.
        
    Returns:
        Configuration dictionary
    """
    if config_path is None:
        # Use default config
        base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        config_path = os.path.join(base_dir, 'config', 'default_config.yaml')
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    return config


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration
        override_config: Configuration to override base with
        
    Returns:
        Merged configuration
    """
    merged = base_config.copy()
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = merge_configs(merged[key], value)
        else:
            merged[key] = value
    
    return merged
