"""Utility functions for input/output operations."""
from pathlib import Path

import yaml


def parse_yaml(file_path: Path) -> dict:
    """Parses a YAML file and returns its contents as a dictionary.

    Args:
        file_path (Path): Path to the YAML file.

    Returns:
        dict: Contents of the YAML file.
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        data = yaml.load(file, Loader=yaml.SafeLoader)

    return data
