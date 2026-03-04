from __future__ import annotations

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("PyBCA")
except PackageNotFoundError:
    __version__ = "0.0.0"

# New API layout
from .api import Backend, Config, Engine, LogLevel, Model, Result, Scheme, UseTqdm, run

# Legacy compatibility
from ._legacy.cli_simClass import BCA_Simulator as LegacyBCA_Simulator
from .cli_simClass import BCASimulator, BCA_Simulator
from .lib import (
    StateConversion,
    TransitionRule,
    convert_events_to_array_coordinates,
    extract_cellspace_and_offset,
    get_event_names_from_file,
    get_rule_ids_from_files,
    has_offset_info,
    load_cell_space_yaml_to_numpy,
    load_multiple_state_conversions,
    load_multiple_transition_rules_to_numpy,
    load_multiple_transition_rules_with_probability,
    load_special_events_from_file,
    load_state_conversions_from_yaml,
    load_transition_rules_yaml,
    numpy_to_cell_space_yaml,
)

__all__ = [
    "__version__",
    # New API
    "Backend",
    "Config",
    "Engine",
    "LogLevel",
    "Model",
    "Result",
    "Scheme",
    "UseTqdm",
    "run",
    # Legacy API (compatible imports)
    "BCA_Simulator",
    "BCASimulator",
    "LegacyBCA_Simulator",
    "StateConversion",
    "TransitionRule",
    "convert_events_to_array_coordinates",
    "extract_cellspace_and_offset",
    "get_event_names_from_file",
    "get_rule_ids_from_files",
    "has_offset_info",
    "load_cell_space_yaml_to_numpy",
    "load_multiple_state_conversions",
    "load_multiple_transition_rules_to_numpy",
    "load_multiple_transition_rules_with_probability",
    "load_special_events_from_file",
    "load_state_conversions_from_yaml",
    "load_transition_rules_yaml",
    "numpy_to_cell_space_yaml",
]
