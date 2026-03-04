"""Legacy PyBCA implementation modules.

This package keeps the pre-refactor implementation so that
compatibility checks can be performed while the new Engine-based API
is introduced.
"""

from .cli_simClass import BCA_Simulator

__all__ = ["BCA_Simulator"]
