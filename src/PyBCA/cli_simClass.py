"""Public simulator entrypoint.

`BCA_Simulator` now points to the migrated core implementation.
The legacy implementation is still available as
`LegacyBCA_Simulator` for parity checks.
"""

from .core.simulator import BCASimulator, BCA_Simulator
from ._legacy.cli_simClass import BCA_Simulator as LegacyBCA_Simulator

__all__ = ["BCA_Simulator", "BCASimulator", "LegacyBCA_Simulator"]
