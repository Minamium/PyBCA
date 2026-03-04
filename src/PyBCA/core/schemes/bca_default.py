from __future__ import annotations

from typing import Callable

from PyBCA.api.config import Config

Stepper = Callable[[int], None]


def build_stepper(config: Config, state) -> Stepper:
    simulator = state.simulator

    def _step(_step_index: int) -> None:
        simulator.step(
            global_prob=config.global_prob,
            seed=config.seed,
            debug=config.debug,
            debug_per_trial=config.debug_per_trial,
            state_gate_enable=config.state_gate_enable,
            state_gate_interval=config.state_gate_interval,
        )

    return _step
