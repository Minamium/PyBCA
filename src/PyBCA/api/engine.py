from __future__ import annotations

import logging
import time
from typing import Any

from tqdm import tqdm

from PyBCA.core.schemes.registry import build_stepper
from PyBCA.core.states.registry import build_state

from .config import Config
from .logging_utils import apply_logging
from .result import Result

logger = logging.getLogger("PyBCA")


class Engine:
    """Top-level wrapper for running PyBCA simulations."""

    def __init__(self, config: Config, apply_logging_flag: bool = True):
        self.config = config

        if apply_logging_flag:
            apply_logging(self.config)

        logger.info(
            "Engine initialized: model=%s scheme=%s backend=%s device=%s trials=%d steps=%d",
            self.config.model_name,
            self.config.scheme_name,
            self.config.backend_name,
            self.config.device,
            self.config.trials,
            self.config.steps,
        )
        logger.debug("Engine config: %s", self.config.as_dict)

        self.state = build_state(self.config)
        self.stepper = build_stepper(self.config, self.state)

    def run(self) -> Result:
        simulator = self.state.simulator
        start = time.perf_counter()

        if self.config.steps > 0:
            iterator = range(self.config.steps)
            if self.config.use_tqdm_bool:
                iterator = tqdm(iterator, desc="Simulation", unit="step")

            for step_idx in iterator:
                self.stepper(step_idx)

        event_history = None
        if self.config.event_history_path is not None:
            event_history = simulator.save_event_histry_for_dataframe(
                path=self.config.event_history_path,
                format=self.config.event_history_format,
                deduplicate=self.config.event_history_deduplicate,
                return_df=self.config.event_history_return_df,
            )

        elapsed = time.perf_counter() - start

        return Result(
            simulator=simulator,
            current_step=int(getattr(simulator, "_current_step", 0)),
            elapsed_sec=float(elapsed),
            event_history=event_history,
            meta={"config": self.config.as_dict},
        )


def run(config: Config | dict[str, Any]) -> Result:
    if isinstance(config, dict):
        config = Config(**config)
    return Engine(config, apply_logging_flag=True).run()
