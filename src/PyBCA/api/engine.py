from __future__ import annotations

import logging
import time
from typing import Any

from tqdm import tqdm

from PyBCA.core.schemes.registry import build_stepper
from PyBCA.core.states.registry import build_state

from .config import Config
from .distributed import (
    barrier,
    build_local_save_kwargs,
    merge_event_history_shards,
    prepare_distributed_run,
    shutdown_process_group,
)
from .logging_utils import apply_logging
from .result import Result

logger = logging.getLogger("PyBCA")


class Engine:
    """Top-level wrapper for running PyBCA simulations."""

    def __init__(self, config: Config, apply_logging_flag: bool = True):
        self.distributed = prepare_distributed_run(config)
        self.config = self.distributed.local_config or self.distributed.original_config

        if apply_logging_flag:
            apply_logging(self.config)

        if self.distributed.context.enabled:
            logger.info(
                "Distributed execution: mode=%s rank=%d local_rank=%d world_size=%d local_trials=%d trial_offset=%d active=%s",
                self.distributed.context.mode,
                self.distributed.context.rank,
                self.distributed.context.local_rank,
                self.distributed.context.world_size,
                self.distributed.partition.local_trials,
                self.distributed.partition.trial_offset,
                self.distributed.active,
            )

        if self.distributed.active:
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
        else:
            logger.info("Rank %d has no local trials and will stay idle.", self.distributed.context.rank)
            self.state = None
            self.stepper = None

    def run(self) -> Result:
        start = time.perf_counter()
        simulator = self.state.simulator if self.state is not None else None
        local_event_history = None
        merged_event_history = None

        try:
            if self.distributed.active and self.config.steps > 0:
                iterator = range(self.config.steps)
                if self.config.use_tqdm_bool:
                    iterator = tqdm(iterator, desc="Simulation", unit="step")

                for step_idx in iterator:
                    self.stepper(step_idx)

            if self.distributed.active and self.config.event_history_path is not None:
                save_kwargs = build_local_save_kwargs(self.distributed)
                local_event_history = simulator.save_event_histry_for_dataframe(
                    path=self.config.event_history_path,
                    format=self.config.event_history_format,
                    deduplicate=self.config.event_history_deduplicate,
                    return_df=self.config.event_history_return_df,
                    **save_kwargs,
                )

            if self.distributed.context.enabled:
                barrier(self.distributed.context)
                merged_event_history = merge_event_history_shards(self.distributed)
                barrier(self.distributed.context)

            elapsed = time.perf_counter() - start
            result_event_history = local_event_history
            if (
                self.distributed.context.enabled
                and self.distributed.context.is_master
                and merged_event_history is not None
                and not self.config.event_history_return_df
            ):
                result_event_history = merged_event_history

            meta: dict[str, Any] = {"config": self.config.as_dict}
            if self.distributed.context.enabled:
                meta["original_config"] = self.distributed.original_config.as_dict
                meta["distributed"] = {
                    "active": self.distributed.active,
                    "context": {
                        "mode": self.distributed.context.mode,
                        "rank": self.distributed.context.rank,
                        "local_rank": self.distributed.context.local_rank,
                        "world_size": self.distributed.context.world_size,
                        "backend": self.distributed.context.backend,
                        "is_master": self.distributed.context.is_master,
                    },
                    "partition": {
                        "local_trials": self.distributed.partition.local_trials,
                        "trial_offset": self.distributed.partition.trial_offset,
                        "trial_end": self.distributed.partition.trial_end,
                        "global_trials": self.distributed.partition.global_trials,
                    },
                    "paths": {
                        "run_dir": self.distributed.run_dir,
                        "manifest_path": self.distributed.manifest_path,
                        "rank_config_path": self.distributed.rank_config_path,
                        "event_history_shard_path": self.distributed.shard_event_history_path,
                        "event_history_merged_path": merged_event_history,
                    },
                }

            return Result(
                simulator=simulator,
                current_step=int(getattr(simulator, "_current_step", 0)) if simulator is not None else 0,
                elapsed_sec=float(elapsed),
                event_history=result_event_history,
                meta=meta,
            )
        finally:
            shutdown_process_group(self.distributed.context)


def run(config: Config | dict[str, Any]) -> Result:
    if isinstance(config, dict):
        config = Config(**config)
    return Engine(config, apply_logging_flag=True).run()
