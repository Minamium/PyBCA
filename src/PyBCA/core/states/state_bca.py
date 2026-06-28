from __future__ import annotations

from dataclasses import dataclass

from PyBCA.api.config import Config
from PyBCA.core.simulator import BCA_Simulator


@dataclass
class BCAState:
    simulator: BCA_Simulator


def build_state(config: Config) -> BCAState:
    simulator = BCA_Simulator(
        cellspace_path=config.cellspace_path,
        rule_paths=list(config.rule_paths),
        device=config.device,
        spatial_event_filePath=config.spatial_event_file_path,
        gui_mode=config.gui_mode,
        use_tqdm=config.use_tqdm_bool,
        trial_constant_sweep=config.trial_constant_sweep,
        record_rule_history=config.record_rule_history,
        rule_history_rule_ids=config.rule_history_rule_ids,
    )
    simulator.Allocate_torch_Tensors_on_Device()
    simulator.set_ParallelTrial(config.trials)
    return BCAState(simulator=simulator)
