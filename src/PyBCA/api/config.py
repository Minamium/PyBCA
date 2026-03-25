from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Mapping, Type, TypeVar

E = TypeVar("E", bound=Enum)


def parse_enum(
    enum_cls: Type[E],
    value: str | E,
    *,
    aliases: Mapping[str, E] | None = None,
) -> E:
    if isinstance(value, enum_cls):
        return value

    text = str(value).strip().lower()
    if aliases is not None and text in aliases:
        return aliases[text]

    for member in enum_cls:
        if text == str(member.value).lower():
            return member

    allowed = [str(member.value) for member in enum_cls]
    if aliases:
        allowed += [f"{k}->{v.value}" for k, v in aliases.items()]
    raise ValueError(f"unknown {enum_cls.__name__}: {value!r}. allowed: {allowed}")


class Model(str, Enum):
    BCA = "bca"


class Scheme(str, Enum):
    DEFAULT = "default"


class Backend(str, Enum):
    TORCH = "torch"


class LogLevel(str, Enum):
    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class UseTqdm(str, Enum):
    TRUE = "true"
    FALSE = "false"


@dataclass(frozen=True)
class Config:
    # kineticEQ ライクな最上位選択子
    model: str | Model = Model.BCA
    scheme: str | Scheme = Scheme.DEFAULT
    backend: str | Backend = Backend.TORCH

    # PyBCA 実行設定
    cellspace_path: str = ""
    rule_paths: tuple[str, ...] | list[str] = ()
    device: str = "cuda"
    trials: int = 1
    steps: int = 1
    global_prob: float = 1.0
    seed: int = 0

    spatial_event_file_path: str | None = None
    gui_mode: bool = False
    use_tqdm: str | UseTqdm = UseTqdm.TRUE
    trial_constant_sweep: dict[str, dict[str, float]] | None = None

    state_gate_enable: bool = False
    state_gate_interval: int = 500
    debug: bool = False
    debug_per_trial: bool = False

    log_level: str | LogLevel = LogLevel.INFO

    # run 後の保存オプション
    event_history_path: str | None = None
    event_history_format: str = "jsonl_trials"
    event_history_deduplicate: bool = True
    event_history_return_df: bool = False

    # torchrun による trial 分散
    distributed_mode: str = "off"          # "off" | "auto" | "torchrun"
    distributed_backend: str = "auto"      # "auto" | "nccl" | "gloo"
    distributed_partition: str = "block"   # 現状は block のみ
    distributed_run_dir: str | None = None
    distributed_record_configs: bool = True
    distributed_merge_event_history: bool = True
    distributed_seed_stride: int = 10000019

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "model",
            parse_enum(Model, self.model, aliases={"default": Model.BCA}),
        )
        object.__setattr__(
            self,
            "scheme",
            parse_enum(Scheme, self.scheme, aliases={"bca": Scheme.DEFAULT}),
        )
        object.__setattr__(
            self,
            "backend",
            parse_enum(Backend, self.backend, aliases={"pytorch": Backend.TORCH}),
        )
        object.__setattr__(
            self,
            "use_tqdm",
            parse_enum(
                UseTqdm,
                self.use_tqdm,
                aliases={"1": UseTqdm.TRUE, "0": UseTqdm.FALSE},
            ),
        )
        object.__setattr__(
            self,
            "log_level",
            parse_enum(
                LogLevel,
                self.log_level,
                aliases={"warn": LogLevel.WARNING, "err": LogLevel.ERROR},
            ),
        )
        object.__setattr__(self, "device", str(self.device).strip())
        object.__setattr__(self, "distributed_mode", str(self.distributed_mode).strip().lower())
        object.__setattr__(self, "distributed_backend", str(self.distributed_backend).strip().lower())
        object.__setattr__(self, "distributed_partition", str(self.distributed_partition).strip().lower())

        rule_paths = tuple(self.rule_paths)
        object.__setattr__(self, "rule_paths", rule_paths)

        if not self.cellspace_path:
            raise ValueError("cellspace_path is required.")
        if len(rule_paths) == 0:
            raise ValueError("rule_paths must not be empty.")
        if self.trials <= 0:
            raise ValueError("trials must be >= 1.")
        if self.steps < 0:
            raise ValueError("steps must be >= 0.")
        if not (0.0 <= float(self.global_prob) <= 1.0):
            raise ValueError("global_prob must be in [0, 1].")
        if self.state_gate_interval <= 0:
            raise ValueError("state_gate_interval must be >= 1.")
        if self.distributed_mode not in {"off", "auto", "torchrun"}:
            raise ValueError("distributed_mode must be one of: off, auto, torchrun.")
        if self.distributed_backend not in {"auto", "nccl", "gloo"}:
            raise ValueError("distributed_backend must be one of: auto, nccl, gloo.")
        if self.distributed_partition not in {"block"}:
            raise ValueError("distributed_partition must be 'block'.")
        if self.distributed_seed_stride <= 0:
            raise ValueError("distributed_seed_stride must be >= 1.")

    @property
    def model_name(self) -> str:
        return self.model.value

    @property
    def scheme_name(self) -> str:
        return self.scheme.value

    @property
    def backend_name(self) -> str:
        return self.backend.value

    @property
    def log_level_name(self) -> str:
        return self.log_level.value

    @property
    def use_tqdm_name(self) -> str:
        return self.use_tqdm.value

    @property
    def use_tqdm_bool(self) -> bool:
        return self.use_tqdm == UseTqdm.TRUE

    @property
    def as_dict(self) -> dict[str, object]:
        return {
            "model": self.model_name,
            "scheme": self.scheme_name,
            "backend": self.backend_name,
            "cellspace_path": self.cellspace_path,
            "rule_paths": list(self.rule_paths),
            "device": self.device,
            "trials": self.trials,
            "steps": self.steps,
            "global_prob": self.global_prob,
            "seed": self.seed,
            "spatial_event_file_path": self.spatial_event_file_path,
            "gui_mode": self.gui_mode,
            "use_tqdm": self.use_tqdm_name,
            "trial_constant_sweep": self.trial_constant_sweep,
            "state_gate_enable": self.state_gate_enable,
            "state_gate_interval": self.state_gate_interval,
            "debug": self.debug,
            "debug_per_trial": self.debug_per_trial,
            "log_level": self.log_level_name,
            "event_history_path": self.event_history_path,
            "event_history_format": self.event_history_format,
            "event_history_deduplicate": self.event_history_deduplicate,
            "event_history_return_df": self.event_history_return_df,
            "distributed_mode": self.distributed_mode,
            "distributed_backend": self.distributed_backend,
            "distributed_partition": self.distributed_partition,
            "distributed_run_dir": self.distributed_run_dir,
            "distributed_record_configs": self.distributed_record_configs,
            "distributed_merge_event_history": self.distributed_merge_event_history,
            "distributed_seed_stride": self.distributed_seed_stride,
        }
