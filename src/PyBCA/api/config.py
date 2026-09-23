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
    execution_mode: str = "reference"  # reference | torch_sparse | cuda
    rng_mode: str = "legacy"           # legacy | independent
    trial_ids: tuple[int, ...] | list[int] | None = None
    trial_offset: int = 0  # logical sweep index, independent of arbitrary trial IDs
    candidate_capacity: int = 4096
    quiet: bool = False

    # Bounded history chunks and atomic restart checkpoint.
    stream_dir: str | None = None
    flush_interval: int = 1000
    checkpoint_interval: int = 10000
    resume_from: str | None = None

    spatial_event_file_path: str | None = None
    gui_mode: bool = False
    use_tqdm: str | UseTqdm = UseTqdm.TRUE
    trial_constant_sweep: dict[str, dict[str, float]] | None = None

    record_rule_history: bool = False
    rule_history_rule_ids: tuple[int, ...] | list[int] | None = None

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
    rule_history_path: str | None = None
    rule_history_format: str = "jsonl_trials"
    rule_history_deduplicate: bool = False
    rule_history_return_df: bool = False

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
        if self.rule_history_rule_ids is not None:
            rule_history_rule_ids = tuple(int(rule_id) for rule_id in self.rule_history_rule_ids)
            object.__setattr__(self, "rule_history_rule_ids", rule_history_rule_ids)
        if self.rule_history_path is not None and not self.record_rule_history:
            object.__setattr__(self, "record_rule_history", True)
        if self.trial_ids is not None:
            object.__setattr__(self, "trial_ids", tuple(int(t) for t in self.trial_ids))
            if len(self.trial_ids) != self.trials or len(set(self.trial_ids)) != self.trials:
                raise ValueError("trial_ids must contain one unique ID per trial")
            if any(t < 0 or t >= 2**64 for t in self.trial_ids):
                raise ValueError("trial_ids must be unsigned 64-bit integers")
        if self.execution_mode not in {"reference", "torch_sparse", "cuda"}:
            raise ValueError("execution_mode must be reference, torch_sparse, or cuda")
        if self.rng_mode not in {"legacy", "independent"}:
            raise ValueError("rng_mode must be legacy or independent")
        if self.rng_mode == "independent" and (self.execution_mode == "reference" or not 0 <= self.seed < 2**64):
            raise ValueError("Independent RNG needs an optimized mode and an unsigned 64-bit seed")
        if self.candidate_capacity < 1 or self.flush_interval < 1 or self.checkpoint_interval < 1:
            raise ValueError("candidate_capacity and save intervals must be positive")
        if self.trial_offset < 0:
            raise ValueError("trial_offset must be non-negative")
        if self.checkpoint_interval % self.flush_interval:
            raise ValueError("checkpoint_interval must be a multiple of flush_interval")
        if self.stream_dir and (self.event_history_path or self.rule_history_path):
            raise ValueError("Use streaming history or final in-memory history exports, not both")
        if self.resume_from and not self.stream_dir:
            raise ValueError("resume_from requires stream_dir")

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
            "execution_mode": self.execution_mode,
            "rng_mode": self.rng_mode,
            "trial_ids": None if self.trial_ids is None else list(self.trial_ids),
            "trial_offset": self.trial_offset,
            "candidate_capacity": self.candidate_capacity,
            "quiet": self.quiet,
            "stream_dir": self.stream_dir,
            "flush_interval": self.flush_interval,
            "checkpoint_interval": self.checkpoint_interval,
            "resume_from": self.resume_from,
            "spatial_event_file_path": self.spatial_event_file_path,
            "gui_mode": self.gui_mode,
            "use_tqdm": self.use_tqdm_name,
            "trial_constant_sweep": self.trial_constant_sweep,
            "record_rule_history": self.record_rule_history,
            "rule_history_rule_ids": (
                None if self.rule_history_rule_ids is None else list(self.rule_history_rule_ids)
            ),
            "state_gate_enable": self.state_gate_enable,
            "state_gate_interval": self.state_gate_interval,
            "debug": self.debug,
            "debug_per_trial": self.debug_per_trial,
            "log_level": self.log_level_name,
            "event_history_path": self.event_history_path,
            "event_history_format": self.event_history_format,
            "event_history_deduplicate": self.event_history_deduplicate,
            "event_history_return_df": self.event_history_return_df,
            "rule_history_path": self.rule_history_path,
            "rule_history_format": self.rule_history_format,
            "rule_history_deduplicate": self.rule_history_deduplicate,
            "rule_history_return_df": self.rule_history_return_df,
            "distributed_mode": self.distributed_mode,
            "distributed_backend": self.distributed_backend,
            "distributed_partition": self.distributed_partition,
            "distributed_run_dir": self.distributed_run_dir,
            "distributed_record_configs": self.distributed_record_configs,
            "distributed_merge_event_history": self.distributed_merge_event_history,
            "distributed_seed_stride": self.distributed_seed_stride,
        }
