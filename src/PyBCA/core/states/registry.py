from __future__ import annotations

from typing import Any, Callable

from PyBCA.api.config import Config, Model
from . import state_bca

Factory = Callable[[Config], Any]

_FACTORIES: dict[Model, Factory] = {
    Model.BCA: state_bca.build_state,
}


def build_state(config: Config) -> Any:
    try:
        factory = _FACTORIES[config.model]
    except KeyError as error:
        raise NotImplementedError(f"unknown model: {config.model}") from error
    return factory(config)
