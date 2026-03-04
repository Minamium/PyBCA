from __future__ import annotations

from typing import Any, Callable

from PyBCA.api.config import Backend, Config, Model, Scheme
from . import bca_default

Stepper = Callable[[int], None]
StepperBuilder = Callable[[Config, Any], Stepper]

_FACTORIES: dict[tuple[Model, Scheme, Backend], StepperBuilder] = {
    (Model.BCA, Scheme.DEFAULT, Backend.TORCH): bca_default.build_stepper,
}


def build_stepper(config: Config, state: Any) -> Stepper:
    key = (config.model, config.scheme, config.backend)
    try:
        builder = _FACTORIES[key]
    except KeyError as error:
        raise NotImplementedError(f"scheme not registered: {key}") from error
    return builder(config, state)
