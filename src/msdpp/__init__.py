__version__ = "0.0.1"

from msdpp.registry_class import Registry

registry = Registry()

from msdpp import (  # noqa: E402
    base,
    data,
    div_method,
    evalindex,
    models,
    task,
)

__all__ = [
    "base",
    "data",
    "div_method",
    "evalindex",
    "models",
    "task",
]
