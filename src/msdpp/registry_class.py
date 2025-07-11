from collections.abc import Callable
from typing import NamedTuple

from msdpp.base.divmethod import BaseDiversificationMethod
from msdpp.base.model import BaseModel


class Mapper(NamedTuple):
    models: dict[str, type[BaseModel]]
    div_methods: dict[str, type[BaseDiversificationMethod]]
    sim_funcs: dict[str, Callable]


class ModelTypeError(Exception):
    def __init__(self) -> None:
        super().__init__("model_class must be a subclass of BaseModel")


class ModelKeyConflictError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is already registered in the registry")


class ModelNotFoundError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(
            f"{name} is not registered in the registry. The registered models are: {list(Registry.mapping.models.keys())}"
        )


class DivTypeError(Exception):
    def __init__(self) -> None:
        super().__init__(
            "div_method_class must be a subclass of BaseDiversificationMethod"
        )


class DivKeyConflictError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is already registered in the registry")


class DivMethodNotFoundError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(
            f"{name} is not registered in the registry. The registered methods are: {list(Registry.mapping.div_methods.keys())}"
        )


class SimFuncConflictError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(f"{name} is already registered in the registry")


class SimFuncNotFoundError(Exception):
    def __init__(self, name: str) -> None:
        super().__init__(
            f"{name} is not registered in the registry. The registered functions are: {list(Registry.mapping.sim_funcs.keys())}"
        )


class Registry:
    mapping = Mapper({}, {}, {})

    def __init__(self) -> None:
        pass

    @classmethod
    def register_model(cls, name: str) -> Callable:
        def _register_model(model_class: type[BaseModel]) -> type[BaseModel]:
            if not issubclass(model_class, BaseModel):
                raise ModelTypeError
            if name in cls.mapping.models:
                raise ModelKeyConflictError(name)

            cls.mapping.models[name] = model_class

            return model_class

        return _register_model

    @classmethod
    def get_model(cls, name: str) -> type[BaseModel]:
        if name not in cls.mapping.models:
            raise ModelNotFoundError(name)
        return cls.mapping.models[name]

    @classmethod
    def register_div_method(cls, name: str) -> Callable:
        def _register_div_method(
            div_method_class: type[BaseDiversificationMethod],
        ) -> type[BaseDiversificationMethod]:
            if not issubclass(div_method_class, BaseDiversificationMethod):
                raise DivTypeError
            if name in cls.mapping.models:
                raise DivKeyConflictError(name)

            cls.mapping.div_methods[name] = div_method_class

            return div_method_class

        return _register_div_method

    @classmethod
    def get_div_method(cls, name: str) -> type[BaseDiversificationMethod]:
        if name not in cls.mapping.div_methods:
            raise DivMethodNotFoundError(name)
        return cls.mapping.div_methods[name]

    @classmethod
    def register_sim_func(cls, name: str) -> Callable:
        def _register_sim_func(sim_func: Callable) -> Callable:
            if name in cls.mapping.models:
                raise SimFuncConflictError(name)

            cls.mapping.sim_funcs[name] = sim_func

            return sim_func

        return _register_sim_func

    @classmethod
    def get_sim_func(cls, name: str) -> Callable:
        if name not in cls.mapping.sim_funcs:
            raise SimFuncNotFoundError(name)
        return cls.mapping.sim_funcs[name]
