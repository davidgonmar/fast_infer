from dataclasses import dataclass, fields
from typing import Any, Type


def auto_init_dataclass(cls: Type[Any]) -> Type[Any]:
    cls = dataclass(cls)

    original_init = cls.__init__

    def __init__(self, **kwargs: Any) -> None:
        filtered = dict()
        field_names = {f.name for f in fields(self)}
        for key, value in kwargs.items():
            if key in field_names:
                filtered[key] = value
        original_init(self, **filtered)

    cls.__init__ = __init__
    cls.to_dict = lambda self: {f.name: getattr(self, f.name) for f in fields(self)}
    return cls
