#!/usr/bin/python3

import re
from math import inf as INF
from dataclasses import dataclass
from typing import Any, TypeVar


_TypeInfo = TypeVar('_TypeInfo', type, tuple[type])


def classname(__type: _TypeInfo) -> str | list[str]:
    r"""
    Return the class name of the type.
    """

    if isinstance(__type, tuple):
        if not len(__type): raise ValueError("tuple cannot be empty")
        return tuple(classname(__type_i) for __type_i in __type)
    
    if not isinstance(__type, type): raise TypeError("input must be a type or tuple of types")
        
    m = re.search( '(?<=\<class \')[\.\w]+(?=\'>)', repr(__type) )
    if not m: raise ValueError("failed to get the class name")
    return m.group(0).rsplit('.', maxsplit = 1)[-1]

def convert(__value: Any, __types: _TypeInfo, __alt_types: _TypeInfo = ()) -> Any:
    r"""
    Try to convert the value to any of the given types recursively.
    """

    if isinstance(__types, tuple): return convert(__value, __types[0], __types[1:])
    if not isinstance(__types, type): raise TypeError("__types must be a type or tuple of types")
    try:
        return __types(__value)
    except Exception:
        return convert(__value, __alt_types[0], __alt_types[1:])

@dataclass
class Parameter:
    name: str
    _value: Any  = None
    default: Any = None

    @property 
    def value(self) -> Any: return self.default if self._value is None else self._value

    @value.setter
    def value(self, __value: Any): self._value = self.checkedValue(__value) 
    
    def checkedValue(self, __value: Any) -> Any: return __value

@dataclass
class FloatParameter(Parameter):
    min: float        = -INF
    max: float        =  INF
    exclude_min: bool = False
    exclude_max: bool = False

    def checkedValue(self, __value: Any) -> Any:
        __value = self.default if __value is None else __value
        assert isinstance(__value, (float, int))
        assert __value > self.min if self.exclude_min else __value >= self.min
        assert __value < self.max if self.exclude_max else __value <= self.min
        return super().checkedValue(__value)

@dataclass
class IntegerParameter(Parameter):
    min: int          = -INF
    max: int          =  INF
    exclude_min: bool = False
    exclude_max: bool = False

    def checkedValue(self, __value: Any) -> Any:
        __value = self.default if __value is None else __value
        assert isinstance(__value, int)
        assert __value > self.min if self.exclude_min else __value >= self.min
        assert __value < self.max if self.exclude_max else __value <= self.min
        return super().checkedValue(__value)

@dataclass
class StateParameter(Parameter):
    values: list[Any] = None

    def checkedValue(self, __value: Any) -> Any:
        __value = self.default if __value is None else __value
        assert self.values is not None
        assert __value in self.values
        return super().checkedValue(__value)

@dataclass
class ArrayParameter(Parameter):
    cls: _TypeInfo = None
    size: int      = None

    def checkedValue(self, __value: Any) -> Any:
        assert isinstance(__value, (list, tuple))
        if self.cls is not None: assert all(map(lambda __o: isinstance(__o, self.cls), __value))
        if self.size is not None: assert len(__value) == self.size
        return super().checkedValue(__value)


class ApplicationError(Exception):
    r"""
    Base class of exceptions raised by app objects.
    """


class ParameterBlock:

    __slots__ = '_mapping'

    def __init__(self) -> None: self._mapping = {}

    def addBlock(self, __name: str) -> None: 
        if not isinstance(__name, str): raise TypeError("block name must be a string")
        if __name in self._mapping: raise ApplicationError(f"name {__name} already exist")
        self._mapping[__name] = ParameterBlock()
        return

    def add(self, opt: Parameter) -> None:
        if not isinstance(opt, Parameter): raise TypeError("opt must be a 'Parameter' instance")
        __name = opt.name
        if not isinstance(__name, str): raise TypeError("parameter name must be a string")
        if __name in self._mapping: raise ApplicationError(f"name {__name} already exist")
        self._mapping[__name] = opt
        return
    
    def __getitem__(self, __name: str) -> Any: return self._mapping[__name]
    

class Application:

    __slots__ = '_args', 'arg_parser'

    def __init__(self) -> None: self._args = ParameterBlock()

    @property
    def opts(self) -> ParameterBlock: return self._args

    def createOptionParser(self) -> None: ...

    def loadOptions(self, file: str) -> None: ...

    def mainloop(self) -> None: ...

