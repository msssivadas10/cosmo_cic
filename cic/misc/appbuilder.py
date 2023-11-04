#!/usr/bin/python3

import sys, os
import re
import logging
import yaml
from math import inf
from argparse import ArgumentParser
from typing import Any, TypeVar, Type, Callable


_TypeInfo = TypeVar('_TypeInfo', type, tuple[type])


def classname(__type: _TypeInfo) -> str | list[str]:
    r"""
    Return the name(s) of the type(s), that is the name of the class.

    Parameters
    ----------
    __type: type or tuple of types

    Returns
    -------
    name: str or tuple of str

    """

    if isinstance(__type, tuple):
        if not len(__type): 
            raise ValueError("tuple cannot be empty")
        return tuple(classname(__type_i) for __type_i in __type)
    
    if not isinstance(__type, type): 
        raise TypeError("input must be a type or tuple of types")
        
    m = re.search( '(?<=\<class \')[\.\w]+(?=\'>)', repr(__type) )
    if not m: 
        raise ValueError("failed to get the class name")
    return m.group(0).rsplit('.', maxsplit = 1)[-1]

def convert(__value: Any, __types: _TypeInfo, __alt_types: _TypeInfo = ()) -> Any:
    r"""
    Try to convert the value to any of the given types recursively.

    Parameters
    ----------
    __value: Any
        Value to convert.
    __types: type or tuple of types
        Type the value is converted. If multiple types are given, try them in order and value on 
        first successfull conversion is returned.
    __alt_types: type or tuple of types

    Returns
    -------
    converted_value: Any

    """

    if isinstance(__types, tuple): return convert(__value, __types[0], __types[1:])
    if not isinstance(__types, type): raise TypeError("__types must be a type or tuple of types")
    try:
        return __types(__value)
    except Exception:
        if len(__alt_types): return convert(__value, __alt_types[0], __alt_types[1:])
        raise TypeError(f"cannot convert value of type {classname(type(__value))}")
    
def arrayShape(__arr: list) -> tuple:
    r"""
    Recursively find the shape of a (multi-dimensional) array.

    Parameters
    ----------
    __arr: array or scalar

    Returns
    -------
    shape: tuple of int

    """

    if not isinstance(__arr, (list, tuple)): return ()
    __shape1 = 0
    __shape2 = None
    for __entry in __arr:
        __shape1     += 1
        __entry_shape = arrayShape(__entry)
        if __shape2 is None: __shape2 = __entry_shape
        if __entry_shape != __shape2: raise TypeError("array does not have a valid shape") #__shape2 = (); break
    if __shape2 is None: __shape2 = ()
    return __shape1, *__shape2

def arrayApply(__func: Callable, __arr: list) -> list:
    r"""
    Recursively apply a function on the entries of a multidimensional array.

    Parameters
    ----------
    __func: callable
    __arr: array or scalar

    Returns
    -------
    res: list

    """
    return __func(__arr) if not isinstance(__arr, (list, tuple)) else [arrayApply(__func, __entry) for __entry in __arr]

def allmap(__func: Callable, *__arr: list) -> bool:
    r"""
    Map the function to the array and check if all results are non-zero. Same as `all(map(__func, *__arr))`.

    Parameters
    ----------
    *__arr: list

    Returns
    -------
    res: bool
    """
    return all(map(__func, *__arr) )

def with_seperation(__str: str, __sep: str = '_') -> str:
    r"""
    Convert from camel-case style to underscore style seperation.

    Parameter
    ---------
    __str: str
    __sep: str, default = `-`

    Returns
    -------
    __newstr: str

    """
    
    if not len(__str): return __str
    __newstr = __str[0]
    caps     = 'QWERTYUIOPASDFGHJKLZXCVBNM'
    for i, __char in enumerate(__str[1:]):
        __newstr += (__sep + __char if __str[i] not in caps and __char in caps else __char)
    return __newstr.lower()


class FieldError(Exception):
    r"""
    Base class of exceptions raised by field instances.
    """

class FieldValueError(FieldError):
    r"""
    Excetion raised when incorrect values are given. 
    """

class FieldTypeError(FieldError):
    r"""
    Exception raised when values of incorrect type are given.
    """

class Field:
    r"""
    Base class of fields.

    Parameters
    ----------
    name: str
    help: str, optional
    optional: bool, default = False
    default: Any, default = None

    Attributes
    ----------
    value: Any

    """

    __slots__     = 'name', '_value', 'help', 'optional', 'default'
    _fields_count = 0

    def __init__(self, 
                 name: str = '', 
                 help: str = '',    
                 optional: bool = False, 
                 default: Any = None,  ) -> None:
        
        if name is None: name = '' 
        if help is None: help = ''
        if not isinstance(name, str): raise TypeError("name must be string")
        if not isinstance(help, str): raise TypeError("help must be string")

        self.name, self.help, self.optional = name.strip(), help.strip(), bool(optional)
        if not len(self.name):
            self.name          = f"{classname(self.__class__).lower()}_{self._fields_count}"
            self._fields_count = self._fields_count + 1
        self._value  = default
        self.default = default

    def __repr__(self) -> str: return f"{classname(self.__class__)}(name='{self.name}', value='{self.value}')"

    def get(self) -> Any:
        r"""
        Return the value of the field. If none, return the default value.
        """ 
        return self._value if self._value is not None else self.default

    def set(self, __value: Any) -> None: 
        r"""
        Set a value for the field.

        Parameters
        ----------
        __value: Any

        """
        self._value = __value
        return 

    @property 
    def value(self) -> Any: return self.get()

    @value.setter
    def value(self, __value: Any) -> None: return self.set(__value)

class ValueField(Field):
    r"""
    A base class representing a field, holding a value (scalar or array) and a name, with 
    additional information.

    Parameters
    ----------
    name: str
    help: str
    optional: bool, default = False
    default, bool, default = None
    ndim: int, default = 0
        If the value is an array, set the dimensions of the array. Default is 0, which means a 
        scalar value. -1 for variable dimension.
    shape: tuple of int
        Shape of the array.
    allow_null: bool, default = True
        Allow None as a value.

    """
    
    __slots__ = 'ndim', 'shape', 'allow_null'

    def __init__(self, 
                 name: str = '', 
                 help: str = '', 
                 optional: bool = False, 
                 default: Any = None, 
                 ndim: int = 0, 
                 shape: int | tuple = None, 
                 allow_null: bool = True) -> None:
        
        if not isinstance(ndim, int): raise TypeError("ndim must be an integer")
        if shape is not None:
            if isinstance(shape, int): shape = (shape, )
            if not isinstance(shape, tuple): 
                raise TypeError("shape must be a tuple of int")
            if not allmap( lambda __o: __o is None or isinstance(__o, int), shape ): 
                raise TypeError("shape must be a tuple of int")
            if not allmap( lambda __o: True if __o is None else __o > 0, shape ): 
                raise TypeError("shape cannot contain negative integers")
            if len(shape) != ndim and ndim > 0: raise TypeError(f"shape should have size {ndim}")
        elif ndim > -1: shape = (None, ) * ndim

        self.ndim, self.shape = ndim, shape
        self.allow_null = allow_null
        super().__init__(name, help, optional, default)

    def check_and_return_value(self, __value: Any) -> Any: 
        r"""
        Check the given value and return it. This is used to check if the given value is 
        valid for the field. It also convert the values if needed.

        Parameters
        ----------
        __value: Any

        Returns
        -------
        checked_value: Any

        """
        return __value

    def set(self, __value: Any) -> None:
        if not isinstance(__value, (list, tuple)):
            if self.ndim > 0: raise TypeError(f"value of '{self.name}' must be an array of dimension {self.ndim}")
            else: __value = self.check_and_return_value(__value)
        else:
            shape = arrayShape(__value)
            ndim  = len(shape)
            if shape[0] and self.ndim >= 0: 
                if ndim != self.ndim: 
                    raise TypeError(f"value of '{self.name}' must be a {self.ndim} dimensional array, got {ndim}")
                for x1, x2 in zip(shape, self.shape): 
                    if x1 == x2 or x2 is None: continue
                    raise TypeError(f"value of '{self.name}' must be an array of shape {self.shape}, got {shape}")
        self._value = arrayApply(self.check_and_return_value, __value)
        return
    
    def is_null(self) -> bool: return self.value is None

class NumberField(ValueField):
    r"""
    A field with a numerical value (scalar or array). 

    Parameters
    ----------
    name: str
    help: str, optional
    optional: bool, default = False
    default: int, float, default = None
    integer: bool, default = False
        If true, only integer values will be allowed.
    min, max: int, float
        Minimum and maximum value. Default is -inf and inf.
    include_min, include_max: bool, default = True
        Tells if the minimum (maximum) value included in the valid range or not. 
    ndim: int, default = 0
        If the value is an array, set the dimensions of the array. Default is 0, which means a 
        scalar value. -1 for variable dimension.
    shape: tuple of int
        Shape of the array.
    allow_null: bool, default = True
        Allow None as a value.

    """
    
    __slots__ = 'integer', 'min', 'max', 'include_min', 'include_max'

    def __init__(self, 
                 name: str = '', 
                 help: str = '', 
                 optional: bool = False, 
                 default: Any = None, 
                 integer: bool = False, 
                 min: float | int = -inf, 
                 max: float | int = inf, 
                 include_min: bool = True, 
                 include_max: bool = True, 
                 ndim: int = 0, 
                 shape: int | tuple = None, 
                 allow_null: bool = True, ) -> None:
        
        if not isinstance(min, (int, float)): raise ValueError("min must be a numeric type (int or float)")
        if not isinstance(max, (int, float)): raise ValueError("max must be a numeric type (int or float)")
        if not isinstance(ndim, int): raise TypeError("ndim must be an integer")
        if shape is not None:
            if isinstance(shape, int): shape = (shape, )
            if not isinstance(shape, tuple): raise TypeError("shape must be a tuple of int")
            if not allmap(lambda __o: __o is None or isinstance(__o, int), shape): 
                raise TypeError("shape must be a tuple of int")
            if len(shape) != ndim and ndim > 0: raise TypeError(f"shape should have size {ndim}")

        self.integer = integer 
        self.min     = min
        self.max     = max
        self.include_min = include_min
        self.include_max = include_max
        super().__init__(name, help, optional, default, ndim, shape, allow_null)

    def check_and_return_value(self, __value: Any) -> Any:
        if __value is None and self.allow_null: return None
        if isinstance(__value, str): 
            __value = __value.strip()
            __value = int(__value) if self.integer else float(__value)
        elif not (isinstance(__value, int) or isinstance(__value, float) and not self.integer):
            raise ValueError(f"value of {self.name} must be of type {'int' if self.integer else 'float or int'}")
        __str = None
        if self.include_min and __value < self.min: __str = f"greater than or equal to {self.min}"
        elif __value <= self.min: __str = f"greater than {self.min}"
        elif self.include_max and __value > self.max: __str = f"less than or equal to {self.max}"
        elif __value >= self.max: __str = f"less than {self.max}"
        if __str is not None: raise ValueError(f"value of '{self.name}' must be {__str}")
        return int(__value) if self.integer else float(__value)
    
class StringField(ValueField):
    r"""
    A field with a string value (scalar or array). 

    Parameters
    ----------
    name: str
    help: str, optional
    optional: bool, default = False
    default: int, float, default = None
    length: int, default = -1
        Length of the string. If negative, length can be varied.
    isfile: bool, default = False
        Tells if the value represent a file name or path.
    ndim: int, default = 0
        If the value is an array, set the dimensions of the array. Default is 0, which means a 
        scalar value. -1 for variable dimension.
    shape: tuple of int
        Shape of the array.
        
    """

    __slots__ = 'length', 'isfile'

    def __init__(self, 
                 name: str = '', 
                 help: str = '', 
                 optional: bool = False, 
                 default: Any = None, 
                 length: int = -1, 
                 isfile: bool = False,
                 ndim: int = -1, 
                 shape: int | tuple = None, 
                 allow_null: bool = True, ) -> None:
        
        if not isinstance(ndim, int): raise TypeError("ndim must be an integer")
        if shape is not None:
            if isinstance(shape, int): shape = (shape, )
            if not isinstance(shape, tuple): raise TypeError("shape must be a tuple of int")
            if not allmap(lambda __o: __o is None or isinstance(__o, int), shape): raise TypeError("shape must be a tuple of int")
            if len(shape) != ndim and ndim > 0: raise TypeError(f"shape should have size {ndim}")
        if not isinstance(length, int): raise TypeError("length must be an integer")
        
        self.length = length
        self.isfile = isfile
        super().__init__(name, help, optional, default, ndim, shape, allow_null)

    def check_and_return_value(self, __value: Any) -> Any:
        if __value is None and self.allow_null: return None
        if not isinstance(__value, str): raise TypeError(f"value of '{self.name}' must be a string")
        if self.length > 0 and len(__value) != self.length: 
            raise TypeError(f"value of '{self.name}' must be a string of length {self.length}")  
        return __value

class FieldBlock(Field):
    r"""
    A class representing a block of fields.

    Parameters
    ----------
    name: str
    help: str, optional
    optional: bool, default = False

    """

    def __init__(self, name: str = '', help: str = '', optional: bool = False) -> None:
        super().__init__(name, help, optional, {})

    def __repr__(self) -> str: return f"<{classname(self.__class__)} '{self.name}'>"

    def set(self, __value: Any) -> None: raise TypeError("FieldBlock does not support setting values")

    def fields(self): return tuple(self.value.items())
    
    def add(self, field: Type[Field]) -> None:
        r"""
        Add a field or block of fields to this block.

        Parameters
        ----------
        field: subclass of Field

        """
        if not isinstance(field, Field): raise TypeError("opt must be an field instance")
        __name = field.name
        if not isinstance(__name, str): raise TypeError("parameter must have a valid string name")
        if not len(__name): raise TypeError("name cannot be an empty string")
        if __name in self.value: raise KeyError(f"name {__name} already exist")
        self.value[__name] = field
        return
        
    def __getitem__(self, __name: str) -> Any: 
        if isinstance(__name, tuple):
            if len(__name) < 2: __name = __name[0]
            else:
                block = self.__getitem__(__name[0])
                if not isinstance(block, FieldBlock): raise TypeError(f"{__name[0]} is not a FieldBlock")
                return block[__name[1:]]
        if __name not in self.value.keys(): raise KeyError(f"name {__name} does not exist")
        item = self.value[__name]
        return item if isinstance(item, FieldBlock) else item.value
    
    def __setitem__(self, __name: str, __value: Any) -> None:
        if isinstance(__name, tuple):
            if len(__name) < 2: __name = __name[0]
            else:
                block = self.__getitem__(__name[0])
                if not isinstance(block, FieldBlock): raise TypeError(f"{__name[0]} is not a FieldBlock")
                block[__name[1:]] = __value
                return
        if __name not in self.value.keys(): raise KeyError(f"name {__name} does not exist")
        item = self.value[__name]
        if isinstance(item, FieldBlock): raise TypeError(f"cannot set value to a FieldBlock '{__name}'")
        item.value = __value
        return
    
    def print(self, indent_level: int = 0, buffer: object = None) -> None:
        r"""
        Print block as a tree.

        Parameters
        -----------
        indent_level: int, default = 0
        buffer: object
            If given, must be a file object, to which the data is written. Default is `sys.stderr`.

        """
        if buffer is None: buffer = sys.stderr
        for _, field_or_block in self.value.items():
            name = ' ' * (4*indent_level) + field_or_block.name
            if isinstance(field_or_block, FieldBlock): 
                buffer.write(f"{name}:\n")
                field_or_block.print(indent_level + 1, buffer)
            else: buffer.write(f"{name}: {field_or_block.value}\n")
        return
    
    def loadfrom(self, file_or_dict: str) -> None:
        if isinstance(file_or_dict, dict):
            for __key, __value in file_or_dict.items():
                field_or_block = self.value.get(__key)
                if field_or_block is None: continue
                if isinstance(field_or_block, FieldBlock): field_or_block.loadfrom(__value)
                else: field_or_block.value = __value
        elif isinstance(file_or_dict, str):
            if not os.path.exists(file_or_dict): raise FileNotFoundError(f"file {file_or_dict} does not exist")
            try:
                with open(file_or_dict, 'r') as fp: return self.loadfrom(yaml.safe_load( fp ))
            except Exception as e: raise ValueError(f"cannot load values from file, {e}")
        else: raise TypeError(f"argument must be a dict or str filename")

    def add_option(self, 
                   name: str = '', 
                   help: str = '', 
                   optional: bool = False, 
                   cls: str = None,
                   **kwargs: Any,  ) -> None:
        r"""
        Add a new option to this option block, recursively creating new bloks if needed.

        Parameters
        ----------
        name: str
        help: str
        optional: bool, default = False
        cls: str, subclass of Field
            Tells the class of the new field. If string, it must be any of str, file, num, int, float, block or 
            None for the most basic field. Otherwise, it should be a subclass of `Field`. 
        **kwargs: Any
            Other keyword arguments are passed to the field constructor.

        See Also
        --------
        StringField
        NumberField
        ValueField

        """

        name = name.split('.', maxsplit = 1)
        if len(name) > 1:
            block_name, name = name
            if block_name not in self._value: self.add(FieldBlock(block_name))
            return self.__getitem__(block_name).add_option(name, help, optional, cls, **kwargs)
        
        if cls in { 'int', 'float', 'num'}: cls = NumberField
        elif cls in { 'str', 'file' }: cls = StringField
        elif cls in { 'block', 'blk' }: cls = FieldBlock
        elif cls in { None, 'any' }: cls = Field
        elif not issubclass(Field): raise ValueError(f"field type must be str or subclass of Field")
        return self.add(cls(name[0], help, optional, **kwargs))


class ApplicationError(Exception):
    r"""
    Base class of exceptions raised by app objects.
    """

class Application:
    __slots__ = 'options', 'arg_parser', 'name', 'description', 'args'

    def __init__(self, name: str = None, description: str = None) -> None:
        self.options     = FieldBlock()
        self.arg_parser  = None
        self.args        = None
        self.name        = name
        self.description = description

        self.create_options()
        self.create_argparser()
        self.create_argslist()

    def load_options(self, src: dict | str) -> None: 
        r"""
        Load app settings from a source (a json file or a dict).

        Parameters
        ----------
        src: str, dict
            Source dict from which values are taken. If a str is given, must the name of the json/yaml 
            file which the values are loaded.

        """
        return self.options.loadfrom(src)
    
    def add_option(self, 
                   name: str = '', 
                   help: str = '', 
                   optional: bool = False, 
                   cls: str = None,
                   **kwargs: Any,  ) -> None:
        r"""
        Add a new option to this option block, recursively creating new bloks if needed.

        Parameters
        ----------
        name: str
        help: str
        optional: bool, default = False
        cls: str, subclass of Field
            Tells the class of the new field. If string, it must be any of str, file, int, float, block or 
            None for the most basic field. Otherwise, it should be a subclass of `Field`. 
        **kwargs: Any
            Other keyword arguments are passed to the field constructor.

        See Also
        --------
        FieldBlock

        """
        return self.options.add_option(name, help, optional, cls, **kwargs)
    
    def create_argparser(self, **kwargs: Any) -> None:
        r"""
        Create a argument parser for this app, to process command line arguments.
        """
        self.arg_parser = ArgumentParser(prog = self.name, description = self.description, **kwargs)

    def add_argument(self, *args: Any, **kwargs: Any) -> None:
        r"""
        Add an argument to the app's argument parser. Values are directly passed to the 
        parser's `add_argument` method.
        """
        self.arg_parser.add_argument(*args, **kwargs)

    def parse_args(self) -> None:
        r"""
        Parse the arguments for this app and stores in the `args` attribute.
        """
        self.args = self.arg_parser.parse_args()

    def exit(self, __msg: str = None, __code: int = 0) -> None:
        r"""
        Exit the app by calling `sys.exit` with a message.
        """
        if __msg is not None: self.message(__msg, __code)
        sys.exit(1)

    def message(self, __msg: str, __code: int = 0) -> None:
        r"""
        Prints a message to the stderr stream.
        """
        if   __code == 1: sys.stderr.write("error: ")
        elif __code == 2: sys.stderr.write("warning: ")
        else: sys.stderr.write("info: ")
        return sys.stderr.write(__msg + "\n")
    
    def basic_logconfig(self, file: str = None, mode: str = 'a', stderr: bool = True, ) -> None:
        r"""
        Do a basic configuration for the app's logging system.

        Parameters
        ----------
        file: str
        mode: str, default = 'a'
        stderr: bool, default = True
        """
        handlers = []
        if file is not None: handlers.append(logging.FileHandler(file, mode = mode)) 
        if stderr: handlers.append(logging.StreamHandler())
        logging.basicConfig(level = logging.INFO, format = "%(asctime)s [%(levelname)s] %(message)s", handlers = handlers)
        return
    
    def log_info(self, __msg: str, *args, **kwargs) -> None:
        r"""
        Log an INFO message
        """
        return logging.info(__msg, *args, **kwargs)
    
    def log_error(self, __msg: str, *args, **kwargs) -> None:
        r"""
        Log an ERROR message
        """
        return logging.error(__msg, *args, **kwargs)
    
    def log_warning(self, __msg: str, *args, **kwargs) -> None:
        r"""
        Log an INFO message
        """
        return logging.warning(__msg, *args, **kwargs)
    
    def run(self) -> None: 
        r"""
        App's mainloop. Re-define this method with whatever the app will do while runtime! 
        """

    def create_options(self) -> None:
        r"""
        Create the options tree for the app.
        """

    def create_argslist(self) -> None:
        r"""
        Create argument list for the app.
        """

    def start_cli(self) -> None:
        r"""
        Start a command line interface for the app.
        """

    def start_gui(self) -> None:
        r"""
        Start a graphical user interface for the app.
        """
        
