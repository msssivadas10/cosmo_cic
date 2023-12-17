#!/usr/bin/python3

import warnings

class ModelDatabase:
    r"""
    An object to store some models as database. This then allow the user to map 
    models with names and get then easily!
    """
    __slots__ = '_db', 'baseclass', 'name'

    def __init__(self, name: str, baseclass: type | tuple[type]) -> None:
        # name of the database
        assert isinstance(name, str)
        self.name = name
        # base class of the objects in the database
        self.baseclass = baseclass
        # models are stored as key-value pairs in a dictionary
        self._db = {}

    def exists(self, key: str) -> bool:
        r"""
        Check if an item with given key exist or not. 
        """
        return key in self._db

    def keys(self) -> list[str]:
        r"""
        Return the keys in the database.
        """
        return list( self._db.keys() )

    def add(self, key: str, value: object) -> None:
        r"""
        Add a model to the database and map to the given `key`. If `key` already exist, 
        an error is raised. Argument `value` must be of correct type.
        """
        if key in self.keys():
            raise ValueError(f"key already exists: '{key}'")
        if not isinstance(value, self.baseclass):
            raise TypeError(f"incorrect type: '{type(value)}'")
        self._db[key] = value
        return
    
    def remove(self, key: str) -> None:
        r"""
        Remove an item  with given key from the database.
        """
        if key not in self.keys(): return
        warnings.warn(f"removing item from the database: '{self.name}'")
        value = self._db.pop(key)
        return 
    
    def get(self, key: str) -> object:
        r"""
        Get an item with given key, from the database. If key not exist, return None.
        """
        return self._db.get(key, None)
