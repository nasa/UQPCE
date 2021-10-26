"""
Created on Aug 4, 2017
"""
from enum import Enum


class EnumStrConv(Enum):
    """
    Define a base class for simplified conversion of enumerated members to and
    from `str`.

    The conversion process is simplified in two ways:

    1. The `str` value of an enumerated member is the name of the member.
    2. The class method, `from_name`, provides a case-insensitive alternative
        to `cls[name]` enumerated member lookup. Note, derived types may
        override the default implementation, if needed.
    """
    __slots__ = ()

    def __str__(self):
        """
        Define a simplified conversion to `str` using the name of the
        enumerated member.
        """
        return self.name

    @classmethod
    def from_name(cls, name: str):
        """
        Define a simplified case-insensitive conversion from a name (in other
        words, a `str`) to an enumerated member.
        """
        return cls[name.upper()]


class EnableValueCallback:
    """
    Define a base class that uses the value attribute of an instance as a
    callback.
    """
    __slots__ = ()

    def __call__(self, *args, **kwargs):
        return self.value(*args, **kwargs)


class EnumCallback(EnableValueCallback, Enum):
    """
    Define a base class enumerated type that supports callable member values.
    """
    __slots__ = ()
    pass


class EnumCallbackStrConv(EnableValueCallback, EnumStrConv):
    """
    Define a base class enumerated type that supports simplified conversion of
    enumerated members to and from `str` and also supports callable member
    values.
    """
    __slots__ = ()
    pass


class WrapCallback:
    """
    Define a wrapper for functions and bound methods to be used as enumerated
    members.

    Note, functions and bound methods cannot be directly used as enumerated
    member values, as they are interpreted as new methods of the enclosing
    enumerated type. As this type demonstrates, instances whose type defines
    the `__call__` special method work as intended.
    """
    __slots__ = ('callback',)

    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args, **kwargs):
        return self.callback(*args, **kwargs)
