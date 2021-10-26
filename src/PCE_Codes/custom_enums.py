from PCE_Codes.enum import EnumStrConv
from enum import auto


class Distribution(EnumStrConv):
    """
    Defines types of distributions.
    """
    CONTINUOUS = auto()
    NORMAL = auto()
    UNIFORM = auto()
    BETA = auto()
    EXPONENTIAL = auto()
    GAMMA = auto()

    DISCRETE = auto()
    DISCRETE_UNIFORM = auto()
    NEGATIVE_BINOMIAL = auto()
    POISSON = auto()
    HYPERGEOMETRIC = auto()


class UncertaintyType(EnumStrConv):
    """
    Defines types of uncertainty.
    """
    ALEATORY = auto()
    EPISTEMIC = auto()


class AlphabetOptDesigns(EnumStrConv):
    """
    Defines types of alphabet optimal designs.
    """
    A = auto()
    D = auto()
