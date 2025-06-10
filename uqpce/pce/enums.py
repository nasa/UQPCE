from enum import auto, Enum


class Distribution(Enum):
    """
    Defines types of distributions.
    """
    CONTINUOUS = auto()
    NORMAL = auto()
    UNIFORM = auto()
    BETA = auto()
    EXPONENTIAL = auto()
    GAMMA = auto()
    LOGNORMAL = auto()
    CONTINUOUS_EPISTEMIC = auto()

    DISCRETE = auto()
    DISCRETE_UNIFORM = auto()
    NEGATIVE_BINOMIAL = auto()
    POISSON = auto()
    HYPERGEOMETRIC = auto()
    DISCRETE_EPISTEMIC = auto()


class UncertaintyType(Enum):
    """
    Defines types of uncertainty.
    """
    ALEATORY = auto()
    EPISTEMIC = auto()
    MIXED = ()


class AlphabetOptDesigns(Enum):
    """
    Defines types of alphabet optimal designs.
    """
    A = auto()
    D = auto()
