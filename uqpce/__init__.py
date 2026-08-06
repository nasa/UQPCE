__version__ = '1.0.1-dev'

from uqpce.pce.pce import PCE

try:
    from uqpce.mdao import interface
    from uqpce.mdao.uqpcegroup import MultiUQPCEGroup, UQPCEGroup
except:
    pass # openmdao not installed, which is fine if analysis-only is desired
