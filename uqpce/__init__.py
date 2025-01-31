__version__ = '1.0.0'

from uqpce.pce.pce import PCE
try:
    from uqpce.mdao.uqpcegroup import UQPCEGroup
    import examples.paraboloid.paraboloid.paraboloid as paraboloid
    from uqpce.mdao import interface
except:
    pass # openmdao not installed, which is fine if analysis-only is desired