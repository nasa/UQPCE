class Error(Exception):
    """
    Parent class for UQPCE errors.
    """
    pass


class VariableInputError(Error):
    """
    Inputs: expression- the expression where the error was raised
            message- the message to be printed when the error is raised
    
    Error raised for errors in Variable inputs.
    """

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message
