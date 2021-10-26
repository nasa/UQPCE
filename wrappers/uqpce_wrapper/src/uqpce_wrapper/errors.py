#*******************************************************************************
#                                                                              *
# Created by:                 Joanna Schmidt                                   *
#                               2019/11/27                                     *
#                                                                              *
# An error class to be used with the UQPCE Wrapper.                            *
#                                                                              *
#*******************************************************************************


class Error(Exception):
    """
    Base class for Errors.
    """
    pass


class EnvironError(Error):
    """
    Inputs: expression- the expression in which the error occurred
            message- the message to output

    An exception for environment Variables
    """

    def __init__(self, message, exec_path_var):
        self.message = (
            f'Set environment variable {exec_path_var} to be the path to '
            'your executable.'
        )
        exit()
