#*******************************************************************************
#                                                                              *
# Created by:                 Joanna Schmidt                                   *
#                               2019/12/06                                     *
#                                                                              *
# Classes to compile other programs.                                           *
#                                                                              *
#*******************************************************************************
import subprocess
from subprocess import PIPE


class Compiler():
    """
    Generic Compiler class.
    """

    def __init__(self):
        pass

    def compile(self):
        """
        Compiles the file into executable.
        """
        pass


class CustomCompiler():
    """
    The user's Compiler class. This is rarely needed by users.
    """

    def __init__(self):
        pass

    def compile(self):
        """
        Include code needed to compile your code here.
        """
        pass
