#                                                                              *
# Created by:                 Joanna Schmidt                                   *
#                               2019/11/27                                     *
#                                                                              *
# An error class to be used with the UQPCE Wrapper.                            *
#                                                                              *
#*******************************************************************************
import os
import numpy as np

from uqpce_wrapper._helpers import in_dict


class Parse():
    """
    Generic Parser class.
    """

    def __init__(self):
        return

    def _read_run_matrix(self, file_name):
        """
        Inputs:    file_name- name of file containing matrix
        
        Reads in a matrix of arbitrary size and returns its outputs.
        """
        with open(file_name, 'r') as matrix:
            lines = matrix.readlines()

        temp_lines = lines.copy()
        line_count = len(temp_lines)
        for line in temp_lines:
            if '#' not in line:
                elem_count = len(line.split())
                break

        outputs = [np.zeros(line_count) for i in range(elem_count)]

        # fill arrays with values
        idx = 0
        for line in lines:
            if '#' not in line:
                curr_line = line.split()

                for j in range(elem_count):
                    outputs[j][idx] = curr_line[j]
                idx += 1

        if idx < line_count:
            for i in range(elem_count):
                outputs[i] = outputs[i][0:idx]

        return(outputs)

    def _write_run_matrix(self, *args, file_name, header_str=''):
        """
        Inputs:    args- the arrays of values to write to the file
                   file_name- the name of the file to write the matrix to
                   header_str- an optional line describing the file contents
                    
        Writes a matrix file of arbitrary size.
        """
        out_str = header_str
        line_count = len(args[0])

        for i in range(line_count):
            new_line = ''

            for arg in args:
                new_line = ''.join((new_line, f'{arg[i]}\t'))

            out_str = ''.join((out_str, new_line, '\n'))

        with open(file_name, 'w') as matrix:
            matrix.write(out_str)


class ParseUQPCE(Parse):
    """
    Parser for UQPCE.
    """

    def __init__(self):
        super().__init__()

    def write_results(self, *args, file_name='results.dat'):
        """
        Inputs:     args- the arrays of values to write to the file
                   file_name- the name of the file to write the matrix to
                   
        A wrapper function for the generic parser _write_run_matrix method.
        """
        self._write_run_matrix(*args, file_name=file_name)


class ParseCustom(Parse):
    """
    Parser for PointMass (3DOF).
    """

    def __init__(self):
        super().__init__()

    def parse_output(self, output):
        """
        Inputs:    output- text outputs from running the model
        
        Parses the output to acquire the numbers from the simulation.
        """
        value = []

        output = output.split('\n')
        line_count = len(output)

        for idx in range(line_count):
            line = output[idx]

            if 'your key word' in line:
                value.append(line.split()[0])  # grab the value you need

        return value

    def write_inputs(self, name_dict):
        """
        Inputs:     name_dict- the dictionary of the present variables and their
                    corresponding arrays
        
        Write the UQPCE matrix values to your executable input files here.
        """
        pass
