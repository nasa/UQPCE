#*******************************************************************************
#                                                                              *
# Created by:                 Joanna Schmidt                                   *
#                               2019/11/22                                     *
#                                                                              *
# A main method to create files, execute a program, and execute UQPCE with one *
# command.                                                                     *
#                                                                              *
#*******************************************************************************

import glob
import os
import shlex
import shutil
import subprocess
from subprocess import PIPE
import sys
import warnings

try:  # UQPCE v0.3.*
    from PCE_Codes._helpers import _warn
    from PCE_Codes.io import DataSet
except ModuleNotFoundError:  # UQPCE v0.2.*
    from PCE_Codes.UQPCE import DataSet, _warn

from uqpce_wrapper.parse import ParseCustom, ParseUQPCE
from uqpce_wrapper.errors import EnvironError
from uqpce_wrapper.compile import CustomCompiler

if __name__ == '__main__':

    warnings.formatwarning = _warn

    exec_path_var = 'EXECPATH'
    path_var = 'PATH'
    run_matrix = 'run_matrix.dat'
    gen_matrix = 'run_matrix_generated.dat'
    input_file = 'input.yaml'
    results_file = 'results.dat'

    empty_list = []  # no access to commandline args, input empty list
    order = 2  # default

#     parser = ParseCustom()  # change out this parser if using diff program
#     compiler = CustomCompiler() # Most users will not use the compilers
#     uq_parser = ParseUQPCE()

    #------------------------- check executable path ---------------------------
    try:
        exec_path = os.environ[exec_path_var]

    except KeyError:
        tb = sys.exc_info()[2]
        _warn('The EXECPATH variable was not found.')

    #--------------------------- preprocessing steps ---------------------------
    if not os.path.isfile(run_matrix):
        print('Generating run_matrix.dat file\n')

        proc = subprocess.Popen(
            shlex.split('python -m PCE_Codes --generate-samples'),
            stdout=PIPE, stderr=PIPE
        )
        proc.wait()  # wait until finished

        try:  # UQPCE v0.3.*
            shutil.move(gen_matrix, run_matrix)
        except FileNotFoundError:  # UQPCE v0.2.*
            out_folder = sorted(glob.glob('outputs*'))[-1]
            temp_folder = os.path.join(out_folder, gen_matrix)
            shutil.move(temp_folder, run_matrix)  # move the file
            os.rmdir(out_folder)  # delete the folder

    #--------------------------- read UQPCE variables --------------------------
    warnings.filterwarnings(action='ignore')  # ignore warnings from UQPCE

    init = DataSet()

    try:  # UQPCE v0.3.*
        var_dict = init.check_settings(input_file, empty_list, empty_list)[0]
        var_list = init.read_var_input(var_dict, order)
    except AttributeError:  # UQPCE v0.2.*
        var_list = init.read_var_input(input_file, empty_list, order, verbose=False)[0]  # ignore settings

    init.read_var_vals(run_matrix, 'vals')

    # region : running Model
    iter_count = len(var_list[0].vals)

    print(f'Your program has {iter_count} executions\n')

    arr_dict = {var.name:var.vals for var in var_list}  # dict of arrays

    #------------------------------ parse inputs -------------------------------
#     parser.write_inputs(arr_dict)

    #--------------------------------- compile ---------------------------------
#     print('Compiling your program\n')
#     pcompiler.compile()

    #--------------------------------- execute ---------------------------------
    print('Beginning execution of your program\n')
    if exec_path[-3:] == '.py':  # execute python scripts
        proc = subprocess.Popen(
            shlex.split(f'python {exec_path}'), stdout=PIPE, stderr=PIPE
        )
        proc.wait()  # wait until finished

    else:
        _warn('No python executable was found.')

    #------------------------------ read results -------------------------------
#     print('Parsing the results\n')
#     outputs = parser.parse_output(outputs)

    #-------------------------- write to UQPCE results -------------------------
#     print('Writing the results to results.dat\n')
#     uq_parser.write_results(outputs, file_name=results_file)

    #-------------------------------- run UQPCE --------------------------------
    print('Running UQPCE\n')

    proc = subprocess.Popen(
        shlex.split('python -m PCE_Codes'), stdout=PIPE, stderr=PIPE
    )
    proc.wait()  # force it to wait until finished

    print('Finished\n')

