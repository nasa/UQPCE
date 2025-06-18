from typing import Union
import warnings

import numpy as np

from uqpce.pce.io import read_input_file
from uqpce.pce.pce import PCE
import openmdao.api as om


def initialize(input_file: str='input.yaml', matrix_file: str='run_matrix.dat', order: Union[None, int]=None):
 
    var_dict, settings = read_input_file(input_file)
    if 'plot' in settings.keys():
        settings.pop('plot')
    if 'verbose' in settings.keys():
        settings.pop('verbose')

    if order is not None:
        settings['order'] = order

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        pce = PCE(outputs=False, plot=False, verbose=False, **settings)

    for key, value in var_dict.items():
        pce.add_variable(**value)

    X = np.loadtxt(matrix_file, ndmin=2)

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        pce.fit(X, np.zeros(X.shape[0]))
        cil, cih = pce.confidence_interval()

    return (
        pce._matrix.var_basis_sys_eval, pce._matrix.norm_sq, 
        pce._pbox.var_basis_resamp.astype(float), pce._pbox.aleat_samps, 
        pce._pbox.epist_samps, X.shape[0], pce.order, pce.variables, 
        pce.significance, pce._X
    )

def set_vals(prob: om.Problem, uncert_var_list: np.ndarray, run_matrix: np.ndarray, deterministic: bool=False):
    if not deterministic:
        for i in range(len(uncert_var_list)):
            prob.set_val(uncert_var_list[i].name, val=run_matrix[:,i])
    else:
        for i in range(len(uncert_var_list)):
            prob.set_val(uncert_var_list[i].name, np.ones(len(run_matrix)) * uncert_var_list[i].get_mean())

def deterministic(uncert_var_list: np.ndarray, deterministic: bool=False):
    if deterministic:
        for var in uncert_var_list:
            var.vals = np.ones(len(var.vals)) * var.get_mean()

def analysis(prob: om.Problem, response: str, input_file: str, matrix_file: str, deterministic: bool=False):
    import os
    import subprocess
    import shlex
    from subprocess import PIPE

    if deterministic:
        import sys
        print(
            'To run UQPCE analysis, ensure that you are not executing with '
            '`deterministic = True`. Analysis was not performed.\n', file=sys.stderr
        )
        return

    res_file = f'uq_results_{response.replace(":", "_")}.dat'

    prob.run_model() # Run the model once to get the values
    responses = prob.get_val(response)

    np.savetxt(res_file, responses.T, header='# '+response, fmt='%.25e')
    res_file = f'uq_results_{response.replace(":", "_")}.dat'
    prob.run_model() # Run the model once to get the values
    responses = prob.get_val(response)
    np.savetxt(res_file, responses.T, header='# '+response, fmt='%.25e')
    command = (
                f'python -m uqpce -i {input_file} -m '
                f'{matrix_file} -r {res_file}'
    )
    sys_name = os.name
    is_linux = sys_name == 'posix'
    is_windows = sys_name == 'nt'
    if is_windows:
        subprocess.check_output(shlex.split(command))

    elif is_linux:
        subprocess.call(shlex.split(command), stdout=PIPE, stderr=PIPE)

    os.remove(res_file)

def counter(prob: om.Problem, subsys_name: str,):
    return prob.model.__dict__[subsys_name].counter
