import glob
import os
import subprocess
import shlex
from ast import literal_eval
from warnings import warn, showwarning
from subprocess import PIPE

import numpy as np

from PCE_Codes._helpers import _warn


def is_accept(truth, calc, perc=0.01):
    """
    Inputs: truth- the actual value that is expected and has been observed
            calc- the calculated value
            perc- the percent of the truth value that the difference in the 
            value `calc` and the value `truth` can be from one another
    
    Ensures that the difference between the truth value and calculated value is 
    within perc * |truth|.
    """
    truth = np.array(truth)
    calc = np.array(calc)

    if (np.abs(truth - calc) < np.abs(perc * truth)).all():
        return True
    else:
        return False


def same_order(truth, calc):
    """
    Inputs: truth- the actual value that is expected and has been observed
            calc- the calculated value
    
    Ensures that the calculated value is in the same order as the truth value.
    """
    truth = np.array(truth)
    calc = np.array(calc)

    size = len(truth)

    truth_gr = truth > calc
    truth_le = truth < calc
    calc_gr = calc >= truth
    calc_le = calc <= truth

    higher = np.zeros(size)
    higher[truth_gr] = truth[truth_gr]
    higher[calc_gr] = calc[calc_gr]

    lower = np.zeros(size)
    lower[truth_le] = truth[truth_le]
    lower[calc_le] = calc[calc_le]

    ratio = higher / lower

    if (ratio <= 10).all():
        return True
    else:
        return False


if __name__ == '__main__':

    showwarning = _warn

    cases = {
        'analytical':[2], 'diamond_airfoil':[2], 'interval_variable':[2],
        'OneraM6':[3], 'general_variable_analytical':[2], 'small_mean':[2],
        'high_order_analytical':[5], 'naca_0012':[2], 'discrete_variable':[3],
        'discrete_parabola':[2], 'user_input_cont_and_disc':[3]
    }

    stats = {
        'analytical':{
            'mean':[9.8], 'variance':[6.6741], 'conf_int':[[5.6205, 15.5860]],
            'mean_err':[2.6271e-14], 'verify_err':[4.8672e-14]
        },
        'diamond_airfoil':{
            'mean':[116.29, 100.9, 116.5], 'variance':[0.1224, 0.098999, 0.0781],
            'conf_int':[[115.53, 117.17], [100.13, 101.72], [115.92, 117.12]],
            'mean_err':[0.0079942, 0.0061665, 0.0010945],
            'verify_err':[0.012474, 0.0086068, 0.0016291]
        },
        'interval_variable':{
            'mean':[123.6], 'variance':[2567.5], 'conf_int':[[54.931, 249.54]],
            'mean_err':[0.0044884], 'verify_err':[0.014425]
        },
        'OneraM6':{
            'mean':[0.017665, 0.29296], 'variance':[3.6598e-06, 2.713e-4],
            'conf_int':[[0.014453 , 0.021927], [0.26163, 0.3262]],
            'mean_err':[2.8941e-06, 2.8925e-05], 'verify_err':[1.1883e-05, 7.3086e-05]
        },
        'general_variable_analytical':{
            'mean':[17.645], 'variance':[15.225], 'conf_int':[[10.591, 25.657]],
            'mean_err':[2.8777e-14], 'verify_err':[4.2751e-14]
        },
        'small_mean':{
            'mean':[6.4413e-07], 'variance':[1.6863e-14],
            'conf_int':[[4.0616e-07, 9.1431e-07]], 'mean_err':[3.7852e-22],
            'verify_err':[5.1457e-22]
        },
        'high_order_analytical':{
            'mean':[11.403], 'variance':[418.27], 'conf_int':[[0.2492, 40.507]],
            'mean_err':[1.063e-12], 'verify_err':[4.4126e-12]
        },
        'naca_0012':{
            'mean':[0.028205, 0.21875], 'variance':[1.236e-6, 1.425e-4],
            'conf_int':[[0.0253, 0.0313], [0.1843, 0.2484]],
            'mean_err':[4.293e-06, 9.0889e-05], 'verify_err':[6.201e-06, 0.00011863]
        },
        'discrete_variable':{
            'mean':[47.366], 'variance':[190.33], 'conf_int':[[24.44, 76.76]],
            'mean_err':[4.6393e-14], 'verify_err':[6.7448e-14]
        },
        'discrete_parabola':{
            'mean':[12.416], 'variance':[48.815], 'conf_int':[[-0.027, 27.79]],
            'mean_err':[5.293e-15], 'verify_err':[4.6074e-15]
        },
        'user_input_cont_and_disc':{
            'mean':[9.4994], 'variance':[3.8326], 'conf_int':[[5.739, 13.505]],
            'mean_err':[6.9154e-15], 'verify_err':[8.1162e-15]
        }
    }

    verif_fail = 0
    case_keys = cases.keys()

    # run a subprocess for each of the verification cases
    path = '../'
    verif_file = 'verification_cases_test.dat'

    case_count = len(case_keys)

    sys_name = os.name
    is_linux = sys_name == 'posix'
    is_windows = sys_name == 'nt'

    # region: run PCE_Codes for each case
    for case in case_keys:
        print(f'Beginning {case} case\n')

        for order in cases[case]:
            command = (
                f'python -m PCE_Codes -i {case}/input.yaml -m '
                f'{case}/run_matrix.dat -r {case}/results.dat '
                f'--verification-matrix-file {case}/verification_run_matrix.dat '
                f'--verification-results-file {case}/verification_results.dat '
                f'--output-directory outputs_{case}_order{order} -o {order} '
                '--verbose --verify'
            )

            if is_windows:
                subprocess.check_output(shlex.split(command))

            elif is_linux:
                subprocess.call(shlex.split(command), stdout=PIPE, stderr=PIPE)

            else:
                raise OSError('This OS is not yet supported.')

        print(f'Completed {case} case\n')
    # endregion: run PCE_Codes for each case

    # region: write summary of results to file
    with open(verif_file, 'w') as veri_file:
        for case in case_keys:
            # if user has folders with same name
            folder_name = glob.glob(f'outputs_{case}*')

            curr_files = []

            for folder in folder_name:
                curr_files.append(
                    np.array(glob.glob(f'{folder}/output.dat')
                    +glob.glob(f'{folder}/**/output.dat'))
                )

            curr_files = sorted(np.concatenate(curr_files))

            j = 0

            means = []
            variances = []
            mean_errs = []
            sig_to_noises = []
            cils = []
            cihs = []
            verify_errs = []
            err_ratios = []

            for file in curr_files:

                with open(file) as out_file:
                    lines = out_file.readlines()

                for line in lines:

                    if 'Mean of response' in line:
                        means.append(float(line.split()[-1]))

                    elif 'Variance of response' in line:
                        variances.append(float(line.split()[-1]))

                    elif 'Mean error of surrogate' in line:
                        mean_errs.append(float(line.split()[-1]))

                    elif 'Signal to noise ratio' in line:
                        sig_to_noises.append(float(line.split()[-1]))

                    elif 'Confidence Interval on Response' in line:
                        conf_int = ''.join((line.split()[5:]))
                        list_conf_int = literal_eval(conf_int)
                        cils.append(float(list_conf_int[0]))
                        cihs.append(float(list_conf_int[1]))

                    elif 'Mean error between model and verification' in line:
                        verify_errs.append(float(line.split()[-1]))

                    elif 'The ratio of verification error' in line:
                        err_ratios.append(float(line.split()[-1]))

                j += 1

            ci_arr = np.array(stats[case]['conf_int']).T
            stats_cils = ci_arr[0, :]
            stats_cihs = ci_arr[1, :]

            # Check that all values are similar to one of the cases
            if not is_accept(stats[case]['mean'], means):
                warn(f'The mean for case {case} is incorrect.')
                verif_fail += 1
                print('mean', case, stats[case]['mean'], means)

            if not is_accept(stats[case]['variance'], variances):
                warn(f'The variance for case {case} is incorrect.')
                verif_fail += 1
                print('variance', case, stats[case]['variance'], variances)

            if not same_order(stats[case]['mean_err'], mean_errs):
                warn(f'The mean error for case {case} is incorrect.')
                verif_fail += 1
                print('mean_err', case, stats[case]['mean_err'], mean_errs)

            # Ensure that CI are within 5% of the target value.
            if not is_accept(stats_cils, cils, perc=0.05):
                warn(f'The low confidence interval for case {case} is incorrect.')
                verif_fail += 1
                print('cils', case, stats_cils, cils)

            # Ensure that CI are within 5% of the target value.
            if not is_accept(stats_cihs, cihs, perc=0.05):
                warn(f'The high confidence interval for case {case} is incorrect.')
                verif_fail += 1
                print('cihs', case, stats_cihs, cihs)

            if not same_order(stats[case]['verify_err'], verify_errs):
                warn(f'The verification error for case {case} is incorrect.')
                verif_fail += 1
                print('verify_err', case, stats[case]['verify_err'], verify_errs)

            # writing this info for all cases to verification_cases_test.dat
            k = 0
            for file in curr_files:
                veri_file.write(
                    f'{file}:\n\tmean: {means[k]}\n\tvariance: {variances[k]}\n'
                    f'\tmean error: {mean_errs[k]}\n\tverify error: {verify_errs[k]}\n'
                    f'\tsignal to noise: {sig_to_noises[k]}\n'
                    f'\tverification/surrogate error ratio: {err_ratios[k]}\n'
                    f'\tCI: [{cils[k]}, {cihs[k]}]\n\n'
                )

                k += 1
    # endregion: write summary of results to file

    print(
        'A summary of the results from each test case are located in '
        f'{verif_file}\n'
    )

    if not verif_fail:
        print(
            'The tests were all successful and all tested values are in within '
            'the accepted limits\n'
        )
    else:
        warn(
            f'{verif_fail} tests failed and had a value outside of the '
            'accepted limits\n'
        )
