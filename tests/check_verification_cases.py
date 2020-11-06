import glob
import os
import subprocess

import numpy as np

if __name__ == '__main__':

    cases = {
        'diamond_airfoil':[2], 'interval_variable':[2], 'OneraM6':[2, 3],
        'analytical':[2], 'general_variable_analytical':[2],
        'high_order_analytical':[5], 'naca_0012':[2]
    }

    case_keys = cases.keys()

    # run a subprocess for each of the verification cases
    path = '../'
    verif_file = 'verification_cases_test.dat'

    case_count = len(case_keys)

    # region: run PCE_Codes for each case
    for case in case_keys:
        os.chdir(f'{case}')
        print(f'Beginning {case} case\n')

        for order in cases[case]:

            p = subprocess.check_output(
                f'python -m PCE_Codes --output-directory '
                f'../outputs_{case}_order{order} -o {order} --verbose --verify'
            )

        os.chdir(path)
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

            curr_files = np.concatenate(curr_files)

            for file in curr_files:

                with open(file) as out_file:
                    lines = out_file.readlines()

                # getting the mean, variance, etc from the output.dat file
                mean = lines[6].split()[-1]
                variance = lines[7].split()[-1]

                mean_err = lines[8].split()[-1]
                verify_err = lines[20].split()[-1]

                sig_to_noise = lines[9].split()[-1]
                conf_int = ''.join((lines[10].split()[5:]))

                # writing this info for all cases to verification_cases_test.dat
                veri_file.write(
                    f'{file}:\n\tmean: {mean}\n\tvariance: '
                    f'{variance}\n\tmean error: {mean_err}\n\tverify error: '
                    f'{verify_err}\n\tsignal to noise: {sig_to_noise}\n\t'
                    f'CI: {conf_int}\n\n'
                )

            # remove the directory
            # shutil.rmtree(folder_name)
    # endregion: write summary of results to file

    print(
        'A summary of the results from each test case are located in '
        f'{verif_file}\n'
    )

        # compare these values to the values from the docs; if they're more than
        # 1% off, raise warning
