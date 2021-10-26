import subprocess
from subprocess import PIPE
import shlex
import shutil

run_matrix = 'run_matrix.dat'
gen_matrix = 'run_matrix_generated.dat'
results_file = 'results.dat'
gen_results = 'results_generated.dat'

proc = subprocess.Popen(
    shlex.split(
        'python -m PCE_Codes --generate-samples --user-func x0**2+x1**2+x2**2'
    ), stdout=PIPE, stderr=PIPE
)

proc.wait()

shutil.move(gen_matrix, run_matrix)
shutil.move(gen_results, results_file)
