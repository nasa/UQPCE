"""
Notices:
Copyright 2020 United States Government as represented by the Administrator of 
the National Aeronautics and Space Administration. All Rights Reserved.
Disclaimers
No Warranty: THE SUBJECT SOFTWARE IS PROVIDED "AS IS" WITHOUT ANY WARRANTY OF 
ANY KIND, EITHER EXPRESSED, IMPLIED, OR STATUTORY, INCLUDING, BUT NOT LIMITED 
TO, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL CONFORM TO SPECIFICATIONS, ANY 
IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE, OR 
FREEDOM FROM INFRINGEMENT, ANY WARRANTY THAT THE SUBJECT SOFTWARE WILL BE ERROR 
FREE, OR ANY WARRANTY THAT DOCUMENTATION, IF PROVIDED, WILL CONFORM TO THE 
SUBJECT SOFTWARE. THIS AGREEMENT DOES NOT, IN ANY MANNER, CONSTITUTE AN 
ENDORSEMENT BY GOVERNMENT AGENCY OR ANY PRIOR RECIPIENT OF ANY RESULTS, 
RESULTING DESIGNS, HARDWARE, SOFTWARE PRODUCTS OR ANY OTHER APPLICATIONS 
RESULTING FROM USE OF THE SUBJECT SOFTWARE.  FURTHER, GOVERNMENT AGENCY 
DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, 
IF PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."
 
Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE 
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY 
PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY 
LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, 
INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S 
USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE 
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY 
PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR 
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS 
AGREEMENT.
"""

from datetime import datetime
from warnings import warn
import logging
import warnings

try:
    from mpi4py.MPI import COMM_WORLD as MPI_COMM_WORLD
except:
    warn('Ensure that all required packages are installed.')
    exit()

from PCE_Codes._helpers import _warn
from PCE_Codes.utility import (
    defaults, arg_parser, uqpce_basis, uqpce_setup, uqpce_iter_setup,
    uqpce_outputs, uqpce_report, uqpce_matrix, IterSettings
)

if __name__ == '__main__':

    comm = MPI_COMM_WORLD
    size = comm.size
    rank = comm.rank

#     logging.basicConfig(level=logging.DEBUG, format='%(relativeCreated)6d %(threadName)s %(message)s', filename=f'UQPCE_{rank}.log')
#     logging.debug(f'\n\nBegin:')

    time_start = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')
    time_start = datetime.strptime(time_start, '%Y-%m-%d %H:%M:%S.%f')

    warnings.formatwarning = _warn

    # Updates all of the initial values for UQPCE from defaults or inputs
    def_args = defaults()
    prog_set = arg_parser(def_args)

    iter_set = IterSettings(time_start=time_start)

    prog_set, iter_set = uqpce_setup(prog_set, iter_set)

    for i in range(iter_set.resp_count):

        iter_set = iter_set._replace(resp_idx=i)

        prog_set = prog_set._replace(resp_order=prog_set.order[i])

        prog_set, iter_set = uqpce_iter_setup(prog_set, iter_set)

        prog_set, iter_set, matrix = uqpce_matrix(prog_set, iter_set)

        prog_set, iter_set, model = uqpce_basis(prog_set, iter_set)

        prog_set, iter_set, press_dict = uqpce_outputs(prog_set, iter_set, matrix, model)

        uqpce_report(prog_set, iter_set, press_dict)

    if rank == 0 and prog_set.verbose:
        print('Analysis finished\n\n')
