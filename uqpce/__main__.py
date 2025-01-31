"""
Notices:
Copyright © 2020-2024 United States Government as represented by the 
Administrator of the National Aeronautics and Space Administration. 
All Rights Reserved.

The NASA UQPCE Software (LAR-20514-1) calls the following third-party software, 
which is subject to the terms and conditions of its licensor, as applicable at 
the time of licensing.  The third-party software is not bundled or included with 
this software but may be available from the licensor.  License hyperlinks are 
provided here for information purposes only.

Python 
Copyright © 2001-2024 Python Software Foundation; All Rights Reserved
PSF license:  https://docs.python.org/3/license.html

SymPy 
Copyright (c) 2006-2023 SymPy Development Team
Modified BSD license: https://github.com/sympy/sympy/blob/master/LICENSE

SciPy 
Copyright (c) 2001-2002 Enthought, Inc. 2003-2024, SciPy Developers.
All rights reserved.
Modified BSD license: https://github.com/scipy/scipy/blob/main/LICENSE.txt

NumPy 
Copyright (c) 2005-2024, NumPy Developers. All rights reserved.
Modified BSD license: https://github.com/numpy/numpy/blob/main/LICENSE.txt

MatPlotLib 
Copyright (c) 2012- Matplotlib Development Team; All Rights Reserved
PSF license: https://matplotlib.org/stable/users/project/license.html

MPI4Py 
Copyright (c) 2013, Lisandro Dalcin. All rights reserved.
BSD license: https://github.com/erdc/mpi4py/blob/master/LICENSE.txt

pyYAML 
Copyright (c) 2017-2021 Ingy döt Net
Copyright (c) 2006-2016 Kirill Simonov
MIT license: https://github.com/yaml/pyyaml/blob/main/LICENSE

openMDAO 
Apache License, Version 2.0: https://github.com/OpenMDAO/OpenMDAO/blob/master/LICENSE.txt

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
DISCLAIMS ALL WARRANTIES AND LIABILITIES REGARDING THIRD-PARTY SOFTWARE, IF 
PRESENT IN THE ORIGINAL SOFTWARE, AND DISTRIBUTES IT "AS IS."

Waiver and Indemnity:  RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE 
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY 
PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY 
LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, 
INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S 
USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE 
UNITED STATES GOVERNMENT, ITS CONTRACTORS, AND SUBCONTRACTORS, AS WELL AS ANY 
PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR 
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.
"""
from uqpce.pce.utility import defaults, arg_parser
from uqpce.pce.pce import PCE
import numpy as np

def main():

    defs = defaults()
    args = arg_parser(defs)
    pce = PCE(**args.dict)
    pce.from_yaml(args.input_file)
    pce.update_settings(**args.dict)

    if args.generate_samples:
        Xt = pce.sample()
        np.savetxt('run_matrix_generated.dat', Xt)

        if args.verify:
            Xv = pce.verify_sample()
            np.savetxt('ver_run_matrix_generated.dat', Xv)

        exit()

    Xt = pce.load_matrix_file(args.matrix_file)
    yt = pce.load_matrix_file(args.results_file)

    pce.fit(Xt, yt)
    pce.check_variables(Xt)

    if args.verify:
        Xv = pce.load_matrix_file(args.verification_matrix_file)
        yv = pce.load_matrix_file(args.verification_results_file)
        yverr = pce.verification(Xv, yv)

    pce.sobols()

    cil, cih = pce.confidence_interval()
    if args.stats:
        pce.check_fit()

    pce.print()
    pce.write_outputs()

if __name__ == '__main__':

    main()