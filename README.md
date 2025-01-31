# UQPCE

Uncertainty Quantification with Polynomial Chaos Expansion ([UQPCE](https://github.com/nasa/UQPCE)) is an open-source, python-based research code for use in parametric, non-deterministic computational studies. UQPCE utilizes a non-intrusive polynomial chaos for computational analyses. The software allows the user to perform an automated uncertainty analysis for any given computational code without requiring modification to the source. UQPCE estimates sensitivities, confidence intervals, and other model statistics, which can be useful in the conceptual design and analysis of flight vehicles. This software was developed for the [Aeronautics Systems Analysis Branch](https://sacd.larc.nasa.gov/asab/) within the [Systems Analysis and Concepts Directorate](https://sacd.larc.nasa.gov/) at [NASA Langley Research Center](https://www.nasa.gov/langley).

## Installation

Install library with `pip install -e .` in the location of your choice.

Run the test case in `python examples/paraboloid/paraboloid_example.py`


## Notices
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

## Disclaimers
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

## Waiver and Indemnity
RECIPIENT AGREES TO WAIVE ANY AND ALL CLAIMS AGAINST THE 
UNITED STATES GOVERNMENT, ITS CONTRACTORS AND SUBCONTRACTORS, AS WELL AS ANY 
PRIOR RECIPIENT.  IF RECIPIENT'S USE OF THE SUBJECT SOFTWARE RESULTS IN ANY 
LIABILITIES, DEMANDS, DAMAGES, EXPENSES OR LOSSES ARISING FROM SUCH USE, 
INCLUDING ANY DAMAGES FROM PRODUCTS BASED ON, OR RESULTING FROM, RECIPIENT'S 
USE OF THE SUBJECT SOFTWARE, RECIPIENT SHALL INDEMNIFY AND HOLD HARMLESS THE 
UNITED STATES GOVERNMENT, ITS CONTRACTORS, AND SUBCONTRACTORS, AS WELL AS ANY 
PRIOR RECIPIENT, TO THE EXTENT PERMITTED BY LAW.  RECIPIENT'S SOLE REMEDY FOR 
ANY SUCH MATTER SHALL BE THE IMMEDIATE, UNILATERAL TERMINATION OF THIS AGREEMENT.