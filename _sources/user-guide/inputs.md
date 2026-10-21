
# Input Files

The required and optional files that UQPCE uses will be discussed below.

```{note}
The basic input files are required when executing UQPCE as a module. Files can be used as inputs for the ``PCE`` object if desired, but there are alternatives that allow users to input required information without input files.
```

## YAML File
	
*default* : ``input.yaml``

* The variable inputs that are of type ``Enum`` allow for any capitalization of the input. 

* All of the values that take a float value can also take ``pi`` as the input value.

* The user-input ``ContinuousVariable`` inputs ``interval_low`` and ``interval_high`` can take ``-oo`` *(-infinity)* and ``oo`` *(infinity)*, respectively.

* Command line arguments can be put in the input file in the "Settings" section:

	* For example, ``--track-convergence-off`` is the flag and ``track_convergence_off: true`` is the equivalent in the "Settings" section.

	* When using the file to set the program options, use the long version of the flag names. Using ``b`` will not work; ``backend`` must be used. 

	* When using a flag option in the input file, these options must be set equal to ``true``. The YAML conventions of Booleans apply.

    * If conflicting values are set in "Settings" and as a command line flag, the command line argument will override the input file argument.

* When adding comments to the YAML file, comments need to begin with a new line a new line and start with ``"#"``. Comments that are in-line with an input should begin with ``" #"`` (space before the ``#``).


### Example

The organization of the YAML file shown in this example is not the only valid way to organize it. YAML files are sensitive to spacing and tabs. Ensure that there are no trailing spaces in the lines. See the [PyYAML](https://pyyaml.org/) documentation for more information.

```
Variable 0:
    name: Mach
    distribution: normal
    mean: 2      
    stdev: 0.02
Variable 1:    # this name doesn't actually get used; call it something helpful
    distribution: uniform
    interval_low: 0
    interval_high: 5000
Variable 2:      
    distribution: continuous
    pdf: 1/sqrt(2*pi*(0.1)**2) * exp(-(x-1)**2/(2*(0.1)**2))
    interval_low: -oo
    interval_high: oo  # indenting with spaces is fine, but
Variable 3: { distribution: continuous, pdf: sin(x), interval_low: 0, interval_high: pi } # this is a valid YAML form
	
Settings:    # Settings can also be in the form that "Variable 4" is in
    order: 3
    significance: 0.05
    verbose: true
    version: true
    plot: true
```


## Matrix File

*default* : ``run_matrix.dat``

* Each column must contain the values for one variable

* Each row contains all of the input values for one run of the program UQPCE is modeling

* Columns should be separated by space or commas

### Example

```{figure} ../images/run_matrix_graphic.png
---
height: 225px
name: run_matrix_ex
---
The setup of the matrix file and how it corresponds to each variable and set of inputs for the program that will be modeled.
```

## Results File

*default* : ``results.dat``

* Each row contains the result corresponding to one row of the run matrix

* There can be more than one column in this file, which also must correspond to one row of the matrix file

* If there are multiple columns in this file, it can be helpful to add a name. These names are included in the file structure so that the outputs are clear.

### Example

```
Propulsion_1st			Propulsion_2nd
44.755				45.675
43.592				56.642
51.861				48.914
50.416				42.973
42.235				52.368
58.457				50.433
```


## Verification Matrix File

*default* : ``verification_run_matrix.dat``

* This file is set up like the matrix file. This file will only be used if the ``--verify`` flag is used.


## Verification Results File

*default* : ``verification_results.dat``

* This file is set up like the results file. This file will only be used if the ``--verify`` flag is used.

```{note}
Using verification runs is highly encouraged and is best practice with UQPCE.
```