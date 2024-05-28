# Use the 4S (Signal Safe Speckle Subtraction) for High-Contrast Imaging
![Python 3.8 | 3.9](https://img.shields.io/badge/python-3.8_|_3.9-blue)
[![Documentation Status](https://readthedocs.org/projects/fours/badge/?version=latest)](https://fours.readthedocs.io/en/latest/?badge=latest)
---

This is the documentation of ``fours``, a Python package for PSF subtraction
with the 4S algorithm for exoplanet high contrast imaging (HCI).
Using the 4S algorithm, we were able to recover the planet **AF Lep b in archival
data from 2011**. This demonstrates the power of the 4S algorithm for data 
post-processing in HCI.

This repository contains the code needed to reproduce the results of 
our paper (**paper in preparation**).

---

## Documentation
A full documentation of the package, including several examples and tutorials 
can be found [on ReadTheDocs](https://fours.readthedocs.io).

This short guide will walk you through the required steps to set up and install
`fours`.

## Installation

The code of `fours` is available on the [PyPI repository](https://pypi.org/project/fours/)
as well as on 
[GitHub](https://github.com/markusbonse/fours). We *strongly* recommend you 
to use a [virtual environment](https://virtualenv.pypa.io/en/latest/) to install
the package.

### Installation from PyPI

Just run:
```bash
pip install fours
```

### Installation from GitHub

Start by cloning the repository and install `fours` as a Python package:

```bash
git clone git@github.com:markusbonse/fours.git ;
cd fours ;
pip install .
```

In case you intend to modify the package you can install the package in 
"edit mode" by using the `-e` flag:

```bash
pip install -e .
```

### Additional Options

Depending on the use case `fours` can be installed with additional options. 
If you install `fours` from GitHub you can add them by:

```bash
pip install -e ".[option1,option2,...]"
```

The following options are available:
1. `dev`: Adds all dependencies needed to build this documentation page with
[sphinx](https://www.sphinx-doc.org/en/master/).
3. `plotting`: Installs the libraries [seaborn](https://seaborn.pydata.org), 
[matplotlib](https://matplotlib.org) and 
[bokeh](https://docs.bokeh.org/en/latest/)
which we use in our plots.

## Demonstration dataset
If you want to reproduce our results or get some example data to play with 
you can download the data used in our paper.
The data is publicly available at **COMING SOON**.

The repository contains three files:

1. `30_data`: These are the NACO L'-band datasets as hdf5 files. 
The data was pre-processed with [PynPoint](https://pynpoint.readthedocs.io/en/latest/).
2. `70_results`: Contains the intermediate results of our paper. If you don't 
have access to a high-performance computing cluster you can use these files.

## Reproduce our results
Check out the [plot gallery](https://fours.readthedocs.io/en/latest/04_use_the_fours/01_general.html)
in the ``fours`` documentation.

## Authors and Citation
All code was written by Markus J. Bonse.
Detailed information on the citation can be found [here](https://fours.readthedocs.io/en/latest/05_citation.html).