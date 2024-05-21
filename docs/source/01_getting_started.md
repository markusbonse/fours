# Getting Started
This short guide will walk you through the required steps to set up and install
`fours`.

```{attention} 
The code was written for **Python 3.8 and above**
``` 

## Installation

The code of `fours` is available on
[GitHub](https://github.com/markusbonse/fours). We *strongly* recommend you 
to use a [virtual environment](https://virtualenv.pypa.io/en/latest/) to install
the package.

### Installation from PyPI
**Currently not available**
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

If you install `fours` from PiPy you can add them by:

```bash
pip install "fours[option1,option2,...]"
```

The following options are available:
1. `dev`: Adds all dependencies needed to build this documentation page with
[sphinx](https://www.sphinx-doc.org/en/master/).
3. `plotting`: Installs the libraries [seaborn](https://seaborn.pydata.org), 
[matplotlib](https://matplotlib.org) and 
[bokeh](https://docs.bokeh.org/en/latest/)
which we use in our plots.