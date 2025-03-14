# How to get the data

This short tutorial explains how to get the dataset and intermediate results 
necessary to reproduce the results and plots of the 
[Use the 4S](../05_citation.rst) paper. 

## Downloading the data from Zenodo
The data is publicly available at Zenodo. There are two parts of the data available:
1. `30_data`: The raw NACO L’ datasets after pre-processing with PynPoint. They are available [here](https://doi.org/10.5281/zenodo.11456704).
2. `70_results`: The intermediate results of the paper. They are available [here](https://doi.org/10.5281/zenodo.11457071).
Please download and unpack the files. the directory should contain three subdirectories:

## Setting up the environmental variable

Once downloaded, we need to tell fours where the data is on your local 
machine. You can do this by setting the following environment variable:

```bash
export FOURS_ROOT_DIR="/path/to/datasets/dir" ;
```

The given path should be the directory which contains the sub-folders 
`30_data`, `70_results`.