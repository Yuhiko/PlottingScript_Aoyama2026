# Plotting scripts for Aoyama et al. (2026)

This repository contains plotting scripts associated with the paper:

**Hydrogen Line Emission in Accreting Very-Low-Mass Objects I: Spectral Analysis of the Shock-Origin Narrow Component**

The source data are archived on Zenodo:  
**10.5281/zenodo.19398530**

## Contents
- `GeneratePlot.py`: script used to generate figures
- `PlotLib.py`: script contains the subroutines

## Requirements
- Python 3.11
- numpy
- pandas
- matplotlib
- astropy

## Data
The authoritative source data are archived on Zenodo.

Download `Aoyama2026.parquet` from the Zenodo record and place it in the working directory.

## Usage
Run:
python GeneratePlot.py

The variables for the x- and y-axes can be specified with the `--x` and `--y` options.  
If no options are given, the script generates figures for combinations of major properties (about 400 figures in total).  
For variable names and their units, please refer to `README_data.txt` in the Zenodo record.
