# Cube Wrangler

Welcome to Cube Wrangler. Cube Wrangler is a Python package that contains utilities for working with [Network Wrangler](https://github.com/wsp-sag/network_wrangler). It contains methods for carrying out the following common tasks:
    
* Creates a set of files than can be read in by a Cube script to create a Cube roadway network.
* Writes out a Network Wrangler transit network in Cube format.  
* Converts an Cube Log file, which is a record of roadway edits done in Cube, into a project card.
* Converts two Cube `LIN` files, which are Cube's way of representing transit, int a project card. Note that Cube's log files to not record transit edits. Rather, Cube writes out an updated `LIN` file. Cube Wrangler assesses the differences in the `LIN` files and creates a project card that represents the edits.


## Installation
[NOT YET IMPLEMENTED]

The Cube Wrangler package is available on PyPI. If you are managing multiple python versions, we suggest using [`virtualenv`](https://virtualenv.pypa.io/en/latest/) or [`conda`](https://conda.io/en/latest/) virtual environments. `conda` is the environment manager that is contained within both the Anaconda and mini-conda applications.

An example instalion using conda in the command line is as follows:

```bash
conda config --add channels conda-forge
conda create python=3.10 -n your_environment_name
conda activate your_environment_name
pip install cube_wrangler
```



