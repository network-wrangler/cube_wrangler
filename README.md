Cube Wrangler is a Python library for CUBE-related utilities with Network Wrangler.

## System Requirements
Cube Wrangler does require Python 3.7+. If you have a different version of Python installed (e.g. from ArcGIS), conda or a similar virtual environment manager can care of installing it for you in the installation instructions below.

## Installation
Cube Wrangler is available from PyPI.

### Latest Official Version
Users can install via standard pip command.
```bash
pip install cube-wrangler
```

Users can also install via package managers such as uv
```bash
uv add cube-wrangler
```

### From GitHub
```bash
pip install git+https://github.com/network-wrangler/cube_wrangler.git@main#egg=cube_wrangler
```

If you wanted to install from a specific tag/version number or branch, replace `@main` with `@<branchname>`  or `@tag`

### From Clone

If you are going to be working on Cube Wrangler locally, you might want to clone it to your local machine and install it from the clone.  The -e will install it in [editable mode](https://pip.pypa.io/en/stable/reference/pip_install/?highlight=editable#editable-installs).

If you have [GitHub desktop](https://desktop.github.com/) installed, you can either do this by using the GitHub user interface by clicking on the green button "clone or download" in the [main cube wrangler repository page](https://github.com/network-wrangler/cube_wrangler).

Otherwise, you can use the command prompt to navigate to the directory that you would like to store your cube wrangler clone and then using a [git command](https://git-scm.com/downloads) to clone it.

```bash
cd path to where you want to put wrangler
git clone https://github.com/network-wrangler/cube_wrangler
```

Then you should be able to install Cube Wrangler in "develop" mode.

```bash
cd cube_wrangler
pip install -e .
```
