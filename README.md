<p align="left">
  <img src="doc/logo.png" width="30%">
</p>

# Infinity API Tools

`infinity_tools` contains modules for interacting with the [Infinity API](https://medium.com/infinity-ai/infinity-api-82149d21c87c) by [Infinity AI, Inc](https://toinfinity.ai/).

The Infinity API provides synthetic data to help ML teams build better, debug faster, and prototype at the click of a button.

# Overview

Multiple synthetic data generators are available through the Infinity API. Each generator has a distinct set of input parameters available through a RESTful API. `infinity_tools` contains wrapper libraries to simplify use of each generator and make common workflows ergonomic. There are also data science, visualization, and machine learning submodules available to further support workflows canonical for a given generator.

Currently, `infinity_tools` supports the following generator APIs:

- [VisionFit](infinity_tools/visionfit/): an exercise/fitness generator targeting computer vision applications.
- [SenseFit](infinity_tools/sensefit/): an exercise/fitness generator targeting IMU and joint angle time-series applications.

The [common](infinity_tools/common/) submodule contains data structures and functionality common across generator-specific support libraries.

# Installation

You can use either a Python virtual environment or Conda to install this package.
If you are using an ARM-based Mac (e.g. M1), please use conda, as noted in the instructions
below.

## OS X (Intel) or Linux
- Create a virtualenv or fresh Conda environment 
    - Using Conda:
        - `conda create --name infinity python=3.9`
        - `conda activate infinity`
    - Using virtualenv
        - Make sure you have python 3.8, 3.9, or 3.10
        - Create new environment
            - `python -m venv env`
        - Activate your virtual environment
            - `source env/bin/activate`
- Make sure you have the latest pip
    - `python -m pip install --upgrade pip`
- Clone the repo 
    - `git clone git@github.com:toinfinityai/infinity-tools.git`
- Install the package into your environment
    - `pip install -e PATH_TO_INFINITY_TOOLS`

## OS X (ARM-based such as M1)
- Ensure you are using conda with the [conda-forge](https://conda-forge.org/) channel. You can check that this is the case with `conda config --show channels`. We recommend using `miniforge` for your conda installation (instructions [here](https://github.com/conda-forge/miniforge)), which sets `conda-forge` as the default channel.
- Create a fresh Conda env 
    - Using conda:
        - `conda create --name infinity python=3.9`
        - `conda activate infinity`
- Make sure you have the latest pip
    - `python -m pip install --upgrade pip`
- Clone the repo 
    - `git clone git@github.com:toinfinityai/infinity-tools.git`
- Edit `pyproject.toml` in the `infinity-tools` repo, changing `tensorflow` to 
  `tensorflow-macos`
- Install the package into your environment
    - `pip install -e PATH_TO_INFINITY_TOOLS`
- If you received an error about `h5py`, please install it with conda
    - `conda install h5py`
- Repeat installing the package
    - `pip install -e PATH_TO_INFINITY_TOOLS`

# Swagger UI

[<img alt="Swagger UI" height="50px" src="https://static1.smartbear.co/swagger/media/assets/images/swagger_logo.svg" />](https://api.toinfinity.ai/api/schema/swagger-ui/)

A Swagger UI is available for the Infinity API [here](https://api.toinfinity.ai/api/schema/swagger-ui/). The Swagger UI provides a web-based interface to perform low-level interactions with the REST endpoint through the browser. A valid authentication token is required.

# Infinity API Tutorial Notebooks

The [Infinity API Tutorials](https://github.com/toinfinityai/infinity-tutorials) repo provides extensive notebook examples that make use of the `infinity_tools` libary. It also provides detailed documentation on API inputs and outputs, including the labels provided with synthetic data.

# Contact
The Infinity API is brought to you by [Infinity AI](https://toinfinity.ai/). We're a small team of dedicated engineers who specialize in generating custom synthetic datasets and built these API hoping they would be useful to people like you! Drop us a line at [info@toinfinity.ai](mailto:info@toinfinity.ai).
