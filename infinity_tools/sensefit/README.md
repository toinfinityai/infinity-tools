<p align="left">
  <img src="../../doc/logo.png" width="30%">
</p>

# SenseFit API

The SenseFit API generates synthetic data for applications at the intersection of on-body **sens**ing (e.g. IMU) and **fit**ness. The API currently allows users to generate simulated time series corresponding to angular position data captured at the wrist (e.g. with a smart watch). Ground-truth labels include continuous, per-frame rep counts.

# SenseFit Client Library

This `sensefit` client library adds higher-level functionality and abstractions on top of the base RESTful Infinity API. This includes ergonomics for interacting with the cloud API and data science/ML tools. The `sensefit` library contains the following submodules:

- [api.py](./api.py): This module contains abstractions and ergonomic functionality built on top of [common.api.py](../common/api.py) specific for the SenseFit API. Some key components include functions to randomly sample input parameters (while specifying any subset exactly), validation of input parameters against constraints, and functions to submit, download, and visualize data from the SenseFit API.
- [datagen.py](./datagen.py): This module defines a `SenseFitGenerator` class that can be used as a data generator when training sequence-based models (e.g. RNNs) on the SenseFit API's outputs. It defines data pre-processing functions specific to the SenseFit API, including data loading and featurization.
- [rnn.py](./rnn.py): This module defines a `SenseFitModel` wrapper class, useful for training Keras models on data generated by the SenseFit API.
- [vis.py](./vis.py): This module defines visualization and summarization functions specific to the SenseFit API.

# SenseFit Tutorial Notebooks

We refer users to the [Infinity API Tutorials](https://github.com/toinfinityai/infinity-tutorials/tree/main/sensefit) repo for extensive notebook examples that make use of this `sensefit` client library.
