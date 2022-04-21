# [Merlin Core](https://github.com/NVIDIA-Merlin/core)

[![PyPI](https://img.shields.io/pypi/v/merlin-core?color=orange&label=version)](https://pypi.python.org/pypi/merlin-core/)
[![LICENSE](https://img.shields.io/github/license/NVIDIA-Merlin/core)](https://github.com/NVIDIA-Merlin/merlin-core/blob/main/LICENSE)
[![Documentation](https://img.shields.io/badge/documentation-blue.svg)](https://nvidia-merlin.github.io/core/main/)

The Merlin Core library provides the core utilities for [NVIDIA Merlin](https://github.com/NVIDIA-Merlin) libraries
like [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular), [Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec)
and [Merlin Models](https://github.com/NVIDIA-Merlin/models).
For example, the [merlin.io.Dataset](https://nvidia-merlin.github.io/core/main/api/merlin.io.html#merlin.io.Dataset) and [merin.schema.Schema](https://nvidia-merlin.github.io/core/main/api/merlin.schema.html#merlin.schema.Schema) classes are fundamental for working with data and building recommender systems with Merlin.

## Installation

### Installing Merlin Core Using Pip

```shell
pip install merlin-core
```

### Installing Merlin Core Using Conda

```shell
conda install -c nvidia -c rapidsai -c numba -c conda-forge merlin-core python=3.7 cudatoolkit=11.2
```

### Running Merlin Core with Docker

As a fundamental library for Merlin, Merlin Core is included in the Merlin Containers.

Refer to the [Merlin Containers](https://nvidia-merlin.github.io/Merlin/main/containers.html) documentation page for information about the Merlin container names, URLs to the container images on the NVIDIA GPU Cloud catalog, and key Merlin components.

## Feedback and Support

To report bugs or get help, please open an issue on the [GitHub repo](https://github.com/NVIDIA-Merlin/core/issues).
