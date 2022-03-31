# [merlin-core](https://github.com/NVIDIA-Merlin/core)

[![Documentation](https://img.shields.io/badge/documentation-blue.svg)](https://nvidia-merlin.github.io/core/main/)
[![PyPI](https://img.shields.io/pypi/v/merlin-core?color=orange&label=version)](https://pypi.python.org/pypi/merlin-core/)
[![LICENSE](https://img.shields.io/github/license/NVIDIA-Merlin/core)](https://github.com/NVIDIA-Merlin/merlin-core/blob/main/LICENSE)

This repository contains core utilities for [NVIDIA Merlin](https://github.com/NVIDIA-Merlin) libraries
like [NVTabular](https://github.com/NVIDIA-Merlin/NVTabular), [Transformers4Rec](https://github.com/NVIDIA-Merlin/Transformers4Rec)
and [merlin-models](https://github.com/NVIDIA-Merlin/models). This includes the [merlin.io.Dataset](https://nvidia-merlin.github.io/core/main/api/merlin.io.html#merlin.io.Dataset) and [merin.schema.Schema](https://nvidia-merlin.github.io/core/main/api/merlin.schema.html#merlin.schema.Schema) classes.

## Installation

### Installing merlin-core using Pip

```
pip install merlin-core
```

### Installing merlin-core using Conda

```
conda install -c nvidia -c rapidsai -c numba -c conda-forge merlin-core python=3.7 cudatoolkit=11.2
```

### Running merlin-core with Docker

NVTabular Docker containers are available in the [NVIDIA Merlin container repository](https://catalog.ngc.nvidia.com/?filters=&orderBy=scoreDESC&query=merlin). There are six different containers:

| Container Name             | Container Location | Functionality |
| -------------------------- | ------------------ | ------------- |
| merlin-tensorflow-inference           |https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-tensorflow-inference      | NVTabular, Tensorflow and Triton Inference |
| merlin-pytorch-inference           |https://catalog.ngc.nvidia.com/orgs/nvidia/teams/merlin/containers/merlin-pytorch-inference           | NVTabular, PyTorch, and Triton Inference |
| merlin-inference           | https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-inference           | NVTabular, HugeCTR, and Triton Inference |
| merlin-training            | https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-training            | NVTabular and HugeCTR                    |
| merlin-tensorflow-training | https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-tensorflow-training | NVTabular, TensorFlow, and HugeCTR Tensorflow Embedding plugin |
| merlin-pytorch-training    | https://ngc.nvidia.com/catalog/containers/nvidia:merlin:merlin-pytorch-training    | NVTabular and PyTorch                    |

To use these Docker containers, you'll first need to install the [NVIDIA Container Toolkit](https://github.com/NVIDIA/nvidia-docker) to provide GPU support for Docker. You can use the NGC links referenced in the table above to obtain more information about how to launch and run these containers.

## Feedback and Support

To report bugs or get help, please open an issue on the [GitHub repo](https://github.com/NVIDIA-Merlin/core/issues).
