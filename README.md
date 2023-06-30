# A two-tier deep learning training pipeline for detecting *Helicobacter pylori* (HP) gastritis on histologic slides

This repository provides scripts to reproduce the results in the paper "A two-tiered deep learning-based model for histologic diagnosis of Helicobacter gastritis".

## Publications

(Under review.)

## License

Copyright (C) 2023 aetherAI Co., Ltd. All rights reserved. Licensed under the CC BY-NC-SA 4.0 license (https://creativecommons.org/licenses/by-nc-sa/4.0/legalcode).

## Requirements

### Hardware Requirements

Make sure the system contains adequate amount of main memory space (minimal: 32 GB) to prevent out-of-memory error.

### Software Stacks

Although Poetry can set up most Python packages automatically, you should install the following native libraries manually in advance.

- CUDA 11.3+

CUDA is essential for PyTorch to enable GPU-accelerated deep neural network training. See https://docs.nvidia.com/cuda/cuda-installation-guide-linux/ .

- OpenMPI 3+

OpenMPI is required for multi-GPU distributed training. If `sudo` is available, you can simply install this by,
```
sudo apt install libopenmpi-dev
```

- Python 3.9+

The development kit should be installed.
```
sudo apt install python3.9-dev
```

- OpenSlide

OpenSlide is a library to read slides. See the installation guide in https://github.com/openslide/openslide .

### Python Packages

We use Poetry to manage Python packages. The environment can be automatically set up by,
```
cd [project folder]
python3.9 -m pip install poetry
poetry install
poetry run poe install
```

## Usage

### Whole-slide model (HMS)

Here, we take 5x-magnified input for example.

Train a model:
```
CUDA_VISIBLE_DEVICES=[4 GPUs] mpirun --bind-to none -np 4 poetry run python -m hms2.pipeline.train --config configs/hms/config_5x.yaml
```

Test the model to generate slide-level predictions:
```
CUDA_VISIBLE_DEVICES=[4 GPUs] mpirun --bind-to none -np 4 poetry run python -m hms2.pipeline.test --config configs/hms/config_5x.yaml
```

Generate heatmaps by CAM:
```
CUDA_VISIBLE_DEVICES=[4 GPUs] mpirun --bind-to none -np 4 poetry run python -m hms2.pipeline.visualize --config configs/hms/config_5x.yaml
```
