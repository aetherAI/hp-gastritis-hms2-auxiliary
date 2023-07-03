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

Here, we take the config with 5x-magnified input (`configs/hms/config_5x.yaml`) for example, which can be modified for your own dataset.

If needed, modify the datalists: `data/datalists/train.csv`, `data/datalists/val.csv`, and `data/datalists/test.csv` refered by the config. Copy or create softlinks of slide images to `data/slides`.

To train a model:
```
CUDA_VISIBLE_DEVICES=[4 GPUs] mpirun --bind-to none -np 4 poetry run python -m hms2.pipeline.train --config configs/hms/config_5x.yaml
```

To test the model to generate slide-level predictions:
```
CUDA_VISIBLE_DEVICES=[4 GPUs] mpirun --bind-to none -np 4 poetry run python -m hms2.pipeline.test --config configs/hms/config_5x.yaml
```

To generate heatmaps by CAM:
```
CUDA_VISIBLE_DEVICES=[4 GPUs] mpirun --bind-to none -np 4 poetry run python -m hms2.pipeline.visualize --config configs/hms/config_5x.yaml
```

### Auxiliary model

We take the config with 5x-magnified input and logistic regression (`configs/auxiliary/config_5x_logistic.yaml`) for example.

It requires a trained HMS model (`results/result_5x/model.pt`) and an annotated dataset (list: `data/datalists/annotated_train.csv`; masks in .npy: `data/masks/5x`).

To extract embedding features:
```
CUDA_VISIBLE_DEVICES=[1 GPU] poetry run python -m scripts.auxiliary.extract --config configs/auxiliary/config_auxiliary_5x_logistic.yaml
```

To find the optimal hyper-parameters for an auxiliary model and train one:
```
poetry run python -m scripts.auxiliary.train --config configs/auxiliary/config_auxiliary_5x_logistic.yaml
```

To generate heatmaps:
```
CUDA_VISIBLE_DEVICES=[1 GPU] poetry run python -m scripts.auxiliary.viz --config configs/auxiliary/config_auxiliary_5x_logistic.yaml
```
