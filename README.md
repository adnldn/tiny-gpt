# Tiny-GPT: A From-Scratch Implementation of GPT-2

This repository contains a from-scratch implementation of GPT-2, created as an educational exercise. The model is currently trained on the TinyStories dataset, but you can easily use a different dataset by modifying `prepare.py`.

## Configuration

The `train.py` file sets default hyperparameters and variables, and can be run without any additional arguments. To override these variables, you can pass `train.py` a JSON configuration file using the `--config_file your_config.json` flag. Additionally, you can override individual variables using the pattern `--variable_name value`. For example, `--device cuda`. Please note that arguments passed in individually take precedence over those given in the `--config_file`.

## Setup

### Data Preparation

The first step is to prepare the data. By default, the TinyStories dataset is used. Run the following command to prepare the data:

```bash
python data/tiny_stories/prepare.py
```

This command creates `meta.pkl`, `train.bin`, and `val.bin` in the `data/tiny_stories/` directory. If the TinyStories dataset is too large, you can set the percentage of the dataset to use by adjusting the `subset_percentage` variable within `prepare.py` (which is set to 100 by default).

### Model Training

Next, train the model using the following command:

```bash
python train.py --config_file config/tiny_stories.py
```

This script supports CPU, CUDA, and MPS devices. The default device is CPU, but you can change this by specifying the `--device` flag. For example, to use CUDA, run:

```bash
python train.py --config_file config/tiny_stories --device cuda
```

Please note that `torch.compile` currently does not work with MPS.

As the model trains, it saves checkpoints to the `out-tiny-stories` directory.

## Generating Samples

Once you have trained the model, you can generate samples from it using the following command:

```bash
python sample.py --out_dir out-tiny-stories
```