# Tiny-GPT: A From-Scratch Implementation of GPT-2
This repository contains a from-scratch implementation of GPT-2, created as an educational exercise. The model is currently trained on the TinyStories dataset, which is loaded from the HuggingFace datasets library. However, you can easily use a different dataset by modifying `prepare.py`. TinyStories is a synthetic dataset of short stories generated by GPT-3.5/4, containing words that 3-4-year-olds would typically understand. The dataset is structured as a dictionary with keys for the ‘train’ and ‘validation’ splits. The value associated with each key is a list of stories, where each story is a string of text. This format is typical of many datasets on Hugging Face. Hence, the dataloaders should not require much, if any, alteration when swapping with another dataset from Hugging Face's datasets library.

## Configuration

The `train.py` file sets default hyperparameters and variables, and it can be run without any additional arguments. To override these variables, you can pass a JSON configuration file to `train.py` using the `--config_file your_config.json` flag. Additionally, you can override individual variables using the pattern `--variable_name value`. For example, `--device cuda`. Please note that arguments passed individually take precedence over those specified in the `--config_file`.


## Setup

### Data Preparation
To prepare the data and tokenisers, follow these steps:

1. Run the command:
```bash
python data/tiny_stories/prepare.py
```
This command downloads the TinyStories dataset (if not already cached) and creates objects that only hold information on how to access the data in memory. This means you shouldn't run into memory issues. If you want to use a smaller portion of the dataset, you can adjust the `data_percentage` variable within `prepare.py` (it's set to 100 by default).

2. Choose your tokeniser. By default, a simple character level encoding is used. If you prefer to use a BPE tokeniser, you can do so by adding the flag `--encoding_type bpe` to the command. For example: 
```bash
python data/tiny_stories/prepare.py --encoding_type bpe
```
The BPE tokeniser will be trained on the same subset of the dataset selected by `data_percentage`, producing a vocabulary of about 30,000 for the full dataset. The tokeniser will be saved as a `tokeniser.pkl` file in the `data/tiny_stories/` directory. The subsequent files will automatically use the correct encoders and decoders, so you don't need to specify them again.

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