from pathlib import Path
import pickle
import numpy as np
from datasets import load_dataset

base_dir = Path(__file__).parent
train_file_path = base_dir / 'train.bin'
val_file_path = base_dir / 'val.bin'
meta_file_path = base_dir / 'meta.pkl'

dataset = load_dataset('roneneldan/TinyStories')
train_size = len(dataset['train']['text'])
val_size = len(dataset['validation']['text'])
dataset_size = train_size + val_size

subset_perc = 5 // 10

train_data = ' '.join(dataset['train']['text'][:train_size * subset_perc])
val_data = ' '.join(dataset['validation']['text'][:val_size * subset_perc])

print(f"Training dataset has {len(train_data)} characters")
print(f"Validation dataset has {len(val_data)} characters")

chars = sorted(set(train_data + val_data))
vocab_size = len(chars)
print("Vocabulary consist of: " + ''.join(chars))
print(f"There are {vocab_size} unique characters")

stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda input_string: list(map(stoi.get, input_string))
decode = lambda encoded_ints: ''.join(map(itos.get, encoded_ints))

train_ids = encode(train_data)
val_ids = encode(val_data)
print(f"The training data has {len(train_ids)} tokens")
print(f"The validation data has {len(val_ids)} tokens")

np.array(train_ids, dtype=np.uint16).tofile(train_file_path)
np.array(val_ids, dtype=np.uint16).tofile(val_file_path)

meta = {
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}

with meta_file_path.open('wb') as f:
    pickle.dump(meta, f)