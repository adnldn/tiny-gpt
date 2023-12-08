import numpy as np
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
from tokenizers.pre_tokenizers import Whitespace
import pickle


class SimpleEncoding:
    def __init__(self, stoi, itos):
        self.stoi = stoi
        self.itos = itos

    def get_vocab_size(self):
        return len(self.itos)
    
    def encode(self, text):
        return [self.stoi.get(ch, self.stoi['<UNK>']) for ch in text]
    
    def decode(self, token_ids):
        return ''.join([self.itos[i] for i in token_ids])
    

class BPETokeniserWrapper:
    def __init__(self, tokeniser):
        self.tokeniser = tokeniser
    
    def get_vocab_size(self):
        return self.tokeniser.get_vocab_size()
    
    def encode(self, text):
        return self.tokeniser.encode(text).ids
    
    def decode(self, token_ids):
        return self.tokeniser.decode(token_ids)

dataset = load_dataset('roneneldan/TinyStories')

train_size = len(dataset['train'])
val_size = len(dataset['validation'])
dataset_size = train_size + val_size

data_percentage = 100

sub_train_size = int(len(dataset['train']) * (data_percentage / 100))
sub_val_size = int(len(dataset['validation']) * (data_percentage / 100))
train_indices = np.random.choice(train_size, sub_train_size, replace=False).tolist()
val_indices = np.random.choice(val_size, sub_val_size, replace=False).tolist()

train_subset = dataset['train'].select(train_indices)
val_subset = dataset['validation'].select(val_indices)

if data_percentage < 100:
    dataset = {'train': train_subset, 'validation': val_subset}

if __name__ == '__main__':
    encoding_type = 'simple'
    if encoding_type == 'simple':
        # simple character level encoding
        chars = [chr(i) for i in range(256)]
        stoi = {ch:i for i, ch in enumerate(chars)}
        itos = {i:ch for i, ch in enumerate(chars)}
        stoi['<UNK>'] = -1
        itos[-1] = '<UNK>'
        encoding = SimpleEncoding(stoi, itos)
        with open('data/tiny_stories/tokeniser.pkl', 'wb') as f:
            pickle.dump(encoding, f)
    elif encoding_type == 'bpe':
        # BPE
        tokeniser = Tokenizer(BPE())
        tokeniser.pre_tokenizer = Whitespace()
        trainer = BpeTrainer(special_tokens=["<PAD>", "<UNK>", "<SOS>", "<EOS>"])
        train_text = [story for story in dataset['train']['text']]
        val_text = [story for story in dataset['validation']['text']]
        text_data = train_text + val_text
        tokeniser.train_from_iterator(text_data, trainer=trainer)
        encoding = BPETokeniserWrapper(tokeniser)
        with open("data/tiny_stories/tokeniser.pkl", 'wb') as f:
            pickle.dump(encoding, f)

    print(f"There are {encoding.get_vocab_size()} unique characters in the {encoding_type} encoding")