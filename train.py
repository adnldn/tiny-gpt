# utilities
import random
import time
import pickle
from contextlib import nullcontext
import importlib
from pathlib import Path
# built-ins
import torch
from torch.utils.data import Dataset, DataLoader
from torch.optim.lr_scheduler import LambdaLR, CosineAnnealingLR
# custom modules
from model import GPT, GPTConfig
from config_parser import parse_config, override_globals


class LLM_Dataset(Dataset):
    def __init__(self, data, block_size, tokeniser):
        self.data = data
        self.block_size = block_size
        self.tokeniser = tokeniser

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        story = "<SOS>" + self.data[idx] + "<EOS>"
        encoded_story = self.tokeniser.encode(story)
        next_idx = (idx + 1) % len(self.data)
        while len(encoded_story) < self.block_size + 1 :
            next_story = "<SOS>" + self.data[next_idx] + "<EOS>"
            encoded_next_story = self.tokeniser.encode(next_story)
            encoded_story += encoded_next_story
            next_idx = (next_idx + 1) % len(self.data)
        start_idx = random.randint(0, len(encoded_story) - self.block_size - 1)
        x = encoded_story[start_idx : start_idx + self.block_size]
        y = encoded_story[start_idx + 1 : start_idx + 1 + self.block_size]
        assert len(x) == len(y), "x and y are not the same length"
        return torch.tensor(x, dtype=torch.long), torch.tensor(y, dtype=torch.long)
    

class ModelInitialiser:
    def __init__(self, config):
        self.config = config
        
    def init_model(self):
        run_time_dict = None
        tokeniser_path = Path('data') / self.config['dataset_name'] / 'tokeniser.pkl'
        with open(tokeniser_path, 'rb') as f:
            tokeniser = pickle.load(f)
        vocab_size = tokeniser.get_vocab_size()
        self.config['vocab_size'] = vocab_size
        keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size'] #  default vocab size atm
        model_args = {k: self.config[k] for k in keys}
        print('\n'.join([f'{k}: {v}' for k, v in model_args.items()]))

        if self.config['init_from'] == 'scratch':
            print("Training model from scratch.\n")
            gpt_config = GPTConfig(**model_args)
            model = GPT(gpt_config)
        elif self.config['init_from'] == 'resume':
            # load checkpoint data
            checkpoint_path = Path(self.config['out_dir']) / 'checkpoint.pt'
            checkpoint = torch.load(checkpoint_path, map_location=self.config['device'])
            run_time_dict = {
                'checkpoint': checkpoint,
                'iter_num': checkpoint['iter_num'],
                'best_val_loss': checkpoint['best_val_loss'] 
            }
            keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']
            model_args.update({k: checkpoint['model_args'][k] for k in keys})
            # init model
            gpt_config = GPTConfig(**model_args)
            model = GPT(gpt_config)
            compile_prefix = '_orig_mod.' #  added when model is torch.compile()'d; removed to load into non-compiled model
            state_dict = {k[len(compile_prefix):] if k.startswith(compile_prefix) 
                    else k: v for k, v in checkpoint['model'].items()}
            model.load_state_dict(state_dict)
            print(f"Resuming training from {checkpoint_path} at iteration number {run_time_dict['iter_num']}\n")
        model.to(self.config['device'])

        if self.config['compile']:
            print("Compiling your model. Please wait...")
            # unoptimised_model = model
            model = torch.compile(model)

        return model, run_time_dict, self.config


def configure_optimiser(model, config, run_time_dict):
    if run_time_dict != None:
        checkpoint = run_time_dict['checkpoint']
        del run_time_dict['checkpoint']
    else:
        checkpoint = None
    optimiser = model.configure_optimisers(
        config['weight_decay'], config['learning_rate'],
        (config['beta1'], config['beta2']), config['device_type']
        )
    if checkpoint != None and 'optimiser' in checkpoint:
        optimiser.load_state_dict(checkpoint['optimiser'])

    return optimiser

def setup_schedulers(optimiser, config):
    warmup_lambda = lambda iter: iter/config['warmup_iters'] if iter < config['warmup_iters'] else 1.0 #  outputs lr_coef
    warmup_scheduler = LambdaLR(optimiser, warmup_lambda) #  outputs lr, i.e., lr_coef * base_lr
    decay_scheduler = CosineAnnealingLR(optimiser, T_max=(config['lr_decay_iters'] - config['warmup_iters']),
                                        eta_min=config['min_lr']) #  outputs lr
    min_lambda = lambda iter: config['min_lr'] / config['learning_rate'] #  outputs lr_coef. Therefore, divide by base_lr to set lr = min_lr
    min_scheduler = LambdaLR(optimiser, min_lambda) #  outputs lr. 
    return warmup_scheduler, decay_scheduler, min_scheduler

def load_data(config):
    tokeniser_path = Path('data') / config['dataset_name'] / 'tokeniser.pkl'
    with open(tokeniser_path, 'rb') as f:
        tokeniser = pickle.load(f)

    train_dataset = LLM_Dataset(dataset['train']['text'], config['block_size'], tokeniser)
    val_dataset = LLM_Dataset(dataset['validation']['text'], config['block_size'], tokeniser)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                                shuffle=True, pin_memory=config['pin_memory'])
    val_loader = DataLoader(val_dataset, batch_size=config['batch_size'],
                                shuffle=True, pin_memory=config['pin_memory'])
    return {'train': train_loader, 'val': val_loader}

    
class Trainer:
    def __init__(self, model, optimiser, schedulers, data_loader_dict, config, run_time_dict,):
        self.model = model
        self.optimiser = optimiser
        self.schedulers = schedulers
        self.data_loader_dict = data_loader_dict
        self.config = config
        self.best_val_loss = float('inf') if run_time_dict == None else run_time_dict['best_val_loss']
        self.iter_num = 0 if run_time_dict == None else run_time_dict['iter_num']
        ptdtype = getattr(torch, self.config['dtype'], torch.float32)
        keys = ['n_layer', 'n_head', 'n_embd', 'block_size', 'bias', 'vocab_size']
        self.model_args = {k: self.config[k] for k in keys}
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config['use_amp']) if self.config['use_amp'] else None
        self.ctx = torch.amp.autocast(device_type=self.config['device'], dtype=ptdtype) if self.config['device'] == 'cuda' else nullcontext()

    @torch.no_grad()
    def estimate_loss(self):
        out = {}
        self.model.eval()
        try:
            for split in ['train', 'val']:
                if split not in self.data_loader_dict:
                    print(f"No {split} data in dataloader")
                    continue
                losses = torch.zeros(self.config['eval_iters'])
                data_loader = self.data_loader_dict[split]
                for k, (X, Y) in enumerate(data_loader):
                    if k >= self.config['eval_iters']:
                        break
                    X, Y = X.to(self.config['device']), Y.to(self.config['device'])
                    with self.ctx:
                        logits, loss = self.model(X, Y)
                    losses[k] = loss.item()
                out[split] = losses.mean().item()
        finally:
            self.model.train()
        return out
    
    def save_checkpoint(self):
        out_dir = Path(self.config['out_dir'])
        out_dir.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            'model': self.model.state_dict(),
            'optimiser': self.optimiser.state_dict(),
            'model_args': self.model_args,
            'iter_num': self.iter_num,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        print(f"Saving checkpoint to {out_dir}\n")
        torch.save(checkpoint, out_dir / 'checkpoint.pt')
    
    def train(self):
        t0 = time.time()
        for X, Y in self.data_loader_dict['train']:
            X, Y = X.to(self.config['device']), Y.to(self.config['device'])

            if self.iter_num % self.config['eval_interval'] == 0:
                losses = self.estimate_loss()
                print(f"Step {self.iter_num}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

                # Log metrics
                if self.config['wandb_log']:
                    lr_stage = 0 if self.iter_num < self.config['warmup_iters'] else 1 if self.config['warmup_iters'] <= self.iter_num <= self.config['lr_decay_iters'] else 2
                    wandb.log({
                        'iter': self.iter_num,
                        'train/loss': losses['train'],
                        'val/loss': losses['val'],
                        'lr': self.schedulers[lr_stage].get_last_lr()[0]
                    })

                # Save checkpoint
                if losses['val'] < self.best_val_loss or self.config['always_save_checkpoint']:
                    self.best_val_loss = losses['val']
                    if self.iter_num > 0:
                        self.save_checkpoint()

            if self.iter_num == 0 and self.config['eval_only']:
                break

            for micro_step in range(self.config['gradient_accumulation_steps']):
                with torch.cuda.amp.autocast(enabled=self.config['use_amp']):
                    logits, loss = self.model(X, Y)
                    loss = loss / self.config['gradient_accumulation_steps']
                # backward pass, with gradient scaling if using AMP
                if self.scaler:
                    self.scaler.scale(loss).backward()
                else:
                    loss.backward()
                if self.config['grad_clip'] != 0.0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
                # step the optimiser and scaler if using AMP
                if self.scaler:
                    self.scaler.step(self.optimiser)
                    self.scaler.update()
                else:
                    self.optimiser.step()
                # Update the learning rate
                if self.iter_num < self.config['warmup_iters']:
                    self.schedulers[0].step()
                elif self.config['warmup_iters'] <= self.iter_num <= self.config['lr_decay_iters']:
                    schedulers[1].step()
                else:
                    schedulers[2].step()
                # flush the gradients as soon as possible to free memory
                self.optimiser.zero_grad(set_to_none=True)

                # timing and logging
                t1 = time.time()
                dt = t1 - t0
                t0 = t1

                self.iter_num += 1

            if self.iter_num > self.config['max_iters']:
                break
                

if __name__ == "__main__":
    torch.manual_seed(5432)

    # default config values for Tiny Stories
    out_dir = 'out'
    eval_interval = 200
    log_interval = 10
    eval_iters = 50
    eval_only = False
    always_save_checkpoint = False
    init_from = 'scratch'
    # data
    dataset_name = 'tiny_stories'
    gradient_accumulation_steps = 1
    batch_size = 64
    # model
    block_size = 256
    n_layer = 8
    n_head = 8
    n_embd = 512
    bias = True
    # optimiser
    dropout = 0.2
    learning_rate = 0.001
    max_iters = 5000
    weight_decay = 1e-1
    beta1 = 0.90
    beta2 = 0.99
    grad_clip = 1.0
    # learning rate
    decay_lr = True
    warmup_iters = 100
    lr_decay_iters = 5000
    min_lr = 0.0001
    # system
    device = 'mps'
    dtype = 'bfloat16'
    compile = False
    # wandb logging
    wandb_log = False
    wandb_project = 'tiny-stories'
    wandb_run_name = 'tiny-gpt'

    # overriden from --config_file and CML arguments
    override_globals(parse_config(globals()), globals())

    if device == 'mps':
        if not torch.backends.mps.is_available():
            if not torch.backends.mps.is_built():
                print("PyTorch install was not built with MPS enabled. Using CPU")
            else:
                print("No Apple silicon or MacOS version doesn't support MPS. Using CPU")
            device = 'cpu'
        else:
            print("Using MPS.\n")
        device = torch.device('mps')

    # device may be a string or torch.device instance. device_type is always a string
    if isinstance(device, torch.device):
        device_type = device.type
    else:
        device_type = device

    pin_memory = True if device_type in ['cuda', 'mps'] else False
    use_amp = device == 'cuda' and dtype == 'bfloat16'

    # dataset specific configuration override
    config = {k: v for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, torch.device))}
    print('\n'.join([f'{k}: {v}' for k, v in config.items()]), '\n')

    if wandb_log:
        import wandb
        wandb.init(project=wandb_project, name=wandb_run_name, config=config)

# ------------------------------------------------------------------------------------------ #
    module = importlib.import_module(f"data.{dataset_name}.prepare")
    SimpleEncoding = module.SimpleEncoding
    BPETokeniserWrapper = module.BPETokeniserWrapper
    
    dataset = module.dataset
    
    model_initialiser = ModelInitialiser(config)
    model, run_time_dict, config = model_initialiser.init_model()

    optimiser = configure_optimiser(model, config, run_time_dict)

    schedulers = setup_schedulers(optimiser, config)

    data_loader_dict = load_data(config)

    trainer = Trainer(model, optimiser, schedulers, data_loader_dict, config, run_time_dict)
    trainer.train()
# ------------------------------------------------------------------------------------------ #
