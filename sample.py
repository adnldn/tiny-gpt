import pickle
from pathlib import Path
import importlib
from contextlib import nullcontext
import torch
from model import GPTConfig, GPT
from config_parser import parse_config, override_globals


class ModelRunner:
    def __init__(self, config):
        self.config = config
        self.init_from = self.config['init_from']
        self.out_dir = self.config['out_dir']
        self.start = self.config['start']
        self.num_samples = self.config['num_samples']
        self.max_new_tokens = self.config['max_new_tokens']
        self.temperature = self.config['temperature']
        self.top_k = self.config['top_k']
        self.device = self.config['device']
        self.compile = self.config['compile']
        self.seed = self.config['seed']

        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(5)
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

        ptdtype = getattr(torch, self.config['dtype'], torch.float32)
        self.ctx = torch.amp.autocast(device_type=self.config['device'], dtype=ptdtype) if self.config['device'] == 'cuda' else nullcontext()

        self.model, self.checkpoint = self.load_model()
        self.encode, self.decode = self.load_encoder_decoder()
        self.x = self.prepare_input()

    def prepare_input(self):
        start = self.config.get('start')
        if start.startswith('FILE:'):
            with open(start[5:], 'r', encoding='utf-8') as f:
                start = f.read() # [-self.model's_context_length['']:]
        start_ids = self.encode(start) 
        return torch.tensor(start_ids, dtype=torch.long, device=self.config['device'])[None]

    def load_model(self):
        checkpoint = None
        if self.init_from == 'resume':
            checkpoint_path = Path(self.out_dir) / 'checkpoint.pt'
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            gptconf = GPTConfig(**checkpoint['model_args'])
            model = GPT(gptconf)
            state_dict = checkpoint['model']
            compile_prefix = '_orig_mod.' #  added when model is torch.compile()'d; removed to load into non-compiled model
            state_dict = {k[len(compile_prefix):] if k.startswith(compile_prefix) 
                            else k: v for k, v in checkpoint['model'].items()}
            model.load_state_dict(state_dict)
        elif self.init_from == 'scratch': #  get's checkpoints model args but not the state_dict
            checkpoint_path = Path(self.out_dir) / 'checkpoint.pt'
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
            gptconf = GPTConfig(**checkpoint['model_args'])
            model = GPT(gptconf)
        model.eval()
        model.to(self.device)
        if self.compile:
            model = torch.compile(model)
        return model, checkpoint

    def load_encoder_decoder(self):
        tokeniser_path = Path('data') / self.checkpoint['config']['dataset_name'] / 'tokeniser.pkl'
        with open(tokeniser_path, 'rb') as f:
            tokeniser = pickle.load(f)
        return tokeniser.encode, tokeniser.decode

    def generate_samples(self):
        with torch.no_grad():
            with self.ctx:
                for _ in range(self.num_samples):
                    yield self.model.generate(self.x, self.max_new_tokens, temperature=self.temperature, top_k=self.top_k)

    def run(self):
        for sample in self.generate_samples():
            print(f"{self.decode(sample[0].tolist())}\n----------------------------------------")


if __name__ == "__main__":
    init_from = 'resume'
    dataset_name = 'tiny_stories'
    out_dir = 'out'
    start = '<SOS>' # or 'FILE:/path/to/desired/file.txt'
    num_samples = 10
    max_new_tokens = 500
    temperature = 0.8
    top_k = 200
    seed = 1337
    device = 'cpu'
    dtype = 'bfloat16' #if device == 'cuda' and torch.cuda.is_available and torch.cuda.is_bf16_supported() else 'float16'
    compile = False

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

    config = {k: v for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str, torch.device))}

    module = importlib.import_module(f"data.{dataset_name}.prepare")
    SimpleEncoding = module.SimpleEncoding
    BPETokeniserWrapper = module.BPETokeniserWrapper

    runner = ModelRunner(config)
    runner.run()