import torch
from transformer_lens.utils import get_device

device = get_device()
if device == 'cuda':
    num_gpus = torch.cuda.device_count()

from huggingface_hub import login
login(token="") # replace your token
