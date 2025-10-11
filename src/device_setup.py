import os
from transformer_lens.utils import get_device

device = get_device()
if device == 'cuda':
    # replace 0 with available gpus
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from huggingface_hub import login
login(token="") # replace your token
