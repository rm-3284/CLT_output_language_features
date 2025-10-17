from device_setup import device
import os
from sae_lens import SAE
from template import load_transcoder_from_hub, TranscoderSet, CrossLayerTranscoder
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

"""
The model we use is ReplacementModel defined in the circuit_tracer.
type(ReplacementModel.transcoders) = TranscoderSet | CrossLayerTranscoder
TranscoderSet -->
type(TranscoderSet.transcoders) = dict[int, SingleLayerTranscoder]
SinglaLayerTranscoder.W_dec is the weight decoding matrix
dimensions are (d_transcoder, d_model) where d_model is the dimension of the model's residual stream
CrossLayerTranscoder --> 
CrossLayerTranscoder.W_dec
The dimension is d_transcoder, n_layers - i, d_model for i-th layer (0-index)
because it writes on all the subsequent layers

We could do ablation with AutoModelForCausalLM
"""

def load_gemma(layer):
    release = "gemma-scope-2b-pt-res"
    root_dir = f'/alt/llms/majd/multilingual-llm-features/SAE/{release}/layer_{layer}/width_16k/'
    file_names = list(os.listdir(root_dir))
    file_names.sort(key=lambda x: int(x.split('_')[-1]))
    file_name = file_names[2]
    sae_id = os.path.join(root_dir, file_name).split(f'{release}/')[1]
    sae, cfg_dict, sparsity = SAE.from_pretrained(
        release=release,  # see other options in sae_lens/pretrained_saes.yaml
        sae_id=sae_id,  # won't always be a hook point
        device=device,
        )
    return sae

class Model_Wrapper():
    def __init__(self, model_name='google/gemma-2-2b', transcoder_name='gemma', device=device, dtype=torch.float32):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transcoder, _ = 0, 0#load_transcoder_from_hub(transcoder_name, device=device, dtype=dtype)
        self.hooks = []

    # Ablation hooks for transcoder
    def add_hooks_transcoder_ablation(self, layer, feature_idx):
        if type(self.transcoder) == TranscoderSet:
            singleLayerTranscoder = self.transcoder.transcoders[layer]
            feature_direction = singleLayerTranscoder.W_dec.T[:, feature_idx].clone()
            decoder_weight = singleLayerTranscoder.W_dec.T
        elif type(self.transcoder) == CrossLayerTranscoder:
            # just take the current layer's decoder matrix
            feature_direction = self.transcoder.W_dec[layer].T[:, 0, feature_idx]
            decoder_weight = self.transcoder.W_dec[layer].T[:, 0, :]
        else:
            raise TypeError(f"Transcoder has to be either TranscoderSet or CrossLayerTranscoder but got {type(self.transcoder)}")
        # it will get multiplied twice in the end
        norm = torch.linalg.norm(feature_direction, dim=0) ** 2
        feature_direction = feature_direction / norm

        def change_activation_hook(module, input, output):
            act = output[0]
            clt_acts = act @ decoder_weight
            coefficient = clt_acts[0, :, feature_idx].to(act.device)
            act = (act - coefficient@((feature_direction).T)).to(act.dtype)
            return (act, output[1])
        
        handle = self.model.model.layers[layer].register_forward_hook(change_activation_hook)
        self.hooks.append(handle)
        return
    
    # Amplification hooks for transcoder
    def add_hooks_transcoder_amplification(self, layer, feature_idx, amplification_value):
        if type(self.transcoder) == TranscoderSet:
            singleLayerTranscoder = self.transcoder.transcoders[layer]
            feature_direction = singleLayerTranscoder.W_dec.T[:, feature_idx].clone()
            decoder_weight = singleLayerTranscoder.W_dec.T
        elif type(self.transcoder) == CrossLayerTranscoder:
            # just take the current layer's decoder matrix
            feature_direction = self.transcoder.W_dec[layer].T[:, 0, feature_idx]
            decoder_weight = self.transcoder.W_dec[layer].T[:, 0, :]
        else:
            raise TypeError(f"Transcoder has to be either TranscoderSet or CrossLayerTranscoder but got {type(self.transcoder)}")
        # without normalization, it should be val * direction
        feature_amplification = feature_direction * amplification_value
        norm = torch.linalg.norm(feature_direction, dim=0) ** 2
        feature_direction = feature_direction / norm

        def change_activation_hook(module, input, output):
            # dimension of output is (batch_size, sequence_length, hidden_size = d_model)
            act = output[0]
            clt_acts = act @ decoder_weight
            coefficient = clt_acts[0, :, feature_idx].to(act.device)
            act = (act - coefficient@((feature_direction).T) + (feature_amplification).T).to(act.dtype)
            return (act, output[1])
        
        handle = self.model.model.layers[layer].register_forward_hook(change_activation_hook)
        self.hooks.append(handle)
        return
    
    # In the original function, they have start_idx and topk_feature_num
    # They choose the k most prominent features starting from start_idx-th most prominent feature
    # I will set start_idx = 0, topk_feature_num = 2 to have the strongest ablation
    def add_hooks_sae_ablation(self, target_layer, lang: str, model = 'gemma-2-2b'):
        lan_list = ['en', 'es', 'fr', 'ja', 'ko', 'pt', 'th', 'vi', 'zh', 'ar']
        ori_lan = lan_list.index(lang)
        if ori_lan == -1:
            raise KeyError(f'lang {lang} is not in the available language list')
        start_idx = 0
        topk_feature_num = 2

        file_dir = f'/export/home/rmitsuhashi/multilingual-llm-features/SAE/sae_acts/{model}/layer_{target_layer}/'
        top_index_per_lan = torch.load(os.path.join(file_dir, 'top_index_per_lan_magnitude.pth'), weights_only=True)
        top_index_per_lan = top_index_per_lan[:, start_idx:start_idx+topk_feature_num]
        sae = load_gemma(target_layer)
        ori_lan_idx = top_index_per_lan[ori_lan]
        if 'Llama' in model:
            ori_feature_direction = sae.decoder.weight[:, ori_lan_idx].clone()
        else:
            ori_feature_direction = sae.W_dec.T[:, ori_lan_idx]
        norm = torch.norm(ori_feature_direction, dim=0)**2
        ori_feature_direction = ori_feature_direction / norm

        def change_activation_hook(module, input, output):
            act = output[0]
            if 'Llama' in model:
                sae_acts = act.to(torch.bfloat16) @ sae.decoder.weight
            else:
                sae_acts = act.to(torch.float32) @ sae.W_dec.T
            coefficient = sae_acts[0, :, ori_lan_idx].to(act.device)
            act = (act-coefficient@((ori_feature_direction).T)).to(act.dtype)

            return (act, output[1])

        handle = self.model.model.layers[target_layer].register_forward_hook(change_activation_hook)
        self.hooks.append(handle)
        return
    
    def remove_all_hook(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

def run_without_hooks(model: Model_Wrapper, prompt: str) -> torch.Tensor:
    inputs = model.tokenizer(prompt, return_tensor="pt")
    with torch.no_grad():
        output = model.model(**inputs)
    logits = output.logits # (batch, seq len, vocab)
    return logits

def run_with_clt_hooks(model: Model_Wrapper, prompt: str, ablation: list[str], amplification: list[tuple[str, float]]) -> torch.Tensor:
    inputs = model.tokenizer(prompt, return_tensors="pt")

    for feature in ablation:
        layer, feature_idx = feature.split('.')
        layer = int(layer)
        feature_idx = int(feature_idx)
        model.add_hooks_transcoder_ablation(layer, feature_idx)
    for feature, amplification_value in amplification:
        layer, feature_idx = feature.split('.')
        layer = int(layer)
        feature_idx = int(feature_idx)
        model.add_hooks_transcoder_amplification(layer, feature_idx, amplification_value)

    with torch.no_grad():
        output = model.model(**inputs)
    logits = output.logits # (batch, seq len, vocab)

    model.remove_all_hook()
    return logits

def run_with_sae_hooks(model: Model_Wrapper, prompt: str, ablation_lang: str) -> torch.Tensor:
    inputs = model.tokenizer(prompt, return_tensor="pt")
    # let us ablate in the last layer
    model.add_hooks_sae_ablation(model.model.config.num_hidden_layers-1, ablation_lang)
    with torch.no_grad():
        output = model.model(**inputs)
    logits = output.logits # (batch, seq len, vocab)

    model.remove_all_hook()
    return logits

def get_best_word(logits: torch.Tensor, words: list[str], model: Model_Wrapper) -> tuple[str, int]:
    last_logits = logits.squeeze(0)[-1]
    _, indices = torch.sort(last_logits, dim=-1, descending=True)
    ranks = []
    for word in words:
        token = model.tokenizer(word)['input_ids'][1]
        mask = (indices == token)
        rank = torch.argmax(mask.int(), dim=-1)
        rank = rank.item() if isinstance(rank, torch.Tensor) else rank
        ranks.append((word, rank))
    ranks.sort(key=lambda x: x[1], reverse=False)
    # word, rank
    return ranks[0]

def get_top_outputs(logits: torch.Tensor, model: Model_Wrapper, k: int=10):
    top_probs, top_token_ids = logits.squeeze(0)[-1].softmax(-1).topk(k)
    top_tokens = [model.tokenizer.decode(token_id) for token_id in top_token_ids]
    top_outputs = list(zip(top_tokens, top_probs.tolist()))
    return top_outputs

def logit_diff(logits: torch.Tensor, target: str, source: str, model: Model_Wrapper):
    l = logits.squeeze(0)[-1]
    s = model.tokenizer(source)['input_ids'][1]
    t = model.tokenizer(target)['input_ids'][1]

    diff = l[t] - l[s]
    diff = diff.item() if isinstance(diff, torch.Tensor) else diff

    return diff

def check_valid_meaning(prompt: str, ans: dict[str, list[str]], model: Model_Wrapper, k: int=10) -> bool:
    logits = run_without_hooks

    _, en = get_best_word(logits, ans['en'], model)
    _, zh = get_best_word(logits, ans['zh'], model)
    return en <= k or zh <= k
