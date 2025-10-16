from device_setup import device
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

class Model_Wrapper():
    def __init__(self, model_name='google/gemma-2-2b', transcoder_name='gemma', device=device, dtype=torch.float32):
        self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device, torch_dtype=dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.transcoder, _ = load_transcoder_from_hub(transcoder_name, device=device, dtype=dtype)
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
    def add_hoooks_transcoder_amplification(self, layer, feature_idx, amplification_value):
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
    
    def remove_all_hook(self):
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    
def run_with_hooks(model: Model_Wrapper, prompt: str, ablation: list[str], amplification: list[tuple[str, float]]):
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
        model.add_hoooks_transcoder_amplification(layer, feature_idx, amplification_value)

    with torch.no_grad():
        output = model.model(**inputs)
    logits = output.logits # (batch, seq len, vocab)

    model.remove_all_hook()
    return logits
