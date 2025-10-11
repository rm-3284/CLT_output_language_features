import matplotlib.pyplot as plt
import os
import pandas as pd
from transformer_lens import HookedTransformer
import torch

from template import base_strings, langs_big, lang_dict

def get_logits(input: str, model: HookedTransformer):
    res = model(input, return_type='logits')
    logits = res.logits[0][-1].detach().clone()
    return logits

def code_switching_histogram(
        dicts: list[tuple[dict[str, str], dict[str, list[str]]]], 
        model: HookedTransformer, 
        langs: set[str] = langs_big, 
        base_strings: dict[str, str] = base_strings,
        ) -> dict[str, dict[str, dict[str, float]]]:
    result_dict = {}
    for l in langs:
        result_dict[l] = {}
        for l2 in langs:
            result_dict[l][l2] = {'wlca': [], 'clca': [], 'wlsa': [], 'clsa': [], 'enca': [], 'ensa': [], 'wrong': []}

    for adj_dict, ans_dict in dicts:
        for l in langs:
            for l2 in langs:
                prompt = base_strings[l].format(adj=adj_dict[l2])
                probability = torch.softmax(get_logits(prompt, model), dim=-1)

                rest = 1
                tmp = []
                for ans in ans_dict[l]:
                    token_idx = model.to_tokens(ans)[1]
                    tmp.append(probability[token_idx])
                result_dict[l][l2]['wlca'].append(sum(tmp))
                rest -= sum(tmp)

                tmp = []
                for ans in ans_dict[l2]:
                    token_idx = model.to_tokens(ans)[1]
                    tmp.append(probability[token_idx])
                result_dict[l][l2]['clca'].append(sum(tmp))
                if l != l2:
                    rest -= sum(tmp)

                tmp = []
                for ans in ans_dict['en']:
                    token_idx = model.to_tokens(ans)[1]
                    tmp.append(probability[token_idx])
                result_dict[l][l2]['enca'].append(sum(tmp))
                if l != 'en' and l2 != 'en':
                    rest -= sum(tmp)

                token_idx = model.to_tokens(adj_dict[l])[1]
                result_dict[l][l2]['wlsa'].append(probability[token_idx])
                rest -= probability[token_idx]

                token_idx = model.to_tokens(adj_dict[l2])[1]
                result_dict[l][l2]['clsa'].append(probability[token_idx])
                if l != l2:
                    rest -= probability[token_idx]

                token_idx = model.to_tokens(adj_dict['en'])[1]
                result_dict[l][l2]['ensa'].append(probability[token_idx])
                if l != 'en' and l2 != 'en':
                    rest -= probability[token_idx]

                result_dict[l][l2]['wrong'].append(rest)
    
    avg_dict = {}
    for l in langs:
        avg_dict[l] = {}
        for l2 in langs:
            avg_dict[l][l2] = {}
    for l in langs:
        for l2 in langs:
            for key in result_dict[l][l2].keys():
                lst_tensors = result_dict[l][l2][key]
                stack = torch.stack(lst_tensors, dim=0)
                overall_mean = torch.mean(stack)
                avg_dict[l][l2][key] = overall_mean.cpu().detach().float().numpy().item()

    return avg_dict

def visualize_histogram(
        data_dict: dict[str, dict[str, dict[str, float]]], 
        langs: set[str] = langs_big,
        lang_dict = lang_dict, interactive = True, png_path: str = "") -> None:
    for l in langs:
        prompt_language = lang_dict['en'][l]
        print(prompt_language)
        df = pd.DataFrame(data_dict[l])
        pd.set_option('display.float_format', lambda x: '%.5f' % x)
        print(df)
        # Transpose the DataFrame to have outer keys as columns and inner keys as index
        df_transposed = df.transpose()

        ax = df_transposed.plot.bar(rot=0, figsize=(10, 6))

        # Add labels and title for clarity
        ax.set_xlabel("Main Category")
        ax.set_ylabel("Value")
        ax.set_title(f"Prompt language {prompt_language}")
        plt.legend(title="Sub Category")
        plt.tight_layout()
        if interactive:
            plt.show()
        else:
            plt_name = "prompt_{prompt_language}.png"
            plt_path = os.path.join(png_path, plt_name)
            plt.savefig(plt_path)
            plt.close()


if __name__ == "__main__":
    from device_setup import device
    model = HookedTransformer.from_pretrained("gemma-2-2b", device=device)

    from data.adjectives.adjectives import small_data
    avg_dict = code_switching_histogram(small_data, model)

    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    output_directory = os.path.join(absolute_directory, "figures/confirming_behavior")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)
    
    visualize_histogram(avg_dict, interactive=False, png_path=output_directory)
