from intervention import (
    visualize_bar_2ddict_outer_inter,
    bar_graph_visualize,
    create_histogram_0_to_10,
    create_multi_series_histogram
)
from steering import (
    Model_Wrapper, 
    run_with_sae_hooks, 
    run_without_hooks,
    get_best_word, 
    get_top_outputs, 
    logit_diff, 
    check_valid_meaning
    )
from template import base_strings
from data.adjectives import big_data

if __name__ == "__main__":
    import os
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    plt_dir = os.path.join(absolute_directory, "plot/sae_steering")
    if not os.path.exists(plt_dir):
        os.makedirs(plt_dir)

    langs = ['en', 'fr', 'ja', 'zh'] # no german

    model = Model_Wrapper()

    logit_dict = dict()
    rank_dict = dict()
    for lang in langs:
        logit_dict[lang] = dict()
        rank_dict[lang] = dict()
        for ans_lang in langs:
            logit_dict[lang][ans_lang] = list()
            rank_dict[lang][ans_lang] = list()
    
    prompt_zh = base_strings['zh']
    # original logit diff
    for adj, ans in big_data:
        for lang in langs:
            prompt = prompt_zh.format(adj=adj[lang])
            logits = run_without_hooks(model, prompt)
            answer, _ = get_best_word(logits, ans[lang], model)
            for ans_lang in langs:
                word, rank = get_best_word(logits, ans[ans_lang], model)
                logit_dif = logit_diff(logits, word, answer, model)
                logit_dict[lang][ans_lang].append(logit_dif)
                rank_dict[lang][ans_lang].append(rank)
    # average out
    for key, val in logit_dict.items():
        for k, v in val.items():
            logit_dict[key][k] = sum(v) / len(v)
    visualize_bar_2ddict_outer_inter(logit_dict, False, os.path.join(plt_dir, 'old_logits_zh'))
    for lang in langs:
        create_multi_series_histogram(rank_dict[lang], interactive=False, plt_path=plt_dir, file_name=f'original_ranks_zh_{lang}')

    # ablation
    for lang in langs:
        logit_dict = dict()
        rank_dict = dict()
        for ablation_lang in langs:
            logit_dict[ablation_lang] = dict()
            rank_dict[ablation_lang] = dict()
            for ans_lang in langs:
                logit_dict[ablation_lang][ans_lang] = list()
                rank_dict[ablation_lang][ans_lang] = list()
        for adj, ans in big_data:
            for ablation_lang in langs:
                prompt = prompt_zh.format(adj=adj[ablation_lang])
                logits = run_with_sae_hooks(model, prompt, ablation_lang)
                answer, _ = get_best_word(logits, ans[lang], model)
                for ans_lang in langs:
                    word, rank = get_best_word(logits, ans[ans_lang], model)
                    logit_dif = logit_diff(logits, word, answer, model)
                    logit_dict[lang][ans_lang].append(logit_dif)
                    rank_dict[lang][ans_lang].append(rank)
        for key, val in logit_dict.items():
            for k, v in val.items():
                logit_dict[key][k] = sum(v) / len(v)
                logit_dict[key] = sum(val) / len(val)
        visualize_bar_2ddict_outer_inter(logit_dict, False, os.path.join(plt_dir, 'new_logits_zh_' + lang))
        for ablation_lang in langs:
            create_multi_series_histogram(rank_dict[ablation_lang], interactive=False, plt_path=plt_dir, file_name=f'new_ranks_zh_{lang}_{ablation_lang}')
