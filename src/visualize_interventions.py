import json
import os
import numpy as np
import pandas as pd
import seaborn as sns
import statistics
import matplotlib.pyplot as plt
from typing import Dict, Any, List
import warnings

from template import lang_to_flores_key

warnings.filterwarnings("ignore", message="use_inf_as_na")

# Re-using the transformation function from the previous step (it's essential)
def transform_dict_to_dataframe(data_dict: Dict[str, Dict[str, Dict[str, Dict[str, float]]]]) -> pd.DataFrame:
    records = []
    for experiment, methods in data_dict.items():
        for method, scores in methods.items():
            for run_name, score in scores.items():
                records.append({
                    'Experiment': experiment,
                    'Method': method,
                    'Run': run_name,
                    'mean': score['mean'],
                    'stdev': score['stdev']
                })
    return pd.DataFrame(records)

def plot_three_level_grouped_facet_by_run_color(df: pd.DataFrame, 
                                                experiment_col: str, 
                                                method_col: str, 
                                                run_col: str, 
                                                score_col: str,
                                                score_label: str,
                                                output_filename: str):
    """
    Generates a swarmplot faceted by Experiment, layering numerical 
    mean and standard deviation labels over the points.
    """
    plt.figure(figsize=(20, 10))
    sns.set_theme(style="whitegrid")
    
    # 1. Define Order (Ensure correct custom order for Experiments)
    experiment_order = sorted(list(set(df[experiment_col].tolist())))
         
    method_order = sorted(list(set(df[method_col])))
    
    # 2. Create the FacetGrid (Outer Grouping: Experiment)
    g = sns.FacetGrid(
        df, 
        col=experiment_col,           # Separate plots for each Experiment
        col_order=experiment_order,
        height=8, 
        aspect=0.8,
        sharex=False,
        # 'margin_titles' needs to be here
        #subplot_kws={'margin_titles': True} 
    )

    # 3. Map the Swarmplot
    # X-axis: Method (Middle Group)
    # Hue: Run/Language (Inner Group - COLOR)
    g.map_dataframe(
        sns.swarmplot, 
        x=method_col,      # X-axis position (Middle Group)
        y=score_col,
        hue=run_col,        # <--- KEY CHANGE: Use Run/Language for color
        order=method_order,
        palette='tab10',    # 'tab10' provides 10 distinct colors for the 7 languages
        size=5,
        legend=True
    )
    
    # 4. Add Custom Enhancements
    # Adjust loc and bbox_to_anchor to move the legend outside the axes
    g.add_legend(
        title=run_col, 
        loc='center left', 
        bbox_to_anchor=(1, 0.5), # Moves it to the far right of the figure
        frameon=True             # Adds a border to make it distinct
    ) 
    
    # 5. Fix: Iterate through all axes to rotate x-ticks
    for ax in g.axes.flat:
        ax.tick_params(axis='x', labelrotation=45)
        # Extra fix: Ensure the 'right' alignment for rotated text
        for label in ax.get_xticklabels():
            label.set_horizontalalignment('right')
    
    # Adjust titles and layout
    g.set_axis_labels(method_col, score_label) 
    g.set_titles(col_template="{col_name}", size=14) 
    g.fig.suptitle(f'Language-Colored Individual {score_label} Scores', fontsize=16, y=1.05)
    
    # Use subplots_adjust to make room for the moved legend on the right
    plt.subplots_adjust(right=0.85, top=0.9)
    
    g.savefig(output_filename, bbox_inches='tight') # bbox_inches='tight' is key!

    df.to_csv(output_filename.replace('.png', '.csv'), index=False)
    
    plt.close()
    print(f"Plot and Table saved to {os.path.dirname(output_filename)}")
    plt.close()

def transform_nested_dict_to_dataframe(data_dict: Dict[str, Dict[str, Dict[str, float]]], ) -> pd.DataFrame:
    """
    Transforms the nested dictionary structure data[method][lang] = float 
    into a long-format Pandas DataFrame suitable for plotting.
    """
    records = []
    
    # Iterate through the outermost dictionary (Method)
    for method, lang_scores in data_dict.items():
        # Iterate through the inner dictionary (Language)
        for language, score in lang_scores.items():
            records.append({
                'Method': method,
                'Language': language,
                'mean': score['mean'],
                'stdev': score['stdev']
            })
            
    return pd.DataFrame(records)


def plot_method_language_comparison_with_labels(df: pd.DataFrame, 
                                                method_col: str, 
                                                language_col: str, 
                                                score_col: str,
                                                score_label: str,
                                                output_filename: str):
    """
    Generates a grouped bar chart and adds the actual numerical score 
    on top of each bar.
    """
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(14, 6))
    
    # Get the current axes object
    ax = plt.gca()

    # Create the Grouped Bar Plot
    # X-axis is the Language, and groups (HUE) are the Methods.

    # 3. Create the Plot
    sns.barplot(
        data=df, x=language_col, y=score_col, hue=method_col,
        palette='viridis', errorbar=None, ax=ax
    )

    # 4. Iterate over patches and the calculated stats simultaneously
    for p, (_, row) in zip(ax.patches, df.iterrows()):
        mean_val = row['mean']
        std_val = row['stdev']

        if pd.isna(mean_val):
            continue

        # Format the string to show Mean and (±STDEV)
        # \n creates a new line to keep the label from getting too wide
        text_to_display = f"{mean_val:.2f}\n(±{std_val:.2f})"

        # Determine vertical alignment (above for positive, below for negative)
        va_align = 'bottom' if mean_val >= 0 else 'top'

        ax.text(
            p.get_x() + p.get_width() / 2,
            mean_val,
            text_to_display,
            ha='center',
            va=va_align,
            fontsize=8,
            fontweight='bold'
        )
    
    # 1. Determine data limits
    max_score = df[score_col].max()
    min_score = df[score_col].min()

    # 2. Calculate range for buffer
    y_range = max_score - min_score
    
    # Set Y-min: If min score is negative, allow a small buffer below it. 
    # Otherwise, set it to 0.
    if min_score < 0:
        y_min = min_score - (y_range * 0.05) # 5% buffer below the lowest point
    else:
        y_min = 0 # Start exactly at zero if all scores are positive

    # Set Y-max: Add 10% buffer above the highest score for the labels
    y_max = max_score + (y_range * 0.15) 
    
    ax.set_ylim(y_min, y_max)
    
    # --- ENHANCEMENTS ---
    plt.title(f'Comparison of Methods Across Languages', fontsize=16)
    plt.xlabel('Language', fontsize=14)
    plt.ylabel(score_label, fontsize=14)
    plt.legend(title='Method', loc='best')
    
    # Adjust Y-limit slightly to make space for the labels above the highest bar
    # Find the maximum value and add 10% of the range to the top
    max_score = df[score_col].max()
    min_score = df[score_col].min()
    y_range = max_score - min_score
    
    # Set the y-limit, ensuring it starts near zero (or below zero if needed)
    y_min = min_score * 0.95 if min_score < 0 else 0
    y_max = max_score + (y_range * 0.1) # Add 10% buffer
    ax.set_ylim(y_min, y_max)
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    
    print(f"Plot saved successfully to {output_filename}")

def plot_dict(data, filename, title):
    sns.set_theme(style="whitegrid")
    plt.figure(figsize=(8, 5))
    ax = sns.barplot(
        x=list(data.keys()), 
        y=list(data.values()), 
        hue=list(data.keys()), 
        palette="viridis", 
        #legend=False
    )
    for p in ax.patches:
        score_value = p.get_height()
        
        # Skip very small values to keep the chart clean
        if abs(score_value) < 0.005: 
            continue

        x_position = p.get_x() + p.get_width() / 2
        
        # DYNAMIC LOGIC FOR NEGATIVES:
        if score_value >= 0:
            # Positive bars: text goes above, alignment is 'bottom'
            va_align = 'bottom'
            offset = 3 # points above
        else:
            # Negative bars: text goes below, alignment is 'top'
            va_align = 'top'
            offset = -3 # points below

        ax.text(
            x_position, 
            score_value, 
            f'{score_value:.2f}', 
            ha='center', 
            va=va_align, 
            fontsize=9,
            # Ensure text is slightly moved away from the bar edge
            # using 'xytext' with 'textcoords' is more reliable than just y_position
        )
    plt.title(f"{title}")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Plot saved successfully to {filename}")

def get_best_result(d: dict[str, tuple[float, int, float]]) -> float:
    # if the logit is highest, that is the best result
    logit_list = []
    for word, ls in d.items():
        logit, rank, prob = ls
        logit_list.append(logit)
    return max(logit_list)

if __name__ == "__main__":
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(absolute_directory, "data", "interventions")

    langs = list(lang_to_flores_key.keys())

    methods = ['description', 'frequency', 'value']
    #experiments = ['original', 'direction_ablation_across_layers', 'direction_ablation_across_layers_everything', 'direction_ablation', 'ablation', 'ablation_everything_except', 'amplification', 'intervention', 'direction_intervention']
    method_to_colname = {'description': 'AnnSel', 'frequency': 'FreqSel', 'value': 'ValSel'}
    #experiments_to_colname = {
    #    'original': 'original', 'ablation': 'distractor ablation', 'ablation_everything_except': 'ablation',
    #    'direction_ablation_across_layers': 'distractor direction ablation', 'direction_ablation_across_layers_everything': 'direction ablation',
    #    'amplification': 'amplification', 'amplification_everything_except': 'non-distractor amplification', 'intervention': 'intervention',
    #    'direction_intervention': 'one-layer direction intervention', 'direction_ablation': 'one-layer distractor direction ablation',
    #}
    experiments = ['original', 'distractor ablation', 'ablation', 'distractor one-layer direction ablation', 
                'one-layer direction ablation', 'distractor multi-layer direction ablation', 
                'multi-layer direction ablation', 'amplification', 'non-distractor amplification', 
                'feature-intervention', 'one-layer direction intervention',]

    for prompt_lang in langs:
        for adj_lang in langs:
            dir_path = os.path.join(data_directory, prompt_lang, adj_lang)

            # before intervention
            """
            before_intervention_logits_list = dict() # key: lang, val: logits
            with open(os.path.join(dir_path, f"{methods[0]}_based_{experiments[1]}_logits_and_ranks.json"), 'r') as f:
                tmp_d = json.load(f)
            for lang in langs:
                before_intervention_logits_list[lang] = list()
            for _, val in tmp_d.items():
                for _, d in val.items():
                    for lang, v in d.items():
                        before_intervention_logits_list[lang].append(v[0])
            before_intervention_rank_list = dict()
            with open(os.path.join(dir_path, "before_intervention_ranks.json"), 'r') as f:
                tmp_d = json.load(f)
            for lang in langs:
                before_intervention_rank_list[lang] = list()
            for _, val in tmp_d.items():
                for lang, v in val.items():
                    before_intervention_rank_list[lang].append(v)
            before_intervention_logit = dict()
            for k, v in before_intervention_logits_list.items():
                before_intervention_logit[k] = sum(v) / len(v)
            before_intervention_rank = dict()
            for k, v in before_intervention_rank_list.items():
                before_intervention_rank[k] = sum(v) / len(v)
            filename = "before_intervention_logit"
            plot_dict(before_intervention_logit, os.path.join(dir_path, f'{filename}.png'), filename)
            filename = "before_intervention_rank"
            plot_dict(before_intervention_rank, os.path.join(dir_path, f'{filename}.png'), filename)"""

            # after intervention
            logit_dict = dict()
            rank_dict = dict()
            for experiment in experiments:
                experiment_key = experiment
                logit_dict[experiment_key] = dict()
                for method in methods:
                    method_key = method_to_colname[method]
                    logit_dict[experiment_key][method_key] = dict()

                    #if experiment == 'original':
                    #    file_name = f"{method}_based_{experiments[1]}_logits_and_ranks.json"
                    #else:
                    #    file_name = f"{method}_based_{experiment}_logits_and_ranks.json"
                    file_name = f"interventions_and_results_{method}.json"
                    with open(os.path.join(dir_path, file_name), 'r') as f:
                        tmp_d = json.load(f)

                    logit_list = dict()
                    #rank_list = dict()
                    for lang in langs:
                        logit_list[lang] = list()
                        #if lang == prompt_lang or lang == adj_lang or lang == 'en':
                        #    rank_list[lang] = list()
                    
                    """
                    for _, vals in tmp_d.items():
                        if experiment == 'original':
                            # just get the original
                            for l, d in vals[prompt_lang].items():
                                logit_list[l].append(d[0])
                        elif experiment == 'ablation' or experiment == 'amplification_everything_except' or experiment == 'direction_ablation' or experiment == "direction_ablation_across_layers":
                            # prompt_lang ablation or everything amp except prompt lang
                            for l, d in vals[prompt_lang].items():
                                logit_list[l].append(d[1])
                        elif experiment == 'amplification' or experiment == 'ablation_everything_except' or experiment == "direction_ablation_across_layers_everything":
                            # adj lang amplification or ablation except adj lang
                            for l in langs:
                                logit_list[l].append(vals[adj_lang][l][1])
                                if l == adj_lang or l == prompt_lang or l == 'en':
                                    rank_list[l].append(vals[adj_lang][l][2])
                        elif experiment == 'intervention' or experiment == 'direction_intervention':
                            # prompt_lang_ablation + adj lang amplification
                            for l in langs:
                                logit_list[l].append(vals[prompt_lang][adj_lang][l][1])
                        else:
                            raise KeyError('invalid experiment name')
                    """
                    for prompt, vals in tmp_d[experiment].items():
                        if experiment == "original":
                            d = vals["langs"]
                            
                        elif experiment == 'distractor ablation' or experiment == 'distractor one-layer direction ablation' or experiment == 'distractor multi-layer direction ablation' or experiment == 'amplification':
                            # prompt lang ablation or amplification everything except
                            d = vals[prompt_lang]["langs"]
                        elif experiment == 'ablation' or experiment == 'one-layer direction ablation' or experiment == 'multi-layer direction ablation' or experiment == 'non-distractor amplification':
                            # adj lang amplification or ablation everything
                            d = vals[adj_lang]['langs']
                        elif experiment == 'feature-intervention' or experiment == 'one-layer direction intervention':
                            # prompt_lang_ablation + adj lang amplification
                            d = vals[prompt_lang][adj_lang]["langs"]
                        else:
                            raise KeyError('invalid experiment name')

                        base = get_best_result(d[adj_lang])
                        for l in langs:
                            target = get_best_result(d[l])
                            logit_list[l].append(target - base)

                    for key, val in logit_list.items():
                        mean = statistics.mean(val)
                        stdev = statistics.stdev(val)
                        logit_dict[experiment_key][method_key][key] = {'mean': mean, 'stdev': stdev}
                    
                    #if experiment == 'amplification':
                    #    rank_dict[method_key] = dict()
                    #    for key, val in rank_list.items():
                    #        mean = statistics.mean(val)
                    #        stdev = statistics.stdev(val)
                    #        rank_dict[method_key][key] = {'mean': mean, 'stdev': stdev}

            #pd.set_option('display.max_rows', None)
            #pd.set_option('display.max_columns', None)
            df = transform_dict_to_dataframe(logit_dict)
            #print(df, flush=True)
            file_name = 'all_interventions.png'
            out_path = os.path.join(dir_path, file_name)
            title = f"Prompt {prompt_lang}, Adj {adj_lang}, interventions"
            plot_three_level_grouped_facet_by_run_color(
                df, 'Method', 'Experiment', 'Run', 'mean', 'logit difference',
                out_path
            )

            #df2 = transform_nested_dict_to_dataframe(rank_dict)
            #print(df2, flush=True)
            #file_name = 'intervention_rank.png'
            #outpath = os.path.join(dir_path, file_name)
            #plot_method_language_comparison_with_labels(df2, 'Method', 'Language', 'mean', 'rank', outpath)
