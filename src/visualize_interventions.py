import json
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from typing import Dict, Any, List

from intervention import (
    visualize_bar_2ddict_outer_inter, bar_graph_visualize,
    create_multi_series_histogram,
)
from template import lang_to_flores_key

# Re-using the transformation function from the previous step (it's essential)
def transform_dict_to_dataframe(data_dict: Dict[str, Dict[str, Dict[str, float]]], 
                               score_column_name: str = 'Score') -> pd.DataFrame:
    records = []
    for experiment, methods in data_dict.items():
        for method, scores in methods.items():
            for run_name, score in scores.items():
                records.append({
                    'Experiment': experiment,
                    'Method': method,
                    'Run': run_name,
                    score_column_name: score
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
    Generates a scatter plot with three levels of grouping:
    1. Outer (Experiment): Separated by Facet.
    2. Middle (Method): Separated by X-axis position.
    3. Inner (Run/Language): Separated by Color (Hue).
    """
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    # 1. Define Order (Ensure correct custom order for Experiments)
    experiment_order = sorted(df[experiment_col].unique()) 
    if set(experiment_order) == {'ablation', 'amplification', 'both'}:
         experiment_order = ['ablation', 'amplification', 'both']
         
    method_order = sorted(df[method_col].unique())
    
    # 2. Create the FacetGrid (Outer Grouping: Experiment)
    g = sns.FacetGrid(
        df, 
        col=experiment_col,           # Separate plots for each Experiment
        col_order=experiment_order,
        height=5, 
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
        size=7,
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
    g.set_titles(col_template="{col_name} Experiment", size=14) 
    g.fig.suptitle(f'Language-Colored Individual {score_label} Scores', fontsize=16, y=1.05)
    
    # Use subplots_adjust to make room for the moved legend on the right
    plt.subplots_adjust(right=0.85, top=0.9)
    
    g.savefig(output_filename, bbox_inches='tight') # bbox_inches='tight' is key!
    plt.close()
    
    print(f"Plot saved successfully to {output_filename}")

def transform_nested_dict_to_dataframe(data_dict: Dict[str, Dict[str, float]], 
                                       score_column_name: str = 'Score') -> pd.DataFrame:
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
                score_column_name: score
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
    sns.barplot(
        data=df,
        x=language_col,
        y=score_col,
        hue=method_col,
        palette='viridis',
        errorbar=None, # Ensure no error bars are plotted
        ax=ax
    )
    
    # --- ADD DATA LABELS ---
    
    # Iterate over the bar containers (patches)
    for p in ax.patches:
        # Determine the height of the bar (the score value)
        score_value = p.get_height()
        
        # Determine the position of the text
        x_position = p.get_x() + p.get_width() / 2  # Center the text horizontally
        y_position = score_value  # Position vertically at the top of the bar

        # Format the score value (e.g., to 2 decimal places)
        # Adjust the format string as needed (e.g., '{:.1f}' for 1 decimal)

        if abs(score_value) < 0.005: 
            # If it's zero, skip drawing the label
            continue
        text_format = '{:.2f}' 
        text_to_display = text_format.format(score_value)
        
        # Add the text label
        ax.text(
            x_position, 
            y_position, 
            text_to_display, 
            ha='center', # Horizontal alignment: center
            va='bottom', # Vertical alignment: position text just above the bar
            fontsize=9
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
        legend=False
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

if __name__ == "__main__":
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(absolute_directory, "data", "interventions")

    langs = list(lang_to_flores_key.keys())

    methods = ['description', 'frequency', 'value']
    experiments = ['ablation', 'ablation_everything_except', 'amplification', 'amplification_everything_except', 'intervention']
    for prompt_lang in os.listdir(data_directory):
        for adj_lang in os.listdir(os.path.join(data_directory, prompt_lang)):
            dir_path = os.path.join(data_directory, prompt_lang, adj_lang)

            # before intervention
            before_intervention_logits_list = dict() # key: lang, val: logits
            with open(os.path.join(dir_path, f"{methods[0]}_based_{experiments[0]}_logits_and_ranks.json"), 'r') as f:
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
            plot_dict(before_intervention_rank, os.path.join(dir_path, f'{filename}.png'), filename)

            # after intervention
            logit_dict = dict()
            rank_dict = dict()
            for experiment in experiments:
                logit_dict[experiment] = dict()
                for method in methods:
                    logit_dict[experiment][method] = dict()

                    file_name = f"{method}_based_{experiment}_logits_and_ranks.json"
                    with open(os.path.join(dir_path, file_name), 'r') as f:
                        tmp_d = json.load(f)

                    logit_list = dict()
                    rank_list = dict()
                    for lang in langs:
                        logit_list[lang] = list()
                        rank_list[lang] = list()
                    for _, vals in tmp_d.items():
                        if experiment == 'ablation' or experiment == 'amplification_everything_except':
                            # prompt_lang ablation or everything amp except prompt lang
                            for l, d in vals[prompt_lang].items():
                                logit_list[l].append(d[1])
                        elif experiment == 'amplification' or experiment == 'ablation_everything_except':
                            # adj lang amplification or ablation except adj lang
                            for l in langs:
                                logit_list[l].append(vals[adj_lang][l][1])
                                rank_list[l].append(vals[adj_lang][l][2])
                        else:
                            # prompt_lang_ablation + adj lang amplification
                            for l in langs:
                                logit_list[l].append(vals[prompt_lang][adj_lang][l][1])
                    for key, val in logit_list.items():
                        logit_dict[experiment][method][key] = sum(val) / len(val)
                    
                    if experiment == 'amplification':
                        rank_dict[method] = dict()
                        for key, val in rank_list.items():
                            rank_dict[method][key] = sum(val) / len(val)

            df = transform_dict_to_dataframe(logit_dict, 'logit_diff')
            #print(df)
            file_name = 'intervention_logits.png'
            out_path = os.path.join(dir_path, file_name)
            title = f"Prompt {prompt_lang}, Adj {adj_lang}, interventions"
            plot_three_level_grouped_facet_by_run_color(
                df, 'Method', 'Experiment', 'Run', 'logit_diff', 'logit difference',
                out_path
            )

            df2 = transform_nested_dict_to_dataframe(rank_dict, 'rank')
            #print(df2)
            file_name = 'intervention_rank.png'
            outpath = os.path.join(dir_path, file_name)
            plot_method_language_comparison_with_labels(df2, 'Method', 'Language', 'rank', 'rank', outpath)
