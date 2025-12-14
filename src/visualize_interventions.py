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
    # The legend now shows the Run/Language colors
    g.add_legend(title=run_col, loc='upper right') 
    
    # Set axis labels for the individual plots
    g.set_axis_labels(method_col, score_label) 
    
    # Set titles for the columns (Experiments)
    g.set_titles(col_template="{col_name} Experiment", size=14) 
    
    # Set the overall title
    g.fig.suptitle(f'Language-Colored Individual {score_label} Scores', fontsize=16, y=1.02)
    
    # 5. Save and return
    plt.tight_layout()
    g.savefig(output_filename)
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
    plt.figure(figsize=(10, 6))
    
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
    
    
    plt.tight_layout()
    plt.savefig(output_filename)
    plt.close()
    
    print(f"Plot saved successfully to {output_filename}")

if __name__ == "__main__":
    current_file_path = __file__
    current_directory = os.path.dirname(current_file_path)
    absolute_directory = os.path.abspath(current_directory)
    data_directory = os.path.join(absolute_directory, "data", "interventions")

    langs = list(lang_to_flores_key.keys())

    methods = ['description', 'frequency', 'value']
    experiments = ['ablation', 'amplification', 'intervention']
    for prompt_lang in os.listdir(data_directory):
        for adj_lang in os.listdir(os.path.join(data_directory, prompt_lang)):
            dir_path = os.path.join(data_directory, prompt_lang, adj_lang)
            
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
                        if experiment == 'ablation':
                            # prompt_lang ablation
                            for l, d in vals[prompt_lang].items():
                                logit_list[l].append(d[1])
                        elif experiment == 'amplification':
                            # any lang amplification
                            for l in langs:
                                logit_list[l].append(vals[l][l][1])
                                rank_list[l].append(vals[l][l][2])
                        else:
                            # prompt_lang_ablation + any lang amplification
                            for l in langs:
                                logit_list[l].append(vals[prompt_lang][l][l][1])
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
