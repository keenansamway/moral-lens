import json
from glob import glob
import os
from datetime import datetime
from matplotlib.legend import Legend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from collections import Counter
import pandas as pd
import numpy as np
import colorsys
from pathlib import Path
from matplotlib.lines import Line2D
import seaborn as sns

import textwrap
from adjustText import adjust_text

from moral_lens.models import load_model_config


TAXONOMY_MACRO = {
    "Consequentialism": ["MaxDependents", "MaxFutureContribution", "MaxHope", "MaxLifeLength", "MaxNumOfLives", "SaveTheStrong", "MaxInspiration"],
    "Deontology": ["SaveTheUnderprivileged", "Egalitarianism", "SaveTheVulnerable", "AnimalRights", "PickRandomly"],
    "Contractualism": ["AppealToLaw", "MaxPastContribution", "RetributiveJustice", "FavorHumans"],
    "Other": ["Other"],
    "Refusal": ["Refusal", ""],
}

TAXONOMY_MACRO = {
    "Consequentialism": ["MaxDependents", "MaxFutureContribution", "MaxHope", "MaxLifeLength", "MaxNumOfLives", "SaveTheStrong", "MaxInspiration", "MaxPastContribution"],
    "Deontology": ["SaveTheUnderprivileged", "Egalitarianism", "SaveTheVulnerable", "AnimalRights", "PickRandomly", "AppealToLaw", "RetributiveJustice", "FavorHumans"],
    "Other": ["Other"],
    "Refusal": ["Refusal", ""],
}

# TAXONOMY_MACRO = {
#     "Consequentialism": ["NumOfLives", "LifeExpectancy", "FutureContribution", "CurrentDependents", "Hope", "PhysicalStrength", "InspirationalValue", "PastContribution"],
#     "Deontology": ["MoralEquality", "SocialBackground", "Vulnerability", "AnimalWelfare", "RandomChoice", "LegalNorms", "Culpability", "HumanSpecies"],
#     "Other": ["Descriptive",],
#     "Refusal": ["Refusal", ""],
# }

# TAXONOMY_MACRO = {
#     "Consequentialism": ["MaxDependents", "MaxFutureContribution", "MaxHope", "MaxLifeLength", "MaxNumOfLives", "SaveTheStrong", "MaxInspiration"],
#     "Deontology": ["SaveTheUnderprivileged", "Egalitarianism", "SaveTheVulnerable", "AnimalRights", "PickRandomly", "AppealToLaw", "MaxPastContribution", "RetributiveJustice", "FavorHumans"],
#     "Other": ["Other"],
#     "Refusal": ["Refusal", ""],
# }

TAXONOMY_MICRO = [
    micro
    for micro_list in TAXONOMY_MACRO.values()
    for micro in micro_list
]

FIGURE_SAVE_DIR = "data/figures/"
os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)

clr = sns.color_palette('tab10')
DEVELOPER_STYLES = {
    "OpenAI": {"color": clr[0], "marker": "o"},
    "Meta": {"color": clr[2], "marker": "X"},
    "Google": {"color": clr[3], "marker": "d"},
    "Alibaba": {"color": clr[6], "marker": "P"},
    "Microsoft": {"color": clr[4], "marker": "<"},
    "Anthropic": {"color": clr[1], "marker": '^'},
    "DeepSeek": {"color": clr[8], "marker": 'v'},
    "AllenAI": {"color": clr[9], "marker": '8'},
}

# Color mapping for macro categories
COLOR_MAP = {
    'Consequentialism': '#E0B274', 'Deontology': '#8CC888', 'Contractualism': '#9CBADE', 'Other': '#D9D9D9', 'Refusal': '#FA5061'
}
# Create micro-level color mapping
def generate_color_shades(base_color, num_shades):
    base_rgb = tuple(int(base_color[1:][i:i+2], 16) for i in (0, 2, 4))
    return ['#{:02x}{:02x}{:02x}'.format(*(min(255, int(c * (0.7 + 0.6 * i / (num_shades - 1)))) for c in base_rgb)) for i in range(num_shades)]
MICRO_COLOR_MAP = {
    micro: (
        COLOR_MAP[macro]
        if len(micro_list) == 1
        else generate_color_shades(COLOR_MAP[macro], len(micro_list))[i]
    )
    for macro, micro_list in TAXONOMY_MACRO.items()
    for i, micro in enumerate(micro_list)
}



# # # # # # # # # # # # # # # # # # # # # # # # #
#
# # # # # # # # # # # # # # # # # # # # # # # # #
def get_reason_counts(df, first_reason_only=False, skip_refusals=False, normalize_by_phenomenon=False):
    def parse_rationales(series):
        split = series.dropna().str.split("; ")
        return split.str[0] if first_reason_only else [item for sublist in split for item in sublist]

    def count_and_filter(items):
        counts = Counter(items)
        if skip_refusals:
            counts = {k: v for k, v in counts.items() if k not in TAXONOMY_MACRO['Refusal']}
        return counts

    if normalize_by_phenomenon:
        all_counts = []
        for _, group in df.groupby('phenomenon_category'):
            counts = count_and_filter(parse_rationales(group['rationales']))
            total = sum(counts.values())
            if total > 0:
                normalized = {k: v / total for k, v in counts.items()}
                all_counts.append(normalized)

        combined = Counter()
        for c in all_counts:
            combined.update(c)

        num_groups = len(all_counts)
        counts = {k: (v / num_groups) * 100 for k, v in combined.items()}
        df_out = pd.DataFrame(list(counts.items()), columns=['rationales', 'percentage'])
        df_out['count'] = df_out['percentage'] * df_out['percentage'].sum() / 100  # Approximate
    else:
        counts = count_and_filter(parse_rationales(df['rationales']))
        total = sum(counts.values())
        df_out = pd.DataFrame(
            [(k, v, v / total * 100) for k, v in counts.items()],
            columns=['rationales', 'count', 'percentage']
        )

    df_out['macro_category'] = df_out['rationales'].apply(
        lambda x: next((k for k, v in TAXONOMY_MACRO.items() if x in v), None)
    )
    return df_out


def create_stacked_rationales_barchart(
    file_paths,
    figsize=(9,15),
    save_dir=None,
    first_reason_only=False,
    skip_refusals=False,
    normalize_by_count=True,
    normalize_by_phenomenon=False,
    sort_consequentialist=False,
    return_dataframe=False,
    legend_cols=2):

    # Group files by model id
    model_files = {}
    for i, file_path in enumerate(file_paths):
        model_id = file_path.split('/')[-1].split('_')[0]
        if model_id not in model_files:
            model_files[model_id] = []
        model_files[model_id].append(file_path)

    # Process each model's data by combining files
    models, model_segments = [], {}
    for model_id, model_file_paths in model_files.items():
        if model_id == "":
            continue

        # Combine all dataframes for this model
        combined_df = pd.concat([pd.read_csv(fp, keep_default_na=False) for fp in model_file_paths])
        models.append(model_id)

        rc_df = get_reason_counts(combined_df, first_reason_only=first_reason_only, skip_refusals=skip_refusals, normalize_by_phenomenon=normalize_by_phenomenon)
        rc_df = rc_df.sort_values('rationales', key=lambda x: [TAXONOMY_MICRO.index(i) for i in x])
        model_segments[model_id] = rc_df[['rationales', 'count', 'percentage', 'macro_category']].to_dict('records')

    # Sort models by the percentage of consequentialism
    if sort_consequentialist:
        models = sorted(models, key=lambda m: sum(seg['percentage'] for seg in model_segments[m] if seg['macro_category'] == 'Consequentialism'), reverse=True)

    # Create the stacked bar chart
    fig, ax = plt.subplots(figsize=figsize)
    legend_handles = {}

    for i, model_id in enumerate(reversed(models)):
        model_name = load_model_config(model_id).model_name
        left_val = 0
        for seg in model_segments[model_id]:
            if normalize_by_count:
                seg_perc = 'percentage'
            else:
                seg_perc = 'count'
            perc, reason, macro = seg[seg_perc], seg['rationales'], seg['macro_category']
            color = MICRO_COLOR_MAP.get(reason, COLOR_MAP.get(macro, '#333333'))
            bar = ax.barh(model_name, perc, left=left_val, color=color)
            if reason not in legend_handles:
                legend_handles[reason] = bar
            left_val += perc

    plt.rcParams.update({'font.size': 14})
    plt.yticks(rotation=0)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x/100:.1f}')) # Format x-axis as 0 to 1

    ordered_micro_categories = [micro for macro in TAXONOMY_MACRO.values() for micro in macro]
    ordered_legend_handles = {micro: legend_handles[micro] for micro in ordered_micro_categories if micro in legend_handles}
    plt.legend(ordered_legend_handles.values(), ordered_legend_handles.keys(),
            bbox_to_anchor=(0.5, -0.05), loc='upper center', ncol=legend_cols)

    ax.margins(y=0.01, x=0.01) # Set margins
    for spine in ax.spines.values(): # Remove spines
        spine.set_visible(False)
    ax.tick_params(axis='y', which='both', length=0) # Remove y axis tick lines
    # Add a title above the plot
    # judge_model_name = load_model_config(file_paths[0].split("/")[-1].split("_")[1]).model_name
    # ax.set_title(f'Judge Model: {judge_model_name}', fontsize=16, pad=15)

    plt.tight_layout()

    if save_dir:
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        plt.savefig(save_dir, dpi=450, bbox_inches='tight')

    plt.show()

    if return_dataframe:
        # Create a more detailed dataframe with all rationales and macro categories
        results = []
        for model_id in models:
            model_name = load_model_config(model_id).model_name
            developer = load_model_config(model_id).developer
            release_date = load_model_config(model_id).release_date
            model_data = {'model_name': model_name, 'model_id': model_id, 'developer': developer, 'release_date': release_date}

            # Add individual rationales
            for seg in model_segments[model_id]:
                model_data[seg['rationales']] = seg['percentage']

            # Calculate and add macro category totals
            macro_totals = {}
            for category in TAXONOMY_MACRO.keys():
                macro_total = sum(seg['percentage'] for seg in model_segments[model_id]
                                 if seg['macro_category'] == category)
                macro_totals[category] = macro_total
                model_data[category] = macro_total

            results.append(model_data)

        # Convert to DataFrame
        results_df = pd.DataFrame(results).fillna(0)

        return results_df

def create_stacked_rationales_barchart_vert(
    file_paths,
    figsize=(15, 7.5),
    save_dir=None,
    first_reason_only=False,
    skip_refusals=False,
    normalize_by_count=True,
    normalize_by_phenomenon=False,
    sort_consequentialist=False,
    return_dataframe=False,
    wrap_width=12):

    # Group files by model id
    model_files = {}
    for i, file_path in enumerate(file_paths):
        model_id = file_path.split('/')[-1].split('_')[0]
        if model_id not in model_files:
            model_files[model_id] = []
        model_files[model_id].append(file_path)

    # Process each model's data by combining files
    models, model_segments = [], {}
    for model_id, model_file_paths in model_files.items():
        if model_id == "":
            continue

        # Combine all dataframes for this model
        combined_df = pd.concat([pd.read_csv(fp, keep_default_na=False) for fp in model_file_paths])
        models.append(model_id)

        rc_df = get_reason_counts(combined_df, first_reason_only=first_reason_only, skip_refusals=skip_refusals, normalize_by_phenomenon=normalize_by_phenomenon)
        rc_df = rc_df.sort_values('rationales', key=lambda x: [TAXONOMY_MICRO.index(i) for i in x])
        model_segments[model_id] = rc_df[['rationales', 'count', 'percentage', 'macro_category']].to_dict('records')

    # Sort models by the percentage of consequentialism
    if sort_consequentialist:
        models = sorted(models, key=lambda m: sum(seg['percentage'] for seg in model_segments[m] if seg['macro_category'] == 'Consequentialism'), reverse=True)

    # Create the stacked vertical bar chart
    fig, ax = plt.subplots(figsize=figsize)
    legend_handles = {}

    for i, model_id in enumerate(models):  # Now left to right (no reverse)
        model_name = load_model_config(model_id).model_name
        wrapped_model_name = textwrap.fill(model_name, width=wrap_width, break_long_words=False)
        bottom_val = 0
        for seg in model_segments[model_id]:
            seg_perc = 'percentage' if normalize_by_count else 'count'
            perc, reason, macro = seg[seg_perc], seg['rationales'], seg['macro_category']
            color = MICRO_COLOR_MAP.get(reason, COLOR_MAP.get(macro, '#333333'))
            bar = ax.bar(wrapped_model_name, perc, bottom=bottom_val, color=color)
            if reason not in legend_handles:
                legend_handles[reason] = bar
            bottom_val += perc

    plt.rcParams.update({'font.size': 14})
    # Wrap x-axis labels by splitting on spaces
    wrapped_labels = [
        textwrap.fill(label.get_text(), width=wrap_width, break_long_words=False)
        for label in ax.get_xticklabels()
    ]
    ax.set_xticklabels(wrapped_labels, rotation=0, ha='center')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y/100:.1f}'))  # Format y-axis as 0 to 1

    # Legend formatting
    ordered_micro_categories = [micro for macro in TAXONOMY_MACRO.values() for micro in macro]
    ordered_legend_handles = {micro: legend_handles[micro] for micro in ordered_micro_categories if micro in legend_handles}
    # legend = ax.legend(
    #     ordered_legend_handles.values(),
    #     ordered_legend_handles.keys(),
    #     loc='upper center',
    #     bbox_to_anchor=(0.5, 0.05),  # Adjust this closer to the plot
    #     bbox_transform=fig.transFigure,  # Attach to the figure itself
    #     ncol=9,
    # )

    macro_legend_groups = {
        macro: [r for r in rationales if r in legend_handles]
        for macro, rationales in TAXONOMY_MACRO.items()
        if macro != "Refusal"  # Optional: skip "Refusal" from legend
    }

    # Adjust horizontal positioning of each legend group (tweak as needed)
    x_positions = [0.12, 0.46, 0.75, 0.9]
    x_positions = [x + 0.05 for x in x_positions]

    # Iterate over each macro category
    for i, (macro, rationales) in enumerate(macro_legend_groups.items()):
        if not rationales:
            continue

        group_handles = [legend_handles[r] for r in rationales if r in legend_handles]
        group_labels = [r for r in rationales if r in legend_handles]

        group_legend = Legend(
            fig,
            group_handles,
            group_labels,
            title=f'{macro}',
            ncol=3 if len(group_labels) > 4 else 2,
            title_fontsize=15,
            fontsize=14,
            loc='upper center',
            bbox_to_anchor=(x_positions[i], 0.03),  # Y position closer to plot
            bbox_transform=fig.transFigure,         # IMPORTANT: place relative to full figure
            frameon=False
        )
        group_legend.get_title().set_ha('center')
        fig.artists.append(group_legend)

    ax.margins(x=0.01, y=0.01)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(axis='x', which='both', length=0)

    judge_model_name = load_model_config(file_paths[0].split("/")[-1].split("_")[1]).model_name
    # ax.set_title(f'Judge Model: {judge_model_name}', fontsize=16, pad=15)

    plt.tight_layout()

    if save_dir:
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        plt.savefig(save_dir, dpi=450, bbox_inches='tight')

    plt.show()

    if return_dataframe:
        results = []
        for model_id in models:
            model_name = load_model_config(model_id).model_name
            developer = load_model_config(model_id).developer
            release_date = load_model_config(model_id).release_date
            model_data = {'model_name': model_name, 'model_id': model_id, 'developer': developer, 'release_date': release_date}

            for seg in model_segments[model_id]:
                model_data[seg['rationales']] = seg['percentage']

            for category in TAXONOMY_MACRO.keys():
                macro_total = sum(seg['percentage'] for seg in model_segments[model_id] if seg['macro_category'] == category)
                model_data[category] = macro_total

            results.append(model_data)

        return pd.DataFrame(results).fillna(0)


# # # # # # # # # # # # # # # # # # # # # # # # #
#
# # # # # # # # # # # # # # # # # # # # # # # # #
def plot_model_performance_vs_moral_feature(
    df,
    moral_feature='Consequentialism',
    performance_metric='GSM8K Platinum',
    figsize=(10, 8),
    developer_styles=None,
    save_dir=None,
    include_title=True,
    include_labels=True,
):
    """
    Create a scatter plot with a performance metric on x-axis and a selected moral stance on y-axis.
    Models are colored by developer and use developer-specific markers.

    Parameters:
    -----------
    df : DataFrame
        DataFrame containing model data with performance metric and moral stance information
    moral_feature : str
        The moral stance to plot on the y-axis (e.g., 'Consequentialism', 'Deontology', 'Contractualism')
    performance_metric : str
        The name of the performance metric to plot on the x-axis (e.g., 'GSM8K Platinum', 'BBQ Accuracy')
    figsize : tuple
        Figure size (width, height)
    developer_styles : dict
        Dictionary mapping developer names to style configurations
    save_dir : str
        Path to save the generated figure
    """

    # Create figure
    plt.figure(figsize=figsize)

    # Create a base scatter plot
    ax = plt.gca()

    # Group by developer
    developers = df['developer'].unique()

    # Plot each developer's models with their specific color and marker
    for developer in developers:
        developer_df = df[df['developer'] == developer]
        style = developer_styles[developer]
        color = style.get('color', 'gray')
        marker = style.get('marker', 'o')

        sns.scatterplot(
            x='gsm8k_score',
            y=moral_feature,
            data=developer_df,
            color=color,
            marker=marker,
            s=200,
            alpha=1,
            label=developer,
            ax=ax
        )

    # Calculate correlation coefficient
    corr = df['gsm8k_score'].corr(df[moral_feature])

    # Add labels for each point
    if include_labels:
        texts = []
        for _, row in df.iterrows():
            texts.append(plt.text(row['gsm8k_score'] + 0.005, row[moral_feature], row['model_name'], fontsize=9))

        # Use adjust_text to prevent overlapping labels
        adjust_text(
            texts,
            only_move={'points': 'xy', 'text': 'xy'},
            force_points=1.0,
            force_text=1.0,
            expand_points=(1.2, 1.2),
            expand_text=(1.2, 1.2),
            lim=1000
        )

    # Add labels and title
    # plt.xlim(0.74, 1.01)

    plt.xlabel(performance_metric, fontsize=13)
    plt.ylabel(f'Proportion of {moral_feature} (%)', fontsize=13)
    title = f'{performance_metric} vs. {moral_feature} (r = {corr:.3f})'
    if include_title:
        plt.title(title, fontsize=15)

    # Add legend with customization
    legend = plt.legend(title="Developer", loc='best', frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    # Set the y-axis to display percentages properly
    # ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

    plt.tight_layout()
    ax.grid(True, alpha=0.3)

    if save_dir:
        os.makedirs(os.path.dirname(save_dir), exist_ok=True)
        plt.savefig(save_dir, dpi=450, bbox_inches='tight')

    plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # #
#
# # # # # # # # # # # # # # # # # # # # # # # # #
def plot_feature_over_time(df, moral_feature='Consequentialism', developer_styles=None, figsize=(10, 8), title=None, save_path=None):
    """
    Create a scatter plot showing the evolution of moral feature scores over time.

    Parameters:
    -----------
    df : DataFrame
        DataFrame containing model data with 'release_date' and moral feature columns
    developer_styles : dict, optional
        Dictionary mapping developer names to style configurations (color, marker)
    figsize : tuple, optional
        Figure size (width, height)
    title : str, optional
        Custom title for the plot
    save_path : str, optional
        Path to save the figure
    moral_feature : str, optional
        The moral feature to plot (e.g., 'Consequentialism', 'Deontology', 'Contractualism')

    Returns:
    --------
    fig, ax : matplotlib Figure and Axes objects
    """
    # Convert release dates to datetime objects for plotting
    df['ReleaseDateTime'] = pd.to_datetime(df['release_date'])

    # Create the figure and axis
    fig, ax = plt.subplots(figsize=figsize)

    # Group by developer
    for developer, group in df.groupby('developer'):
        style = developer_styles.get(developer, {'color': 'gray', 'marker': 'o'}) if developer_styles else {'color': 'gray', 'marker': 'o'}
        color = style.get('color', 'gray')
        marker = style.get('marker', 'o')

        # Sort by release date
        group = group.sort_values('ReleaseDateTime')

        # Plot each developer's models
        # Use seaborn scatter plot with better aesthetics
        sns.scatterplot(
            x=group['ReleaseDateTime'],
            y=group[moral_feature],
            color=color,
            marker=marker,
            s=150,
            alpha=0.8,
            label=developer,
            ax=ax
        )

    # Add model names as annotations
    texts = []
    for i, row in df.iterrows():
        texts.append(ax.annotate(
            row['model_id'],
            (row['ReleaseDateTime'], row[moral_feature] + 0.05),
            fontsize=8,
        ))

    # Use adjust_text to prevent overlapping labels
    adjust_text(
        texts,
        only_move={'points': 'xy', 'text': 'xy'},
        force_points=1.0,
        force_text=1.0,
        expand_points=(1.2, 1.2),
        expand_text=(1.2, 1.2),
        lim=1000
    )

    # Format the x-axis to show dates nicely
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.xticks(rotation=45)

    # Calculate correlation
    corr = df['ReleaseDateTime'].astype(int).corr(df[moral_feature])

    # Add labels and title
    ax.set_xlabel('Model Release Date', fontsize=12)
    ax.set_ylabel(f'{moral_feature} Score (%)', fontsize=12)

    # Set title
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'Evolution of {moral_feature} Over Time (r = {corr:.3f})', fontsize=14)

    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.1f}'))

    # Add grid
    ax.grid(True, alpha=0.3)

    # Add legend
    legend = plt.legend(title="Ddveloper", loc='best', frameon=True)
    legend.get_frame().set_facecolor('white')
    legend.get_frame().set_alpha(0.9)

    plt.tight_layout()

    # Save the figure if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')

    plt.show()


# # # # # # # # # # # # # # # # # # # # # # # # #
#
# # # # # # # # # # # # # # # # # # # # # # # # #
def get_gsm8k_dataframe():
    def extract_gsm8k_result(result_file):
        with open(result_file, 'r') as f:
            result = json.load(f)

        result_obj = result['results']['gsm8k_platinum']

        mean = result_obj['exact_match,strict-match']
        std = result_obj['exact_match_stderr,strict-match']

        if 'exact_match,old-flexible-extract' in result_obj:
            mean = max(mean, result_obj['exact_match,old-flexible-extract'])
            std = max(std, result_obj['exact_match_stderr,old-flexible-extract'])

        if 'exact_match,flexible-extract' in result_obj:
            mean = max(mean, result_obj['exact_match,flexible-extract'])
            std = max(std, result_obj['exact_match_stderr,flexible-extract'])

        model_name = result['model_name']
        model_cfg = load_model_config(model_name)

        return model_cfg, mean, std

    files = glob("lm_eval_results/gsm8k_platinum/*/results_*.json")
    sorted_results = []
    for result_file in files:
        model_cfg, mean, std = extract_gsm8k_result(result_file)
        # model_name = model_cfg.model_name
        model_id = model_cfg.save_id
        # release_date = model_cfg.release_date
        sorted_results.append((model_id, mean, std))

    # Create and return the DataFrame
    return pd.DataFrame(sorted_results, columns=["model_id", "gsm8k_score", "gsm8k_std"])

def get_bbq_dataframe():
    models_bbq_accuracy = {
        "claude-3-haiku:beta": 0.625,
        "claude-3-sonnet:beta": 0.9,
        "claude-3.5-sonnet-20240620:beta": 0.949,
        "claude-3.7-sonnet:beta": 0.921,
        "command_r": 0.724,
        "command_r_plus": 0.899,
        "deepseek-r1": 0.966,
        "deepseek-chat": 0.967,
        "gemini-flash-1.5": 0.947,
        "gemini-pro-1.5": 0.945,
        "gemini-2.0-flash-001": 0.954,
        "gemini-2.0-flash-lite-001": 0.92,
        "gemini-2.5-flash-preview": 0.977,
        "gemini-2.5-pro-preview": 0.964,
        "gpt-3.5-turbo-0125": 0.606,
        "gpt-4o-2024-05-13": 0.951,
        "gpt-4o-mini-2024-07-18": 0.882,
        "gpt-4.1-2025-04-14": 0.926,
        "gpt-4.1-mini-2025-04-14": 0.921,
        "gpt-4.1-nano-2025-04-14": 0.875,
        "llama-3-8b-instruct": 0.765,
        "llama-3-70b-instruct": 0.91,
        "llama-3.1-70b-instruct": 0.954,
        "llama-3.1-405b-instruct": 0.945,
        "llama-4-scout": 0.875,
        "llama-4-maverick": 0.93,
        "qwen-2.5-7b-instruct": 0.906,
        "qwen-2.5-72b-instruct": 0.954,
    }
    return pd.DataFrame(
        [(model_id, score) for model_id, score in models_bbq_accuracy.items()],
        columns=["model_id", "bbq_score"]
    )


# # # # # # # # # # # # # # # # # # # # # # # # #
#
# # # # # # # # # # # # # # # # # # # # # # # # #
def get_model_quality_df(file_paths, sort_quality=False):
    # Create a dictionary to store dataframes by model
    model_dfs = {}

    # Group dataframes by model
    for file in file_paths:
        # Extract model_id and sample number from filename
        filename = os.path.basename(file)
        model_id = filename.split('_')[0]
        sample_num = filename.split('_s')[1].replace('.csv', '')

        if model_id not in model_dfs:
            model_dfs[model_id] = {}

        model_dfs[model_id][sample_num] = pd.read_csv(file, keep_default_na=False)

    # Create a list to store model quality data
    model_quality_data = []

    for model_id, samples in model_dfs.items():
        # Get model configuration
        model_cfg = load_model_config(model_id)

        # Concatenate all samples for this model into a single dataframe
        all_samples_df = pd.concat(samples.values())

        # Process binary yes/no answers for each quality measure
        metrics = ['consistency', 'logic', 'bias', 'pluralism']
        results = {}

        for metric in metrics:
            # Convert to lowercase and count only valid responses
            valid_responses = all_samples_df[metric].str.lower().isin(['yes', 'no'])
            if valid_responses.sum() > 0:
                # Calculate the proportion of 'yes' responses among valid responses
                results[metric] = (all_samples_df.loc[valid_responses, metric].str.lower() == 'yes').mean()
            else:
                results[metric] = np.nan

        # Only add complete entries
        if not any(np.isnan(value) for value in results.values()):
            model_quality_data.append({
                'model_id': model_cfg.save_id,
                'model_name': model_cfg.model_name,
                **results
            })

    # Create DataFrame from the collected data all at once
    model_quality_df = pd.DataFrame(model_quality_data)

    model_quality_df['acceptability'] = 1 - model_quality_df['bias']

    # Define metric groups for different quality calculations
    core_metrics = ['consistency', 'logic', 'acceptability']
    all_metrics = core_metrics + ['pluralism']

    # Calculate quality scores (more maintainable if metrics change)
    model_quality_df['avg_quality'] = model_quality_df[all_metrics].mean(axis=1)
    model_quality_df['avg_quality_sans_pluralism'] = model_quality_df[core_metrics].mean(axis=1)

    if sort_quality:
        model_quality_df = model_quality_df.sort_values(by='avg_quality', ascending=False)

    return model_quality_df



# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# import glob
from matplotlib.patches import Patch


# # # # # # # # # # # # # # # # # # # # # # # # #
#
# # # # # # # # # # # # # # # # # # # # # # # # #
def plot_rationale_distribution_(file_path, color_map=None, normalize_within_phenomenon=True, figsize=(18, 12)):
    # Default color map if none provided
    if color_map is None:
        color_map = {
            'Consequentialism': '#E0B274',
            'Deontology': '#8CC888',
            'Contractualism': '#9CBADE',
            'Other': '#D9D9D9',
            'Refusal': '#FA5061'
        }

    judge_model = file_path.split('/')[-1].split('_')[1]
    decision_model = file_path.split('/')[-1].split('_')[0]
    all_model_dfs = [process_model_files([file_path], judge_model)]

    # Combine the data from all models
    if not all_model_dfs:
        raise ValueError("No data found for the specified models")

    df_categorized_combined = pd.concat(all_model_dfs)

    # Get micro-categories counts by phenomenon and model
    micro_counts = df_categorized_combined.groupby(['phenomenon_category', 'model', 'micro_category']).size().unstack(fill_value=0)

    # Generate color shades for micro categories
    micro_color_map = generate_micro_color_map(TAXONOMY_MACRO, color_map)

    # Order micro categories according to their macro categories
    ordered_macro_columns = list(color_map.keys())
    ordered_micro = []
    for macro in ordered_macro_columns:
        if macro in TAXONOMY_MACRO:
            ordered_micro.extend(TAXONOMY_MACRO[macro])

    # Keep only columns that exist in the data
    ordered_micro = [col for col in ordered_micro if col in micro_counts.columns]
    if ordered_micro:
        micro_counts = micro_counts[ordered_micro]

    # Reset index to make plotting easier
    micro_counts_reset = micro_counts.reset_index()

    # Prepare the plot
    fig = plt.figure(figsize=figsize)
    phenomena = micro_counts_reset['phenomenon_category'].unique()
    models = df_categorized_combined['model'].unique()

    # Calculate bar width based on number of models
    num_models = len(models)
    group_width = 0.8  # Width of the group of bars
    bar_width = group_width / num_models

    # Generate patterns for different models (first model solid, others with patterns)
    patterns = [''] + ['//' + '/' * (i % 3) for i in range(num_models-1)]

    # Create index for x-axis positioning
    index = np.arange(len(phenomena))

    # Initialize bottom values for stacked bars for each model
    bottoms = {model: np.zeros(len(phenomena)) for model in models}

    # Plot each micro category as a stack
    for i, micro in enumerate(ordered_micro):
        if micro not in micro_counts.columns:
            continue

        # Get color for this micro category
        color = micro_color_map.get(micro, '#D9D9D9')

        # Plot data for each model
        for j, model in enumerate(models):
            # Get data for this model and micro category
            model_data = []
            for phenom in phenomena:
                # Extract values
                values = micro_counts_reset[
                    (micro_counts_reset['phenomenon_category'] == phenom) &
                    (micro_counts_reset['model'] == model)
                ][micro].values

                value = values[0] if len(values) > 0 else 0
                model_data.append(value)

            # Normalize if requested
            if normalize_within_phenomenon:
                # Calculate totals for each phenomenon for this model
                for k, phenom in enumerate(phenomena):
                    total = micro_counts_reset[
                        (micro_counts_reset['phenomenon_category'] == phenom) &
                        (micro_counts_reset['model'] == model)
                    ][ordered_micro].sum(axis=1).values

                    total = total[0] if len(total) > 0 else 1

                    if total > 0:
                        model_data[k] = (model_data[k] / total) * 100

            # Calculate x positions for bars
            bar_positions = index + bar_width * (j - (num_models-1)/2)

            # Plot bars with appropriate patterns
            plt.bar(
                bar_positions, model_data, bar_width,
                bottom=bottoms[model],
                color=color,
                label=micro if i == 0 and j == 0 else "",
                hatch=patterns[j]
            )

            # Update bottom values for next stack
            bottoms[model] += model_data

    # Create legend elements
    legend_elements = [
        # Decision model patches with a title
        Patch(facecolor='white', edgecolor='none', label='Judge Models:')
    ]

    # Add model patches to legend
    for i, model in enumerate(models):
        legend_elements.append(
            Patch(facecolor='lightgray', edgecolor='gray', hatch=patterns[i], label=model)
        )

    # Add spacer and rationale title
    legend_elements.extend([
        Patch(facecolor='none', edgecolor='none', label=''),
        Patch(facecolor='white', edgecolor='none', label='Rationales:')
    ])

    # Add all the micro category patches
    for micro in ordered_micro:
        if micro in micro_counts.columns:
            color = micro_color_map.get(micro, '#D9D9D9')
            legend_elements.append(Patch(facecolor=color, label=micro))

    # Create legend
    plt.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.16, 1), ncol=1, fontsize=10)

    # Add labels and title
    plt.xlabel('Phenomenon Category', fontsize=14)
    plt.ylabel('Percentage (%)' if normalize_within_phenomenon else 'Count', fontsize=14)
    plt.title(f'Distribution of Rationales by Phenomenon (Decision model: {decision_model})', fontsize=16)
    plt.xticks(index, phenomena, rotation=45, ha='right')

    plt.tight_layout()
    # return fig

# Helper functions that were used in the original code
def process_model_files(file_paths, model_name):
    """Process CSV files for a model and return categorized DataFrame."""
    all_categorized = []

    for file_path in file_paths:
        # Read the CSV file
        df = pd.read_csv(file_path, keep_default_na=False)

        # Apply categorization to each row
        for _, row in df.iterrows():
            if pd.isna(row['rationales']) or row['rationales'] == '':
                continue

            rationales = row['rationales'].split(';')
            for rationale in rationales:
                rationale = rationale.strip()
                # Check if rationale is in any category
                for category, values in TAXONOMY_MACRO.items():
                    if rationale in values:
                        all_categorized.append({
                            'dilemma_id': row['id'],
                            'macro_category': category,
                            'micro_category': rationale,
                            'phenomenon_category': row.get('phenomenon_category', 'Unknown'),
                            'model': model_name
                        })
                        break

    # Convert to DataFrame
    return pd.DataFrame(all_categorized)

def generate_color_shades(base_color, num_shades):
    """Generate shades of a base color."""
    base_rgb = tuple(int(base_color[1:][i:i+2], 16) for i in (0, 2, 4))
    return ['#{:02x}{:02x}{:02x}'.format(*(min(255, int(c * (0.7 + 0.6 * i / (num_shades - 1)))) for c in base_rgb)) for i in range(num_shades)]

def generate_micro_color_map(taxonomy_macro, color_map):
    """Generate color mapping for micro categories."""
    micro_color_map = {}
    for macro, micro_list in taxonomy_macro.items():
        if len(micro_list) == 1:
            micro_color_map[micro_list[0]] = color_map[macro]
        else:
            shades = generate_color_shades(color_map[macro], len(micro_list))
            for i, micro in enumerate(micro_list):
                micro_color_map[micro] = shades[i]
    return micro_color_map



# # # # # # # # # # # # # # # # # # # # # # # # #
#
# # # # # # # # # # # # # # # # # # # # # # # # #
def plot_rationale_comparison_by_category(file_path, figsize=(22, 24)):
    """
    Plot a comparison of rationales used when the decision falls into category1 vs category2,
    aggregating data across all chain-of-thought runs.

    Args:
        decision_model_id: ID of the decision model
        judge_model_id: ID of the judge model
        results_dir: Directory containing the results
        figsize: Size of the figure
    """

    # Combine all dataframes
    df = pd.read_csv(file_path, keep_default_na=False)

    # Split rationales into individual items
    df_expanded = df.copy()
    df_expanded['rationale_list'] = df_expanded['rationales'].str.split(';')

    # Explode the rationales to get one row per rationale
    df_expanded = df_expanded.explode('rationale_list')
    df_expanded['rationale_list'] = df_expanded['rationale_list'].str.strip()

    # Group by phenomenon_category, decision_category, and rationale
    grouped_df = df_expanded.groupby(['phenomenon_category', 'decision_category', 'rationale_list']).size().reset_index(name='count')

    # Create a figure with subplots for each phenomenon category
    phenomena = df['phenomenon_category'].unique()

    # Use a 2-column layout
    n_rows = (len(phenomena) + 1) // 2  # Calculate rows needed for 2 columns

    fig, axes = plt.subplots(n_rows, 2, figsize=figsize)
    axes = axes.flatten()  # Flatten the axes array for easier indexing

    # Define grayscale colors for categories (darker for second category)
    cat1_color = '#B0B0B0'  # Lighter gray
    cat2_color = '#606060'  # Darker gray

    # Define colors for macro categories
    color_map = {
        'Consequentialism': '#E0B274',
        'Deontology': '#8CC888',
        'Contractualism': '#9CBADE',
        'Other': '#D9D9D9',
        'Refusal': '#FA5061'
    }

    # Generate micro-level colors
    micro_color_map = {}
    for macro, micro_list in TAXONOMY_MACRO.items():
        if len(micro_list) == 1:
            micro_color_map[micro_list[0]] = color_map[macro]
        else:
            base_color = color_map[macro]
            base_rgb = tuple(int(base_color[1:][i:i+2], 16) for i in (0, 2, 4))
            shades = ['#{:02x}{:02x}{:02x}'.format(*(min(255, int(c * (0.7 + 0.6 * i / (len(micro_list) - 1)))) for c in base_rgb))
                     for i in range(len(micro_list))]
            for i, micro in enumerate(micro_list):
                micro_color_map[micro] = shades[i]

    # Create plots for each phenomenon category
    for i, phenom in enumerate(phenomena):
        if i >= len(axes):
            break  # Safety check

        phenom_data = grouped_df[grouped_df['phenomenon_category'] == phenom]

        if phenom_data.empty:
            axes[i].set_visible(False)
            continue

        # Get category names for this phenomenon
        category1 = df[df['phenomenon_category'] == phenom]['category1'].iloc[0]
        category2 = df[df['phenomenon_category'] == phenom]['category2'].iloc[0]

        # Prepare data for each category
        cat1_data = phenom_data[phenom_data['decision_category'] == category1]
        cat2_data = phenom_data[phenom_data['decision_category'] == category2]

        # Get all rationales used for this phenomenon
        all_rationales = set(phenom_data['rationale_list'].unique())

        # Prepare data for plotting
        rationales = sorted(list(all_rationales),
                          key=lambda x: [TAXONOMY_MICRO.index(x) if x in TAXONOMY_MICRO else len(TAXONOMY_MICRO)])

        cat1_counts = [cat1_data[cat1_data['rationale_list'] == r]['count'].sum() if r in cat1_data['rationale_list'].values else 0
                     for r in rationales]
        cat2_counts = [cat2_data[cat2_data['rationale_list'] == r]['count'].sum() if r in cat2_data['rationale_list'].values else 0
                     for r in rationales]

        # Total counts for normalization
        total_cat1 = sum(cat1_counts)
        total_cat2 = sum(cat2_counts)

        # Convert to percentages
        cat1_pct = [count/total_cat1*100 if total_cat1 > 0 else 0 for count in cat1_counts]
        cat2_pct = [count/total_cat2*100 if total_cat2 > 0 else 0 for count in cat2_counts]

        # Colors for each rationale
        colors = [micro_color_map.get(r, '#D9D9D9') for r in rationales]

        # Plot with horizontal bars
        y = np.arange(len(rationales))
        height = 0.35

        ax = axes[i]
        bars1 = ax.barh(y - height/2, cat1_pct, height, label=category1, color=colors, alpha=0.7)
        bars2 = ax.barh(y + height/2, cat2_pct, height, label=category2, color=colors, alpha=1.0)

        # Add labels and legend
        ax.set_title(f'{phenom}: {category1} vs {category2}', fontsize=14)
        ax.set_yticks(y)
        ax.set_yticklabels(rationales, fontsize=12)
        ax.set_xlabel('Percentage (%)', fontsize=12)

        # Create a custom legend with grayscale colors
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor=cat2_color, label=category2),
            Patch(facecolor=cat1_color, label=category1)
        ]
        ax.legend(handles=legend_elements, fontsize=12, loc='lower right')

        # Add a grid for better readability
        ax.grid(axis='x', linestyle='--', alpha=0.3)

        # Add counts as text on the bars, positioned to avoid overlap
        def add_labels(bars, counts, total):
            for bar, count in zip(bars, counts):
                if count > 0:
                    width = bar.get_width()
                    # Position label outside of bar for better visibility
                    ax.text(width + 0.2, bar.get_y() + bar.get_height()/2,
                           f'{count}', ha='left', va='center', fontsize=10,
                           bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', pad=1))

        # add_labels(bars1, cat1_counts, total_cat1)
        # add_labels(bars2, cat2_counts, total_cat2)

        # Add bar separators
        for y_pos in y:
            ax.axhline(y=y_pos, color='gray', linestyle=':', alpha=0.2)

    # Hide any unused subplots
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)

    # plt.suptitle(f'Rationale Distribution by Decision Category (All CoT Runs)\n{decision_model} judged by {judge_model}', fontsize=18)
    plt.tight_layout()
    plt.show()