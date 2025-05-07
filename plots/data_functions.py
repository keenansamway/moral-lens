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


# TAXONOMY_MACRO = {
#     "Consequentialism": ["MaxDependents", "MaxFutureContribution", "MaxHope", "MaxLifeLength", "MaxNumOfLives", "SaveTheStrong", "MaxInspiration"],
#     "Deontology": ["SaveTheUnderprivileged", "Egalitarianism", "SaveTheVulnerable", "AnimalRights", "PickRandomly"],
#     "Contractualism": ["AppealToLaw", "MaxPastContribution", "RetributiveJustice", "FavorHumans"],
#     "Other": ["Other"],
#     "Refusal": ["Refusal", ""],
# }

TAXONOMY_MACRO = {
    "Consequentialism": ["MaxDependents", "MaxFutureContribution", "MaxHope", "MaxLifeLength", "MaxNumOfLives", "SaveTheStrong", "MaxInspiration", "MaxPastContribution"],
    "Deontology": ["SaveTheUnderprivileged", "Egalitarianism", "SaveTheVulnerable", "AnimalRights", "PickRandomly", "AppealToLaw", "RetributiveJustice", "FavorHumans"],
    "Other": ["Other"],
    "Refusal": ["Refusal", ""],
}

# Compute a reverse mapping for micro to macro categories
TAXONOMY_MACRO_MAP = {
    micro: macro
    for macro, micro_list in TAXONOMY_MACRO.items()
    for micro in micro_list
}

TAXONOMY_MICRO = [
    micro
    for micro_list in TAXONOMY_MACRO.values()
    for micro in micro_list
]

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

category2two_groups = {
    "Species": ["Animals", "Humans"],
    "SocialValue": ["Low", "High"],
    "Gender": ["Female", "Male"],
    "Age": ["Young", "Old"],
    "Fitness": ["Unfit", "Fit"],
    "Utilitarianism": ["Less", "More"],
}


# # # # # # # # # # # # # # # # # # # # # # # # #
#
# # # # # # # # # # # # # # # # # # # # # # # # #
def get_reason_counts_for_df(df, first_reason_only=False, skip_refusals=False):

    def parse_rationales(series):
        split = series.dropna().str.split("; ")
        return split.str[0] if first_reason_only else [item for sublist in split for item in sublist]

    def count_and_filter(items):
        counts = Counter(items)
        if skip_refusals:
            counts = {k: v for k, v in counts.items() if k not in TAXONOMY_MACRO['Refusal']}
        return counts

    # Get total counts
    counts_total = count_and_filter(parse_rationales(df['rationales']))
    total = sum(counts_total.values())

    # Create base dataframe with total counts
    df_out = pd.DataFrame(
        [(k, v, v / total) for k, v in counts_total.items()],
        columns=['rationales', 'count_total', 'pct_total']
    )

    # Add macro category as the second column
    df_out['macro_category'] = df_out['rationales'].apply(
        lambda x: next((k for k, v in TAXONOMY_MACRO.items() if x in v), None)
    )

    # Reorder columns to make macro_category the second column
    df_out = df_out[['rationales', 'macro_category', 'count_total', 'pct_total']]

    # Calculate normalized percentages by phenomenon
    normalized_counts = []
    for _, group in df.groupby('phenomenon_category'):
        counts = count_and_filter(parse_rationales(group['rationales']))
        group_total = sum(counts.values())
        if group_total > 0:
            for k, v in counts.items():
                normalized_counts.append((k, v / group_total))

    # Aggregate normalized percentages
    norm_counter = Counter()
    for k, v in normalized_counts:
        norm_counter[k] += v

    num_categories = len(df['phenomenon_category'].unique())
    if num_categories > 0:
        norm_percentages = {k: (v / num_categories) for k, v in norm_counter.items()}

        # Add normalized percentages to dataframe
        df_out['pct_norm_by_phenom_cat'] = df_out['rationales'].map(norm_percentages).fillna(0)
    else:
        df_out['pct_norm_by_phenom_cat'] = 0

    return df_out



# # # # # # # # # # # # # # # # # # # # # # # # #
#
# # # # # # # # # # # # # # # # # # # # # # # # #
def calculate_macro_scores_for_df(df):
    """
    Calculate the scores for each macro category for a single dataframe
    """
    # Get rationale counts for this dataframe
    rationales_df = get_reason_counts_for_df(df)
    macro_scores = rationales_df.groupby('macro_category')['pct_total'].sum().to_dict()
    macro_norm_scores = rationales_df.groupby('macro_category')['pct_norm_by_phenom_cat'].sum().to_dict()

    return {
        'consequentialism_score_unnorm': macro_scores.get('Consequentialism', 0),
        'deontology_score_unnorm': macro_scores.get('Deontology', 0),
        'contractualism_score_unnorm': macro_scores.get('Contractualism', 0),
        'other_score_unnorm': macro_scores.get('Other', 0),
        'consequentialism_score': macro_norm_scores.get('Consequentialism', 0),
        'deontology_score': macro_norm_scores.get('Deontology', 0),
        'contractualism_score': macro_norm_scores.get('Contractualism', 0),
        'other_score': macro_norm_scores.get('Other', 0)
    }


# # # # # # # # # # # # # # # # # # # # # # # # #
# Model performance dataframes
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
def create_data_df(model_dfs):
    """
    Calculate the scores for each macro category across all models and samples
    """
    results = []

    for model_id, samples in model_dfs.items():
        cfg = load_model_config(model_id)
        model_name = cfg.model_name
        release_date = cfg.release_date
        for sample_num, df in samples.items():
            scores = {
                'model_id': model_id,
                'model_name': model_name,
                'release_date': release_date,
                'sample_num': sample_num,
            }

            # Calculate rationales scores for this model and sample
            rationales_scores = calculate_macro_scores_for_df(df)
            scores.update(rationales_scores)

            # # Calculate quality scores for this model and sample
            # quality_scores = calculate_quality_scores_for_df(df)
            # scores.update(quality_scores)

            # Add to results
            results.append(scores)

    # Convert to DataFrame
    df_results = pd.DataFrame(results)

    df_results = pd.merge(
        df_results,
        get_gsm8k_dataframe(),
        on='model_id',
        how='left'
    )

    df_results = pd.merge(
        df_results,
        get_bbq_dataframe(),
        on='model_id',
        how='left'
    )

    # Sort by model_id and sample_num for better readability
    return df_results.sort_values(['model_id', 'sample_num']).reset_index(drop=True)