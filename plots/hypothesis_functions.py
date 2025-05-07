import os
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
import numpy as np
from scipy import stats
import seaborn as sns
from statsmodels.formula.api import ols
import statsmodels.api as sm
import pandas as pd

FIGURE_SAVE_DIR = 'data/hypothesis_figures'
os.makedirs(FIGURE_SAVE_DIR, exist_ok=True)

# Create a function to run and report hypothesis tests
def run_hypothesis_test(df, hypothesis_num, description, test_function, alpha=0.05):
    print(f"Hypothesis {hypothesis_num}: {description}")
    result = test_function(df)
    print(f"p-value: {result['p_value']:.5f}")
    if result['p_value'] < alpha:
        print(f"Conclusion: Reject null hypothesis (p < {alpha})")
    else:
        print(f"Conclusion: Fail to reject null hypothesis (p > {alpha})")
    print(f"Test statistic: {result['statistic']:.5f}")
    if 'effect_size' in result:
        print(f"Effect size: {result['effect_size']:.5f}")
    print("-" * 80)
    return result



# Hypothesis 1: Larger models do not perform better at moral reasoning than smaller ones
def test_hypothesis_1(df):
    # Aggregate data by model to get mean quality score per model
    model_data = df.groupby('model_name').agg({
        'parameter_category': 'first',
        'quality_score': 'mean'
    }).reset_index()

    # Run ANOVA
    formula = 'quality_score ~ C(parameter_category)'
    model = ols(formula, data=model_data).fit()
    anova_table = sm.stats.anova_lm(model, typ=2)

    # Effect size (Eta-squared)
    ss_total = anova_table['sum_sq'].sum()
    eta_squared = anova_table.iloc[0]['sum_sq'] / ss_total

    # Create boxplot
    plt.figure(figsize=(6, 4))
    # Define the order for parameter_category
    order = ['Small', 'Medium', 'Large', 'X-Large']
    sns.boxplot(x='parameter_category', y='quality_score', data=df, order=order)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Model Size', fontsize=12)
    plt.ylabel('Moral Reasoning Quality', fontsize=12)
    plt.grid(axis='y', linestyle='-', alpha=0.3)
    # plt.title('Quality Score by Parameter Category')
    # plt.savefig('hypothesis1_boxplot.png')
    plt.savefig(f"{FIGURE_SAVE_DIR}/size_vs_moral_quality.pdf", dpi=300, bbox_inches='tight')

    return {
        'p_value': anova_table.iloc[0]['PR(>F)'],
        'statistic': anova_table.iloc[0]['F'],
        'effect_size': eta_squared
    }



# Hypothesis 2: DPO does not decrease the proportion of consequentialist responses
def test_hypothesis_2(df):
    # Aggregate data by model to get mean consequentialism score per model
    model_data = df.groupby('model_name').agg({
        'is_dpo': 'first',
        'consequentialism_score': ['mean', 'std']
    }).reset_index()

    # Flatten multi-index columns
    model_data.columns = ['_'.join(col).strip('_') for col in model_data.columns.values]
    model_data = model_data.rename(columns={'is_dpo_first': 'is_dpo'})

    # Run t-test
    dpo_models = model_data[model_data['is_dpo']]['consequentialism_score_mean']
    non_dpo_models = model_data[~model_data['is_dpo']]['consequentialism_score_mean']

    # Run one-tailed t-test
    t_stat, p_val_one_tailed = stats.ttest_rel(dpo_models, non_dpo_models, alternative='less')

    # Effect size (Cohen's d)
    mean_dpo, mean_non_dpo = dpo_models.mean(), non_dpo_models.mean()
    std_dpo, std_non_dpo = dpo_models.std(), non_dpo_models.std()
    pooled_std = np.sqrt(((len(dpo_models) - 1) * std_dpo**2 + (len(non_dpo_models) - 1) * std_non_dpo**2) /
                         (len(dpo_models) + len(non_dpo_models) - 2))
    cohen_d = (mean_dpo - mean_non_dpo) / pooled_std

    # Create boxplot
    # plt.figure(figsize=(10, 6))
    # sns.boxplot(x='is_dpo', y='consequentialism_score_mean', data=df)
    # plt.title('Consequentialism Score by DPO Status')
    # plt.xlabel('DPO')
    # plt.ylabel('Consequentialism Score')
    # plt.savefig('hypothesis2_boxplot.png')

    # Create a better visualization - slope graph showing before/after for each model with error bars
    plt.figure(figsize=(6, 4))

    # Extract base model names (without SFT/DPO suffix)
    model_data['base_model'] = model_data['model_name'].apply(
        lambda x: x.split('SFT')[0].split('DPO')[0].strip()
    )

    # Create a mapping from base model to matching SFT/DPO pairs
    paired_models = {}
    for base in model_data['base_model'].unique():
        sft_model = model_data[(model_data['base_model'] == base) & (~model_data['is_dpo'])]
        dpo_model = model_data[(model_data['base_model'] == base) & (model_data['is_dpo'])]

        if len(sft_model) == 1 and len(dpo_model) == 1:
            paired_models[base] = {
                'sft_score': sft_model['consequentialism_score_mean'].values[0],
                'sft_std': sft_model['consequentialism_score_std'].values[0],
                'dpo_score': dpo_model['consequentialism_score_mean'].values[0],
                'dpo_std': dpo_model['consequentialism_score_std'].values[0]
            }

    # Plot the pairs with error bars
    colors = sns.color_palette("husl", len(paired_models))
    for i, (base, scores) in enumerate(paired_models.items()):
        # Plot the line connecting means
        plt.plot([0, 1], [scores['sft_score'], scores['dpo_score']], '-', color=colors[i], alpha=0.7)

        # Plot points with error bars
        plt.errorbar(0, scores['sft_score'], yerr=scores['sft_std'],
                    fmt='o', color=colors[i], capsize=5, label=base)
        plt.errorbar(1, scores['dpo_score'], yerr=scores['dpo_std'],
                    fmt='o', color=colors[i], capsize=5)

    # Add mean values with error bars
    mean_sft = np.mean([scores['sft_score'] for scores in paired_models.values()])
    mean_dpo = np.mean([scores['dpo_score'] for scores in paired_models.values()])
    std_sft = np.mean([scores['sft_std'] for scores in paired_models.values()])
    std_dpo = np.mean([scores['dpo_std'] for scores in paired_models.values()])

    plt.plot([0, 1], [mean_sft, mean_dpo], '-', color='lightslategray', linewidth=2)
    plt.errorbar(0, mean_sft, yerr=std_sft, fmt='P', color='lightslategray', capsize=5,
                markersize=8, linewidth=2, label='Mean')
    plt.errorbar(1, mean_dpo, yerr=std_dpo, fmt='P', color='lightslategray', capsize=5,
                markersize=8, linewidth=2)

    # Add labels and formatting
    # plt.title('Change in Consequentialism Score: SFT $\\rightarrow$ DPO', fontsize=14)
    plt.xticks([0, 1], ['SFT', 'DPO'], fontsize=12)
    plt.ylabel('Consequentialism', fontsize=12)
    plt.grid(axis='y', linestyle='-', alpha=0.3)

    # Add p-value annotation
    # sig_text = f"p = {p_val_one_tailed:.3f}" + (" (significant)" if p_val_one_tailed < 0.05 else "")
    # plt.annotate(sig_text + f"\nCohen's d = {cohen_d:.2f}",
    #             xy=(0.5, 0.05), xycoords='axes fraction',
    #             ha='center', fontsize=11,
    #             bbox=dict(boxstyle='round', fc='white', ec='gray'))

    # Add legend with smaller font and place it outside the plot
    plt.legend( loc='upper center', fontsize=9.3, ncols=4, bbox_to_anchor=(0.5, 1.1))
    plt.tight_layout()
    plt.savefig(f"{FIGURE_SAVE_DIR}/sft_to_dpo_means.pdf", dpi=300, bbox_inches='tight')

    return {
        'p_value': p_val_one_tailed,
        'statistic': t_stat,
        'effect_size': cohen_d,
        'means': (mean_dpo, mean_non_dpo)
    }



# Hypothesis 3: Instruction-tuned models are not better at moral reasoning than SFT models
def test_hypothesis_3(df):
    # Define SFT models (those with 'SFT' in name) vs instruction-tuned models (RLHF/DPO/RLVR)
    df['is_sft_only'] = df['model_name'].str.contains('SFT') & ~df['model_name'].str.contains('DPO|RLHF|RLVR')
    df['is_instruction_tuned'] = df['model_name'].str.contains('DPO|RLHF|RLVR')

    # Only include models that are clearly in one category or the other
    relevant_models = df[df['is_sft_only'] | df['is_instruction_tuned']]

    if len(relevant_models) > 0:
        # Aggregate data by model to get mean quality score per model
        model_data = relevant_models.groupby('model_name').agg({
            'is_instruction_tuned': 'first',
            'quality_score': 'mean'
        }).reset_index()

        # Run t-test
        instr_models = model_data[model_data['is_instruction_tuned']]['quality_score']
        sft_models = model_data[~model_data['is_instruction_tuned']]['quality_score']

        if len(instr_models) > 0 and len(sft_models) > 0:
            t_stat, p_val = stats.ttest_ind(instr_models, sft_models, equal_var=False)

            # Effect size (Cohen's d)
            mean1, mean2 = instr_models.mean(), sft_models.mean()
            std1, std2 = instr_models.std(), sft_models.std()
            pooled_std = np.sqrt(((len(instr_models) - 1) * std1**2 + (len(sft_models) - 1) * std2**2) /
                                (len(instr_models) + len(sft_models) - 2))
            cohen_d = (mean1 - mean2) / pooled_std

            # Create boxplot
            plt.figure(figsize=(6, 4))
            sns.boxplot(x='is_instruction_tuned', y='quality_score', data=relevant_models)
            # plt.title('Quality Score by Training Type')
            # plt.savefig('hypothesis3_boxplot.png')

            return {
                'p_value': p_val,
                'statistic': t_stat,
                'effect_size': cohen_d,
                'means': (mean1, mean2)
            }

    # If there are not enough models in both categories
    return {
        'p_value': 1.0,
        'statistic': 0.0,
        'error': 'Not enough models in both categories for comparison'
    }



# Hypothesis 4: There is no relationship between a model's performance in mathematical reasoning and its performance in moral reasoning
def test_hypothesis_4(df):
    # Aggregate data by model to get mean scores per model
    model_data = df.groupby('model_name').agg({
        'gsm8k_score': 'first',
        'quality_score': 'mean'
    }).reset_index()

    # Drop rows with NaN values in either column
    model_data = model_data.dropna(subset=['gsm8k_score', 'quality_score'])

    # Calculate Pearson correlation
    corr, p_val = stats.pearsonr(model_data['gsm8k_score'], model_data['quality_score'])

    # Create scatter plot with regression line
    plt.figure(figsize=(6, 4))
    sns.regplot(x='gsm8k_score', y='quality_score', data=model_data)
    # plt.title('Moral Reasoning Quality vs Mathematical Reasoning')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('GSM8K Score', fontsize=12)
    plt.ylabel('Moral Reasoning Quality', fontsize=12)
    # plt.ylim(0, 1)
    plt.grid(True, linestyle='-', alpha=0.3)
    # plt.savefig('hypothesis4_scatter.png')
    plt.savefig(f"{FIGURE_SAVE_DIR}/gsm8k_vs_moral_quality.pdf", dpi=300, bbox_inches='tight')

    return {
        'p_value': p_val,
        'statistic': corr,
        'effect_size': corr  # For correlation, r is the effect size
    }



# Hypothesis 5: There is no relationship between a model's performance in mathematical reasoning and its proportion of consequentialist rationales
def test_hypothesis_5(df):
    # Aggregate data by model to get mean scores per model
    model_data = df.groupby('model_name').agg({
        'gsm8k_score': 'first',
        'consequentialism_score': 'mean'
    }).reset_index()

    # Drop rows with NaN values in either column
    model_data = model_data.dropna(subset=['gsm8k_score', 'consequentialism_score'])

    # Calculate Pearson correlation
    corr, p_val = stats.pearsonr(model_data['gsm8k_score'], model_data['consequentialism_score'])

    # Create scatter plot with regression line
    plt.figure(figsize=(6, 4))
    sns.regplot(x='gsm8k_score', y='consequentialism_score', data=model_data)
    # plt.title('Consequentialist Responses vs Mathematical Reasoning')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('GSM8K Score', fontsize=12)
    plt.ylabel('Consequentialism', fontsize=12)
    # plt.ylim(0, 1)
    plt.grid(True, linestyle='-', alpha=0.3)
    # plt.savefig('hypothesis5_scatter.png')
    plt.savefig(f"{FIGURE_SAVE_DIR}/gsm8k_vs_consequentialism.pdf", dpi=300, bbox_inches='tight')

    return {
        'p_value': p_val,
        'statistic': corr,
        'effect_size': corr  # For correlation, r is the effect size
    }



# Hypothesis 6: There is no relationship between a model's degree of social bias and its proportion of deontological rationales
def test_hypothesis_6(df):
    # Aggregate data by model to get mean scores per model
    model_data = df.groupby('model_name').agg({
        'bbq_score': 'first',
        'deontology_score': 'mean'
    }).reset_index()

    # Drop rows with NaN values in either column
    model_data = model_data.dropna(subset=['bbq_score', 'deontology_score'])

    # Calculate Pearson correlation
    corr, p_val = stats.pearsonr(model_data['bbq_score'], model_data['deontology_score'])

    # Create scatter plot with regression line
    plt.figure(figsize=(6, 4))
    sns.regplot(x='bbq_score', y='deontology_score', data=model_data)
    # plt.title('Consequentialist Responses vs Mathematical Reasoning')
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('BBQ Accuracy', fontsize=12)
    plt.ylabel('Deontology', fontsize=12)
    plt.ylim(0.1, 0.7)
    plt.grid(True, linestyle='-', alpha=0.3)
    # plt.savefig('hypothesis5_scatter.png')
    plt.savefig(f"{FIGURE_SAVE_DIR}/bbq_bias_vs_deontology.pdf", dpi=300, bbox_inches='tight')

    return {
        'p_value': p_val,
        'statistic': corr,
        'effect_size': corr  # For correlation, r is the effect size
    }

# Hypothesis 7: There is no relationship between a model's acceptability and its proportion of deontological rationales
def test_hypothesis_7(df):
    # Aggregate data by model to get mean scores per model
    model_data = df.groupby('model_name').agg({
        'acceptability': 'mean',
        'deontology_score': 'mean',
    }).reset_index()

    # Drop rows with NaN values in either column
    model_data = model_data.dropna(subset=['acceptability', 'deontology_score'])

    # Calculate Pearson correlation
    corr, p_val = stats.pearsonr(model_data['acceptability'], model_data['deontology_score'])

    # Create scatter plot with regression line
    plt.figure(figsize=(6, 4))
    sns.regplot(x='acceptability', y='deontology_score', data=model_data)
    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Acceptability (1 - bias)', fontsize=12)
    plt.ylabel('Deontology', fontsize=12)
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.savefig(f"{FIGURE_SAVE_DIR}/acceptability_vs_deontology.pdf", dpi=300, bbox_inches='tight')

    return {
        'p_value': p_val,
        'statistic': corr,
        'effect_size': corr  # For correlation, r is the effect size
    }


# Hypothesis 8: There is no change in the proportion of deontological rationales over time
def test_hypothesis_8(df):
    # Aggregate data by model to get mean scores and release date per model
    model_data = df.groupby('model_name').agg({
        'release_date': 'first',
        'consequentialism_score': 'mean'
    }).reset_index()

    # Drop rows with NaN values in either column
    model_data = model_data.dropna(subset=['release_date', 'consequentialism_score'])

    # Convert release_date to datetime if it's not already
    model_data['release_date'] = pd.to_datetime(model_data['release_date'])

    # Create a numeric version of the date for correlation analysis
    model_data['release_ordinal'] = model_data['release_date'].map(lambda x: x.toordinal())

    # Calculate Pearson correlation
    corr, p_val = stats.pearsonr(model_data['release_ordinal'], model_data['consequentialism_score'])

    # Create scatter plot with regression line
    plt.figure(figsize=(6, 4))
    sns.regplot(x='release_ordinal', y='consequentialism_score', data=model_data)

    ax = plt.gca()
    x_ticks = ax.get_xticks()
    ax.set_xticklabels([pd.Timestamp.fromordinal(int(tick)).strftime('%m-%Y') for tick in x_ticks])
    # plt.tick_params(axis='x', rotation=30)  # Rotate x labels so they don't overlap

    plt.tick_params(axis='both', which='major', labelsize=12)
    plt.xlabel('Release Date', fontsize=12)
    plt.ylabel('Consequentialism', fontsize=12)
    plt.grid(True, linestyle='-', alpha=0.3)
    plt.savefig(f"{FIGURE_SAVE_DIR}/consequentialism_over_time.pdf", dpi=300, bbox_inches='tight')

    return {
        'p_value': p_val,
        'statistic': corr,
        'effect_size': corr  # For correlation, r is the effect size
    }