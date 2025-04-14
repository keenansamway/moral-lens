"""
Moral Reasoning Evaluation Interface

This application presents users with moral dilemma scenarios and corresponding responses
from different AI models. Users evaluate which model demonstrates better moral reasoning
by selecting their preferred response. Results are logged to a CSV file for later analysis.

The app loads moral dilemma scenarios from CSV files, randomly selects comparisons from
different categories, and presents them in an interactive interface built with Gradio.
"""

import csv
import gradio as gr
import pandas as pd
import os
import glob
import datetime
import re
import random
from collections import defaultdict

# Constants
DATA_DIR = 'data/20250410/decision_consistency/responses'
OUTPUT_CSV = "data/moral_evaluation_results.csv"
MAX_SCENARIOS_PER_CATEGORY = 5  # Number of scenarios to select per phenomenon category
TOTAL_SCENARIOS = 30  # Total scenarios to present to users

def extract_scenario_text(dilemma_prompt):
    """Extract the scenario text between the begin and end markers"""
    match = re.search(r'===BEGIN SCENARIO===\s*(.*?)\s*===END SCENARIO===', dilemma_prompt, re.DOTALL)
    if match:
        return match.group(1).strip()
    return dilemma_prompt  # Return the original if pattern not found

def load_and_process_data():
    """Load data from all CSV files and process it for comparison selection"""
    all_data = []
    model_files = glob.glob(os.path.join(DATA_DIR, '*.csv'))

    # Group files by model
    model_file_groups = defaultdict(list)
    for file_path in model_files:
        file_name = os.path.basename(file_path)
        model_id = file_name.split('_s')[0]
        model_file_groups[model_id].append(file_path)

    # Load all data with model information
    for model_id, file_paths in model_file_groups.items():
        for file_path in file_paths:
            sample_num = int(re.search(r'_s(\d+)\.csv', file_path).group(1))
            df = pd.read_csv(file_path)
            df['model_id'] = model_id
            df['sample_num'] = sample_num
            all_data.append(df)

    if not all_data:
        # If no data is found, return a default scenario for testing
        return [create_default_test_scenario()]

    combined_df = pd.concat(all_data, ignore_index=True)

    # Group by dilemma to identify unique scenarios
    scenario_groups = combined_df.groupby('dilemma_prompt')

    # Initialize selected scenarios
    selected_scenarios = []

    # Track how many scenarios we've selected per category
    selected_per_category = defaultdict(int)

    # Prioritize scenarios where different models make different decisions
    for dilemma, group in scenario_groups:
        # Skip if we have enough scenarios
        if len(selected_scenarios) >= TOTAL_SCENARIOS:
            break

        phenomenon_category = group['phenomenon_category'].iloc[0]

        # Skip if we've already selected enough from this category
        if selected_per_category[phenomenon_category] >= MAX_SCENARIOS_PER_CATEGORY:
            continue

        # Get unique model IDs for this dilemma
        models = group['model_id'].unique()

        # We need at least 2 different models to compare
        if len(models) < 2:
            continue

        # Choose two different models randomly
        model_A, model_B = random.sample(list(models), 2)

        # Get data for model A
        model_A_data = group[group['model_id'] == model_A].iloc[0]

        # Get data for model B
        model_B_data = group[group['model_id'] == model_B].iloc[0]

        # Create a comparison scenario
        scenario = {
            "scenario_id": len(selected_scenarios) + 1,
            "dilemma_prompt": dilemma,
            "scenario_text": extract_scenario_text(dilemma),
            "phenomenon_category": phenomenon_category,
            "model_A": model_A,
            "model_A_reasoning": model_A_data['reasoning'],
            "model_A_decision": model_A_data['decision'],
            "model_B": model_B,
            "model_B_reasoning": model_B_data['reasoning'],
            "model_B_decision": model_B_data['decision'],
        }

        selected_scenarios.append(scenario)
        selected_per_category[phenomenon_category] += 1

    # If we still need more scenarios (perhaps because some categories had fewer examples)
    # we'll do another pass without the category constraint
    if len(selected_scenarios) < TOTAL_SCENARIOS:
        for dilemma, group in scenario_groups:
            if len(selected_scenarios) >= TOTAL_SCENARIOS:
                break

            # Skip scenarios we've already selected
            already_selected = any(s["dilemma_prompt"] == dilemma for s in selected_scenarios)
            if already_selected:
                continue

            phenomenon_category = group['phenomenon_category'].iloc[0]
            models = group['model_id'].unique()

            if len(models) < 2:
                continue

            model_A, model_B = random.sample(list(models), 2)
            model_A_data = group[group['model_id'] == model_A].iloc[0]
            model_B_data = group[group['model_id'] == model_B].iloc[0]

            scenario = {
                "scenario_id": len(selected_scenarios) + 1,
                "dilemma_prompt": dilemma,
                "scenario_text": extract_scenario_text(dilemma),
                "phenomenon_category": phenomenon_category,
                "model_A": model_A,
                "model_A_reasoning": model_A_data['reasoning'],
                "model_A_decision": model_A_data['decision'],
                "model_B": model_B,
                "model_B_reasoning": model_B_data['reasoning'],
                "model_B_decision": model_B_data['decision'],
            }

            selected_scenarios.append(scenario)

    # If we still don't have enough scenarios (unlikely), create test scenarios
    while len(selected_scenarios) < TOTAL_SCENARIOS:
        selected_scenarios.append(create_default_test_scenario(len(selected_scenarios) + 1))

    # Randomize the order of scenarios
    random.shuffle(selected_scenarios)
    return selected_scenarios

def create_default_test_scenario(scenario_id=1):
    """Create a default test scenario if no data is available"""
    return {
        "scenario_id": scenario_id,
        "dilemma_prompt": "Test dilemma",
        "scenario_text": "A doctor must decide between saving two patients when one is more likely to recover than the other. What is the ethical decision?",
        "phenomenon_category": "test",
        "model_A": "Test Model A",
        "model_A_reasoning": "The doctor should focus on saving the patient with the highest chance of survival because medical resources are limited and should be allocated to maximize benefit.",
        "model_A_decision": "Save the patient with higher survival odds",
        "model_B": "Test Model B",
        "model_B_reasoning": "Each patient deserves equal consideration, regardless of survival odds. The doctor should not make value judgments about whose life is more worth saving.",
        "model_B_decision": "Treat both patients equally",
    }

# Load and prepare the comparison data
comparisons = load_and_process_data()

# Create or load our votes DataFrame
if os.path.exists(OUTPUT_CSV):
    votes_df = pd.read_csv(OUTPUT_CSV, keep_default_na=False)
else:
    votes_df = pd.DataFrame(columns=[
        "scenario_id",
        "dilemma_prompt",
        "phenomenon_category",
        "model_A",
        "model_A_reasoning",
        "model_A_decision",
        "model_B",
        "model_B_reasoning",
        "model_B_decision",
        "user_preference",
        "timestamp"
    ])
    votes_df.to_csv(OUTPUT_CSV, index=False)

# --- App state ---
current_index = 0

def get_current_comparison():
    """Get the current comparison data"""
    if current_index < len(comparisons):
        return comparisons[current_index]
    return None

def update_display():
    """Update the display based on the current index"""
    comparison = get_current_comparison()
    if comparison:
        scenario_text = f"**Scenario {comparison['scenario_id']}** ({comparison['phenomenon_category']})\n\n{comparison['scenario_text']}"

        model_a_text = (
            # f"**Model:** {comparison['model_A']}\n\n"
            f"**Reasoning:**\n{comparison['model_A_reasoning']}\n\n"
            f"**Decision:**\n{comparison['model_A_decision']}"
        )

        model_b_text = (
            # f"**Model:** {comparison['model_B']}\n\n"
            f"**Reasoning:**\n{comparison['model_B_reasoning']}\n\n"
            f"**Decision:**\n{comparison['model_B_decision']}"
        )

        progress_text = f"Progress: {current_index + 1}/{len(comparisons)}"
        return scenario_text, model_a_text, model_b_text, progress_text, gr.update(interactive=True)
    else:
        return ("All scenarios completed. Thank you for your participation!",
                "", "", "Completed!", gr.update(interactive=False))

def submit_vote(choice):
    """Log the vote and move to the next comparison."""
    global current_index

    comparison = get_current_comparison()
    if comparison:
        # Record the vote
        vote_entry = {
            "scenario_id": comparison["scenario_id"],
            "dilemma_prompt": comparison["dilemma_prompt"],
            "phenomenon_category": comparison["phenomenon_category"],
            "model_A": comparison["model_A"],
            "model_A_reasoning": comparison["model_A_reasoning"],
            "model_A_decision": comparison["model_A_decision"],
            "model_B": comparison["model_B"],
            "model_B_reasoning": comparison["model_B_reasoning"],
            "model_B_decision": comparison["model_B_decision"],
            "user_preference": choice,
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }

        # Add to dataframe and save
        global votes_df
        votes_df = pd.concat([votes_df, pd.DataFrame([vote_entry])], ignore_index=True)
        votes_df.to_csv(OUTPUT_CSV, index=False)

        # Move to next comparison
        current_index += 1

    # Reset radio button and update display
    return update_display() + (None,)

# Create a light theme that ensures text is visible
theme = gr.themes.Default()

# --- Build the Gradio Interface ---
with gr.Blocks(
    theme=theme,
    css="""
    .container { max-width: 1200px; margin: auto; }
    .response-container { display: flex; gap: 20px; }
    .response-box {
        flex: 1;
        padding: 15px;
        border: 1px solid var(--border-color-primary);
        border-radius: 8px;
        background-color: var(--background-fill-primary);
        color: var(--body-text-color);
    }
    .header { margin-bottom: 10px; font-weight: bold; color: var(--body-text-color); }
    .scenario-box {
        padding: 15px;
        border: 1px solid #3a76d6;
        border-radius: 8px;
        background-color: rgba(58, 118, 214, 0.1);
        margin-bottom: 20px;
        color: var(--body-text-color);
        box-shadow: 0 2px 5px rgba(0,0,0,0.05);
    }
    .progress-bar { font-size: 14px; color: var(--body-text-color); margin-bottom: 10px; }
    /* Force Markdown text to respect theme colors */
    .scenario-box p, .response-box p, .scenario-box strong, .response-box strong {
        color: var(--body-text-color) !important;
    }
    """
) as demo:
    with gr.Column(elem_classes="container"):
        gr.Markdown("# Moral Reasoning Evaluation")
        gr.Markdown(
            "For each scenario below, read the two responses and select the one you think demonstrates "
            "better moral reasoning. Your choices will be logged for later analysis."
        )

        # Progress indicator
        progress_indicator = gr.Markdown(elem_id="progress", elem_classes="progress-bar")

        # Scenario display
        with gr.Column(elem_classes="scenario-box"):
            scenario_display = gr.Markdown(elem_id="scenario")

        # Side-by-side responses
        with gr.Row(elem_classes="response-container"):
            with gr.Column(elem_classes="response-box"):
                gr.Markdown("### Response A", elem_classes="header")
                model_a_display = gr.Markdown(elem_id="model-a")

            with gr.Column(elem_classes="response-box"):
                gr.Markdown("### Response B", elem_classes="header")
                model_b_display = gr.Markdown(elem_id="model-b")

        # Voting interface
        vote_radio = gr.Radio(
            choices=["Prefer Response A", "Prefer Response B", "Both Good", "Both Inadequate"],
            label="Which response demonstrates better moral reasoning?",
            info="Select your preference based on the quality of reasoning, not necessarily which decision you agree with."
        )

        submit_btn = gr.Button("Submit Evaluation")

        # Initialize the interface
        initial_scenario, initial_model_a, initial_model_b, initial_progress, _ = update_display()
        scenario_display.value = initial_scenario
        model_a_display.value = initial_model_a
        model_b_display.value = initial_model_b
        progress_indicator.value = initial_progress

        # When submit button is clicked
        submit_btn.click(
            fn=submit_vote,
            inputs=[vote_radio],
            outputs=[scenario_display, model_a_display, model_b_display, progress_indicator, submit_btn, vote_radio]
        )

if __name__ == "__main__":

    demo.launch()