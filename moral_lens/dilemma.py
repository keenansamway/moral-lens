import asyncio
import os
from pathlib import Path

import pandas as pd

from typing import List, Optional

from moral_lens.models import ModelFactory, load_model_config
from moral_lens.data_models import ChatMessage, LLMResponse, MessageRole, Prompt
from moral_lens.data_models import Provider
from moral_lens.utils import load_yaml_file, fuzzy_match_decisions, parse_reasoning_and_decision
from moral_lens.config import ModelConfig, PathConfig

from dataclasses import dataclass
from typing import Optional


def is_valid_response(response_obj: LLMResponse) -> bool:
    """Simple validation function to check if the response exists and is not empty."""
    content = response_obj.content
    two_choices = response_obj.two_choices

    if content is None or content.strip() == "":
        return False

    content = content.replace("'", "").replace(",", "").lower()
    refusal_phrases = [
        "im sorry",
        "i am sorry",
        "i cant assist",
        "i cant help",
    ]
    if any(phrase in content for phrase in refusal_phrases):
        return False

    if two_choices is not None:
        reasoning, decision = parse_reasoning_and_decision(content)
        decision = fuzzy_match_decisions(decision, two_choices)
        if len(decision) == 0:
            return False

    return True

class DilemmaRunner:
    def __init__(
        self,
        model_id: str,
        decision_run_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        results_dir: str = "data/results",
        choices_filename: str = "choices.csv",
        override_decision_temperature: Optional[float] = None,
    ):
        # Setup the path configuration
        path_config = PathConfig(results_dir=results_dir)

        # Setup the model configuration
        self.model_cfg = load_model_config(model_id)
        if override_decision_temperature is not None:
            self.model_cfg.override_temperature(override_decision_temperature)

        # Set the choices filename
        choices_path = path_config.get_file(choices_filename)
        self.choices_df = pd.read_csv(choices_path)

        # Set the output file path
        self.output_file = path_config.get_response_output_file(
            decision_save_id=self.model_cfg.save_id,
            run_name=decision_run_name,
            dataset_name=dataset_name,
        )
        if self.output_file.exists():
            print(f"Output file already exists at {self.output_file}. Use `overwrite=True` in .run() to overwrite it.")

        # Load the dilemma template and system prompts
        dilemma_template_path = path_config.get_file("prompts_dilemma.yaml")
        template_yaml_obj = load_yaml_file(dilemma_template_path)

        self.dilemma_template = template_yaml_obj.get("dilemma_template", None)
        if self.dilemma_template is None:
            raise ValueError(f"dilemma_template not found in {dilemma_template_path}.")

        system_prompts = template_yaml_obj.get("system_prompts", None)
        if system_prompts is None:
            raise ValueError(f"system_prompts not found in {dilemma_template_path}.")
        self.system_prompt_template = system_prompts['detailed']

        self.data: Optional[pd.DataFrame] = None


    async def run(
        self, overwrite: bool = False, batch_size: int = 1
    ) -> None:
        # Check if the output file exists
        file_exists = self.output_file.exists()

        if file_exists and not overwrite:
            # Load existing data
            self.load_data()

            # Find rows with empty responses
            empty_rows = self.data[self.data['raw_response'].str.len() == 0].index.to_list()
            if not empty_rows:
                print(f"No empty responses found in {self.output_file}. Use `overwrite=True` to rerun all.")
                return

            rows_to_process = empty_rows

        elif file_exists and overwrite:
            # Load existing data and process all rows
            self.load_data()
            rows_to_process = list(range(len(self.data)))

        else:
            # File doesn't exist, create from scratch
            two_choices = self.choices_df['two_choices'].str.split('; ').to_list()

            # Initialize the dataframe with all rows
            self.data = pd.DataFrame({
                "id": self.choices_df['id'],
                "system_prompt": [self.system_prompt_template] * len(self.choices_df),
                "dilemma_prompt": [""] * len(self.choices_df),
                "two_choices": self.choices_df['two_choices'],
                "two_choices_set": self.choices_df['two_choices_unordered_set'].to_list(),
                "phenomenon_category": self.choices_df['phenomenon_category'].to_list(),
                "category1": self.choices_df['category1'].to_list(),
                "category2": self.choices_df['category2'].to_list(),
                "decision_model_id": self.model_cfg.model_id,
                "decision_temperature": self.model_cfg.temperature,
                "attempt_count": [0] * len(self.choices_df),
                "thinking": [""] * len(self.choices_df),
                "raw_response": [""] * len(self.choices_df),
            })

            rows_to_process = list(range(len(self.data)))

        # Prepare prompts for rows to process
        prompts = []
        processed_indices = []

        for idx in rows_to_process:
            if file_exists:
                # Use existing prompts from loaded data
                system_prompt = self.data.loc[idx, 'system_prompt']
                dilemma_prompt = self.data.loc[idx, 'dilemma_prompt']
            else:
                # Create new prompts
                choice1, choice2 = two_choices[idx]
                if not isinstance(choice1, str) or not isinstance(choice2, str):
                    print(f"Skipping invalid choices at index {idx}: {two_choices[idx]}")
                    continue

                system_prompt = self.system_prompt_template
                dilemma_prompt = self.dilemma_template.format(choice1=choice1, choice2=choice2)

                # Update the dataframe with the prompts
                self.data.loc[idx, 'system_prompt'] = system_prompt
                self.data.loc[idx, 'dilemma_prompt'] = dilemma_prompt

            messages = [
                ChatMessage(role=MessageRole.system, content=system_prompt),
                ChatMessage(role=MessageRole.user, content=dilemma_prompt)
            ]

            prompts.append(Prompt(messages=messages))
            processed_indices.append(idx)

        if not prompts:
            print("No prompts to process.")
            return

        # Get model and process prompts
        model = ModelFactory.get_model(model=self.model_cfg)

        if self.model_cfg.provider == Provider.huggingface:
            responses = model.ask_batch_with_retry(prompts, validation_fn=is_valid_response, batch_size=batch_size)
            model.unload()
        else:
            responses = await model.ask_async_with_retry(prompts, validation_fn=is_valid_response)

        # Update the dataframe with responses
        for i, response in enumerate(responses):
            idx = processed_indices[i]
            attempts = response.attempts if response else 0
            thinking = response.thinking_content if response else ""
            content = response.content if response else ""

            # Update rows
            self.data.loc[idx, 'attempt_count'] += attempts
            self.data.loc[idx, 'thinking'] = thinking
            self.data.loc[idx, 'raw_response'] = content

        # Sort the dataframe by ID and save
        self.data = self.data.sort_values(by='id')
        os.makedirs(self.output_file.parent, exist_ok=True)
        self.data.to_csv(self.output_file, index=False)
        print(f"[INFO] Responses {'updated' if file_exists else 'saved'} to {self.output_file}.")

        self.process()


    def process(self) -> None:
        """
        Process the raw responses to extract reasoning and decisions.
        This function should be called after run() to parse the model outputs.
        """
        # Check if output file exists
        if not self.output_file.exists():
            raise FileNotFoundError(f"Response output file not found at {self.output_file}. Please run the dilemma analysis first.")

        # Load existing data
        self.load_data()

        # Extract necessary data
        responses = self.data['raw_response'].to_list()
        two_choices = self.data['two_choices'].str.split('; ').to_list()
        category1s = self.data['category1'].to_list()
        category2s = self.data['category2'].to_list()

        # Initialize results lists
        reasonings = []
        decisions = []
        decision_categories = []

        # Process each response
        for i, (response, choices) in enumerate(zip(responses, two_choices)):
            # Skip empty or None responses
            if not response or response == "":
                reasonings.append("")
                decisions.append("")
                decision_categories.append("")
                continue

            # Parse the response
            try:
                reasoning, decision = parse_reasoning_and_decision(response)
                decision = fuzzy_match_decisions(decision, choices)

                reasonings.append(reasoning)
                decisions.append(decision)

                # Determine the decision category
                if decision == choices[0]:
                    decision_categories.append(category1s[i])
                elif decision == choices[1]:
                    decision_categories.append(category2s[i])
                else:
                    decision_categories.append("")
            except Exception as e:
                print(f"Error processing response at index {i}: {e}")
                reasonings.append("")
                decisions.append("")
                decision_categories.append("")

        # Update the dataframe with processed results
        self.data['reasoning'] = reasonings
        self.data['decision'] = decisions
        self.data['decision_category'] = decision_categories

        # Sort by ID (to maintain consistency with run() function)
        self.data = self.data.sort_values(by='id')

        # Save the updated dataframe
        os.makedirs(self.output_file.parent, exist_ok=True)
        self.data.to_csv(self.output_file, index=False)
        print(f"[INFO] Processed responses saved to {self.output_file}.")


    def load_data(self):
        if not self.output_file.exists():
            print(f"[INFO] Response output file not found: {self.output_file}")
        self.data = pd.read_csv(self.output_file, keep_default_na=False)


def put_submodule_in_path(submodule_name: str) -> None:
    """
    Adds the specified submodule to the system path.
    """
    import sys
    import os
    from pathlib import Path

    project_root = Path(os.getcwd())
    submodule_path = str(project_root / submodule_name)
    if submodule_path not in sys.path:
        sys.path.append(submodule_path)

if __name__ == "__main__":
    put_submodule_in_path("moral_lens")

    runner = DilemmaRunner(
        model_id="gemini-2.0-flash-001",
        choices_filename="choices_1pct.csv"
    )
    asyncio.run(runner.run())
    runner.process()