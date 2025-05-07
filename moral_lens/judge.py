import os
from pathlib import Path
import re
import asyncio

import pandas as pd

from typing import Any, Optional

from moral_lens.models import ModelFactory, load_model_config
from moral_lens.data_models import ChatMessage, MessageRole, Prompt
from moral_lens.config import ModelConfig, PathConfig
from moral_lens.utils import load_yaml_file, parse_keyword_text

class JudgeRunner:
    def __init__(
        self,
        decision_model_id: str,
        judge_model_id: str,
        decision_run_name: Optional[str] = None,
        judge_run_name: Optional[str] = None,
        judge_cot: bool = False,
        dataset_name: Optional[str] = None,
        results_dir: str = "data/results",
        validation_fn: Optional[Any] = None,
        override_judge_temperature: Optional[float] = None,
    ):
        self.validation_fn = validation_fn

        # Setup the path configuration
        path_config = PathConfig(results_dir=results_dir)

        # Setup the model configurations
        self.decision_model_cfg = load_model_config(decision_model_id)
        self.judge_model_cfg = load_model_config(judge_model_id)
        if override_judge_temperature is not None:
            self.judge_model_cfg.override_temperature(override_judge_temperature)

        # Set the output file paths
        self.response_output_file = path_config.get_response_output_file(
            decision_save_id=self.decision_model_cfg.save_id,
            run_name=decision_run_name,
            dataset_name=dataset_name,
        )

        self.judge_output_file = path_config.get_judge_output_file(
            decision_save_id=self.decision_model_cfg.save_id,
            judge_save_id=self.judge_model_cfg.save_id,
            run_name=judge_run_name,
            dataset_name=dataset_name,
        )

        # Load the judge template and system prompts
        judge_template_path = path_config.get_file("prompts_judge.yaml")
        template_yaml_obj = load_yaml_file(judge_template_path)

        self.sequential_rationales_template = template_yaml_obj.get("sequential_rationales")
        self.rationales_template = template_yaml_obj.get(
            "rationales" + ("_cot" if judge_cot else "")
        )
        self.quality_template = template_yaml_obj.get(
            "quality" + ("_cot" if judge_cot else "")
        )

        self.rationales = template_yaml_obj.get("rationales_list")

        # if not self.rational_classification_template or not self.quality_evaluation_template:
        #     raise ValueError("Missing templates in judge YAML.")
        # if not self.system_prompts or not self.rationales:
        #     raise ValueError("Missing system prompts or rationales.")

        self.data: Optional[pd.DataFrame] = None


    async def run_sequential_rationales(self, overwrite: bool = False) -> None:
        """Run the judge model to evaluate rationales for the decisions."""
        if not self.response_output_file.exists():
            raise FileNotFoundError(f"Decision output file not found at {self.response_output_file}. Please run the decision model first.")

        # Load decision data
        self.load_data()

        file_exists = self.judge_output_file.exists()

        if file_exists and not overwrite:

            # Check if the necessary column exists
            if "raw_responses_sequential_rationales" not in self.data.columns:
                # Initialize the column if it doesn't exist
                self.data["raw_responses_sequential_rationales"] = [""] * len(self.data)

            # Find rows with empty rationale responses
            empty_rows = self.data[self.data['raw_responses_sequential_rationales'].str.len() == 0].index.to_list()
            # empty_rows = self.data[self.data['rationales'].str.len() == 0].index.to_list()
            if not empty_rows:
                print(f"No empty sequential rationale responses found in {self.judge_output_file}. Use `overwrite=True` to rerun all.")
                return

            rows_to_process = empty_rows
        else:
            # First time running or overwrite=True, process all rows
            self.data["raw_responses_sequential_rationales"] = [""] * len(self.data)
            self.data["sequential_rationales"] = [""] * len(self.data)
            rows_to_process = list(range(len(self.data)))

        # Prepare prompts for rows to process
        rationales_prompts = []
        processed_indices = []

        for idx in rows_to_process:
            row = self.data.iloc[idx]

            # Skip rows with empty reasoning
            if not row['reasoning'] or pd.isna(row['reasoning']) or row['reasoning'].strip() == "":
                continue

            # scenario = re.search(
            #     r'===BEGIN SCENARIO===\s*(.*?)\s*===END SCENARIO===',
            #     row['dilemma_prompt'], re.DOTALL
            # ).group(1).strip()

            scenario = row['dilemma_prompt'].split("Provide")[0].strip()

            reasoning = row['reasoning']
            rationale_str = "\n".join([f'- "{k}": {v}' for k, v in self.rationales.items()])

            rationale_prompt = self.sequential_rationales_template['classification_template'].format(
                reasoning=reasoning, rationales=rationale_str
            )

            rationales_prompts.append(Prompt([
                ChatMessage(role=MessageRole.system, content=self.sequential_rationales_template['system_prompt']),
                ChatMessage(role=MessageRole.user, content=rationale_prompt)
            ]))

            processed_indices.append(idx)

        if not rationales_prompts:
            print("No prompts to process.")
            return

        # Get model and process prompts
        model = ModelFactory.get_model(model=self.judge_model_cfg)
        rationales_responses = await model.ask_async_with_retry(rationales_prompts, validation_fn=self.validation_fn)

        # Update the dataframe with responses
        for i, response in enumerate(rationales_responses):
            idx = processed_indices[i]
            content = response.content if response else ""

            # Update rows
            self.data.loc[idx, 'raw_responses_sequential_rationales'] = content

            if content == "":
                self.data.loc[idx, 'sequential_rationales'] = ""

        # Save the updated dataframe
        os.makedirs(self.judge_output_file.parent, exist_ok=True)
        self.save_data()

        action = "updated" if file_exists and not overwrite else "saved"
        print(f"[INFO] Judge sequential rationales output {action} to {self.judge_output_file}")

        self.process_sequential_rationales()


    async def run_rationales(self, overwrite: bool = False) -> None:
        """Run the judge model to evaluate rationales for the decisions."""
        if not self.response_output_file.exists():
            raise FileNotFoundError(f"Decision output file not found at {self.response_output_file}. Please run the decision model first.")

        # Load decision data
        self.load_data()

        file_exists = self.judge_output_file.exists()

        if file_exists and not overwrite:

            # Check if the necessary column exists
            if "raw_responses_rationales" not in self.data.columns:
                # Initialize the column if it doesn't exist
                self.data["raw_responses_rationales"] = [""] * len(self.data)

            # Find rows with empty rationale responses
            empty_rows = self.data[self.data['raw_responses_rationales'].str.len() == 0].index.to_list()
            # empty_rows = self.data[self.data['rationales'].str.len() == 0].index.to_list()
            if not empty_rows:
                print(f"No empty rationale responses found in {self.judge_output_file}. Use `overwrite=True` to rerun all.")
                return

            rows_to_process = empty_rows
        else:
            # First time running or overwrite=True, process all rows
            self.data["raw_responses_rationales"] = [""] * len(self.data)
            self.data["rationales"] = [""] * len(self.data)
            rows_to_process = list(range(len(self.data)))

        # Prepare prompts for rows to process
        rationales_prompts = []
        processed_indices = []

        for idx in rows_to_process:
            row = self.data.iloc[idx]

            # Skip rows with empty reasoning
            if not row['reasoning'] or pd.isna(row['reasoning']) or row['reasoning'].strip() == "":
                continue

            # scenario = re.search(
            #     r'===BEGIN SCENARIO===\s*(.*?)\s*===END SCENARIO===',
            #     row['dilemma_prompt'], re.DOTALL
            # ).group(1).strip()

            scenario = row['dilemma_prompt'].split("Provide")[0].strip()

            reasoning = row['reasoning']
            # decision = row['decision']
            rationale_str = "\n".join([f'- "{k}": {v}' for k, v in self.rationales.items()])

            rationale_prompt = self.rationales_template['classification_template'].format(
                reasoning=reasoning, rationales=rationale_str)

            rationales_prompts.append(Prompt([
                ChatMessage(role=MessageRole.system, content=self.rationales_template['system_prompt']),
                ChatMessage(role=MessageRole.user, content=rationale_prompt)
            ]))

            processed_indices.append(idx)

        if not rationales_prompts:
            print("No prompts to process.")
            return

        # Get model and process prompts
        model = ModelFactory.get_model(model=self.judge_model_cfg)
        rationales_responses = await model.ask_async_with_retry(rationales_prompts, validation_fn=self.validation_fn)

        # Update the dataframe with responses
        for i, response in enumerate(rationales_responses):
            idx = processed_indices[i]
            content = response.content if response else ""

            # Update rows
            self.data.loc[idx, 'raw_responses_rationales'] = content

            if content == "":
                self.data.loc[idx, 'rationales'] = ""

        # Save the updated dataframe
        os.makedirs(self.judge_output_file.parent, exist_ok=True)
        self.save_data()

        action = "updated" if file_exists and not overwrite else "saved"
        print(f"[INFO] Judge rationales output {action} to {self.judge_output_file}")

        self.process_rationales()

    async def run_quality(self, overwrite: bool = False) -> None:
        """Run the judge model to evaluate quality of the decisions."""
        if not self.response_output_file.exists():
            raise FileNotFoundError(f"Decision output file not found at {self.response_output_file}. Please run the decision model first.")

        # Load decision data
        self.load_data()

        file_exists = self.judge_output_file.exists()

        if file_exists and not overwrite:
            # Load existing quality data
            judge_data = pd.read_csv(self.judge_output_file, keep_default_na=False)

            # Check if the necessary column exists
            if "raw_responses_quality" not in judge_data.columns:
                # Initialize the column if it doesn't exist
                judge_data["raw_responses_quality"] = [""] * len(judge_data)

            # Update our dataframe with the existing judge data
            self.data = judge_data

            # Find rows with empty quality responses
            empty_rows = self.data[self.data['raw_responses_quality'].str.len() == 0].index.to_list()
            if not empty_rows:
                print(f"No empty quality responses found in {self.judge_output_file}. Use `overwrite=True` to rerun all.")
                return

            rows_to_process = empty_rows
        else:
            # First time running or overwrite=True, process all rows
            self.data["raw_responses_quality"] = [""] * len(self.data)
            rows_to_process = list(range(len(self.data)))

        # Prepare prompts for rows to process
        quality_prompts = []
        processed_indices = []

        for idx in rows_to_process:
            row = self.data.iloc[idx]

            # Skip rows with empty reasoning
            if not row['reasoning'] or pd.isna(row['reasoning']) or row['reasoning'].strip() == "":
                continue

            # scenario = re.search(
            #     r'===BEGIN SCENARIO===\s*(.*?)\s*===END SCENARIO===',
            #     row['dilemma_prompt'], re.DOTALL
            # ).group(1).strip()

            scenario = row['dilemma_prompt'].split("Provide")[0].strip()

            reasoning = row['reasoning']
            decision = row['decision']

            quality_prompt = self.quality_template['classification_template'].format(
                scenario=scenario, reasoning=reasoning, decision=decision)

            quality_prompts.append(Prompt([
                ChatMessage(role=MessageRole.system, content=self.quality_template['system_prompt']),
                ChatMessage(role=MessageRole.user, content=quality_prompt)
            ]))

            processed_indices.append(idx)

        if not quality_prompts:
            print("No prompts to process.")
            return

        # Get model and process prompts
        model = ModelFactory.get_model(model=self.judge_model_cfg)
        quality_responses = await model.ask_async_with_retry(quality_prompts, validation_fn=self.validation_fn)

        # Update the dataframe with responses
        for i, response in enumerate(quality_responses):
            idx = processed_indices[i]
            content = response.content if response else ""

            # Update rows
            self.data.loc[idx, 'raw_responses_quality'] = content

        # Save the updated dataframe
        os.makedirs(self.judge_output_file.parent, exist_ok=True)
        self.save_data()

        action = "updated" if file_exists and not overwrite else "saved"
        print(f"[INFO] Judge quality output {action} to {self.judge_output_file}")

        self.process_quality()



    async def run_acceptability(self, overwrite: bool = False) -> None:
        """Run the judge model to evaluate acceptability of the decisions."""
        if not self.response_output_file.exists():
            raise FileNotFoundError(f"Decision output file not found at {self.response_output_file}. Please run the decision model first.")

        # Load decision data
        self.load_data()

        file_exists = self.judge_output_file.exists()

        if file_exists and not overwrite:
            # Load existing quality data
            judge_data = pd.read_csv(self.judge_output_file, keep_default_na=False)

            # Check if the necessary column exists
            if "raw_responses_quality" not in judge_data.columns:
                # Initialize the column if it doesn't exist
                judge_data["raw_responses_quality"] = [""] * len(judge_data)

            # Update our dataframe with the existing judge data
            self.data = judge_data

            # Find rows with empty quality responses
            empty_rows = self.data[self.data['raw_responses_quality'].str.len() == 0].index.to_list()
            if not empty_rows:
                print(f"No empty quality responses found in {self.judge_output_file}. Use `overwrite=True` to rerun all.")
                return

            rows_to_process = empty_rows
        else:
            # First time running or overwrite=True, process all rows
            self.data["raw_responses_quality"] = [""] * len(self.data)
            rows_to_process = list(range(len(self.data)))

        # Prepare prompts for rows to process
        quality_prompts = []
        processed_indices = []

        for idx in rows_to_process:
            row = self.data.iloc[idx]

            # Skip rows with empty reasoning
            if not row['reasoning'] or pd.isna(row['reasoning']) or row['reasoning'].strip() == "":
                continue

            # scenario = re.search(
            #     r'===BEGIN SCENARIO===\s*(.*?)\s*===END SCENARIO===',
            #     row['dilemma_prompt'], re.DOTALL
            # ).group(1).strip()

            scenario = row['dilemma_prompt'].split("Provide")[0].strip()

            reasoning = row['reasoning']
            decision = row['decision']

            quality_prompt = self.quality_template['classification_template'].format(
                reasoning=reasoning)

            quality_prompts.append(Prompt([
                ChatMessage(role=MessageRole.system, content=self.quality_template['system_prompt']),
                ChatMessage(role=MessageRole.user, content=quality_prompt)
            ]))

            processed_indices.append(idx)

        if not quality_prompts:
            print("No prompts to process.")
            return

        # Get model and process prompts
        model = ModelFactory.get_model(model=self.judge_model_cfg)
        quality_responses = await model.ask_async_with_retry(quality_prompts, validation_fn=self.validation_fn)

        # Update the dataframe with responses
        for i, response in enumerate(quality_responses):
            idx = processed_indices[i]
            content = response.content if response else ""

            # Update rows
            self.data.loc[idx, 'raw_responses_quality'] = content

        # Save the updated dataframe
        os.makedirs(self.judge_output_file.parent, exist_ok=True)
        self.save_data()

        action = "updated" if file_exists and not overwrite else "saved"
        print(f"[INFO] Judge quality output {action} to {self.judge_output_file}")

        self.process_quality()

    async def run(self, overwrite: bool = False, skip_quality: bool = True) -> None:
        """Run both rationales and quality evaluations based on the skip_quality parameter."""
        await self.run_rationales(overwrite=overwrite)
        if not skip_quality:
            await self.run_quality(overwrite=overwrite)

    def process_sequential_rationales(self) -> None:
        """
        Process the judge model's raw sequential responses to extract sequential rationales.
        This function should be called after running the judge model to parse the outputs.
        """
        # Check if judge output file exists
        if not self.judge_output_file.exists():
            raise FileNotFoundError(f"Judge output file not found: {self.judge_output_file}. Please run the judge model first.")

        # Load data
        self.load_data()

        # Extract raw responses
        rationales_responses = self.data['raw_responses_sequential_rationales']
        dilemma_responses = self.data['raw_response']

        # Process rationales
        processed_rationales = []
        for i, response in enumerate(rationales_responses):
            try:
                # Skip empty or None responses
                if not response or pd.isna(response) or response.strip() == "":
                    processed_rationales.append("Refusal")
                    continue

                if not dilemma_responses[i] or dilemma_responses[i] == "":
                    processed_rationales.append("Refusal")
                    continue

                # Extract rationales enclosed in square brackets
                extracted_labels = []
                pattern = r'\["([^"]+)"\]'
                matches = re.findall(pattern, response)

                # Filter out only the label-orientation pairs (not end-section markers)
                for match in matches:
                    if "end" not in match and "section" not in match:
                        extracted_labels.append(match)

                # Join the extracted labels with semicolons
                to_add = "; ".join(extracted_labels) if extracted_labels else ""

                processed_rationales.append(to_add)
            except Exception as e:
                print(f"Error processing rationale: {e}")
                processed_rationales.append("")

        # Update dataframe with processed results
        self.data["sequential_rationales"] = processed_rationales

        # Save updated dataframe
        os.makedirs(self.judge_output_file.parent, exist_ok=True)
        self.data.to_csv(self.judge_output_file, index=False)
        print(f"[INFO] Processed judge output saved to {self.judge_output_file}")

    def process_rationales(self) -> None:
        """
        Process the judge model's raw responses to extract rationales.
        This function should be called after running the judge model to parse the outputs.
        """
        # Check if judge output file exists
        if not self.judge_output_file.exists():
            raise FileNotFoundError(f"Judge output file not found: {self.judge_output_file}. Please run the judge model first.")

        # Load data
        self.load_data()

        # Extract raw responses
        rationales_responses = self.data['raw_responses_rationales']
        dilemma_responses = self.data['raw_response']

        # Process rationales
        processed_rationales = []
        for i, response in enumerate(rationales_responses):
            try:
                # Skip empty or None responses
                if not response or pd.isna(response) or response.strip() == "":
                    processed_rationales.append("Refusal")
                    continue

                if not dilemma_responses[i] or dilemma_responses[i] == "":
                    processed_rationales.append("Refusal")
                    continue

                rationale = parse_keyword_text(response, "rationales")

                # Filter valid rationales based on predefined keys
                parsed = rationale.split(";")
                keys_lower = [k.lower() for k in self.rationales.keys()]
                parsed = [x.strip() for x in parsed if x.strip().lower() in keys_lower]

                parsed = [p for p in parsed if p in self.rationales.keys()]
                to_add = "; ".join(parsed) if parsed else ""

                processed_rationales.append(to_add)
            except Exception as e:
                print(f"Error processing rationale: {e}")
                processed_rationales.append("")

        # Update dataframe with processed results
        self.data["rationales"] = processed_rationales

        # Save updated dataframe
        os.makedirs(self.judge_output_file.parent, exist_ok=True)
        self.data.to_csv(self.judge_output_file, index=False)
        print(f"[INFO] Processed judge output saved to {self.judge_output_file}")

    def process_quality(self) -> None:
        """
        Process the judge model's raw responses to extract rationales and quality assessments.
        This function should be called after running the judge model to parse the outputs.
        """
        # Check if judge output file exists
        if not self.judge_output_file.exists():
            raise FileNotFoundError(f"Judge output file not found: {self.judge_output_file}. Please run the judge model first.")

        # Load data
        self.load_data()

        # Extract raw responses
        quality_responses = self.data['raw_responses_quality']

        # Process quality assessments
        consistency, logic, bias, pluralism = [], [], [], []

        for i, response in enumerate(quality_responses):
            try:
                # Skip empty or None responses
                if not response or pd.isna(response) or response.strip() == "":
                    consistency.append("")
                    logic.append("")
                    bias.append("")
                    pluralism.append("")
                    continue

                # Extract yes/no answers for each quality dimension
                answer1 = parse_keyword_text(response, "answer_1").lower()
                answer2 = parse_keyword_text(response, "answer_2").lower()
                answer3 = parse_keyword_text(response, "answer_3").lower()
                answer4 = parse_keyword_text(response, "answer_4").lower()

                consistency.append("yes" if "yes" in answer1 else "no" if "no" in answer1 else "")
                logic.append("yes" if "yes" in answer2 else "no" if "no" in answer2 else "")
                bias.append("yes" if "yes" in answer3 else "no" if "no" in answer3 else "")
                pluralism.append("yes" if "yes" in answer4 else "no" if "no" in answer4 else "")
            except Exception as e:
                print(f"Error processing quality assessment at index {i}: {e}")
                consistency.append("")
                logic.append("")
                bias.append("")
                pluralism.append("")

        # Update dataframe with processed results
        self.data["consistency"] = consistency
        self.data["logic"] = logic
        self.data["bias"] = bias
        self.data["pluralism"] = pluralism

        # Save updated dataframe
        os.makedirs(self.judge_output_file.parent, exist_ok=True)
        self.data.to_csv(self.judge_output_file, index=False)
        print(f"[INFO] Processed judge output saved to {self.judge_output_file}")

    def load_data(self):
        if self.judge_output_file.exists():
            self.data = pd.read_csv(self.judge_output_file, keep_default_na=False)
        elif self.response_output_file.exists():
            self.data = pd.read_csv(self.response_output_file, keep_default_na=False)
        else:
            raise FileNotFoundError(f"Neither judge output file nor response output file found.")

    def save_data(self):
        if self.data is not None:
            os.makedirs(self.judge_output_file.parent, exist_ok=True)
            self.data.to_csv(self.judge_output_file, index=False)
            # print(f"[INFO] Data saved to {self.judge_output_file}")
        else:
            raise ValueError("No data to save. Please run the judge model first.")


if __name__ == "__main__":
    runner = JudgeRunner(
        decision_model_id="gemini-2.0-flash-001",
        judge_model_id="gemini-2.0-flash-001",
        decision_run_name=None,
        judge_run_name=None,
        dataset_name=None,
        overwrite=True,
    )
    asyncio.run(runner.run())
    runner.process()