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
        if self.judge_output_file.exists():
            print(f"Output file already exists at {self.judge_output_file}. Use `overwrite=True` in .run() to overwrite it.")

        # Load the judge template and system prompts
        judge_template_path = path_config.get_file("prompts_judge.yaml" if not judge_cot else "prompts_judge_cot.yaml")
        template_yaml_obj = load_yaml_file(judge_template_path)

        self.rational_classification_template = template_yaml_obj.get("rational_classification_template")
        self.quality_evaluation_template = template_yaml_obj.get("quality_evaluation_template")
        self.system_prompts = template_yaml_obj.get("system_prompts")
        self.rationales = template_yaml_obj.get("rationales")

        if not self.rational_classification_template or not self.quality_evaluation_template:
            raise ValueError("Missing templates in judge YAML.")
        if not self.system_prompts or not self.rationales:
            raise ValueError("Missing system prompts or rationales.")

        self.data: Optional[pd.DataFrame] = None

    async def run(
        self, overwrite: bool = False, skip_quality: bool = True
    ) -> None:
        if not self.response_output_file.exists():
            raise FileNotFoundError(f"Decision output file not found at {self.response_output_file}. Please run the decision model first.")

        if self.judge_output_file.exists() and not overwrite:
            print(f"Output file already exists at {self.judge_output_file}. Use `overwrite=True` to overwrite it.")
            return

        # Load data
        self.load_data()

        rationales_prompts = []
        quality_prompts = []

        for _, row in self.data.iterrows():
            scenario = re.search(
                r'===BEGIN SCENARIO===\s*(.*?)\s*===END SCENARIO===',
                row['dilemma_prompt'], re.DOTALL
            ).group(1).strip()
            reasoning = row['reasoning']
            decision = row['decision']

            rationale_str = "\n".join([f'- "{k}": {v}' for k, v in self.rationales.items()])
            rationale_prompt = self.rational_classification_template.format(
                scenario=scenario, reasoning=reasoning, decision=decision, rationales=rationale_str)
            quality_prompt = self.quality_evaluation_template.format(
                scenario=scenario, reasoning=reasoning, decision=decision)

            rationales_prompts.append(Prompt([
                ChatMessage(role=MessageRole.system, content=self.system_prompts['rationales']),
                ChatMessage(role=MessageRole.user, content=rationale_prompt)
            ]))

            quality_prompts.append(Prompt([
                ChatMessage(role=MessageRole.system, content=self.system_prompts['quality']),
                ChatMessage(role=MessageRole.user, content=quality_prompt)
            ]))

        model = ModelFactory.get_model(model=self.judge_model_cfg)
        rationales_responses = await model.ask_async_with_retry(rationales_prompts, validation_fn=self.validation_fn)
        if not skip_quality:
            quality_responses = await model.ask_async_with_retry(quality_prompts, validation_fn=self.validation_fn)

        self.data["raw_responses_rationales"] = [r.content for r in rationales_responses]
        if not skip_quality:
            self.data["raw_responses_quality"] = [r.content for r in quality_responses]

        # Save the raw responses to the judge output file
        os.makedirs(self.judge_output_file.parent, exist_ok=True)
        self.data.to_csv(self.judge_output_file, index=False)
        print(f"[INFO] Judge output saved to {self.judge_output_file}")

        self.process_rationales()
        if not skip_quality:
             self.process_quality()

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

        # Process rationales
        processed_rationales = []
        for response in rationales_responses:
            try:
                # Skip empty or None responses
                if not response or pd.isna(response):
                    processed_rationales.append("")
                    continue

                rationale = parse_keyword_text(response, "rationales")

                # Filter valid rationales based on predefined keys
                parsed = rationale.split(";")
                keys_lower = [k.lower() for k in self.rationales.keys()]
                parsed = [x.strip() for x in parsed if x.strip().lower() in keys_lower]

                processed_rationales.append("; ".join(parsed))
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
                if not response or pd.isna(response):
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

                consistency.append("yes" if "yes" in answer1 else "no")
                logic.append("yes" if "yes" in answer2 else "no")
                bias.append("yes" if "yes" in answer3 else "no")
                pluralism.append("yes" if "yes" in answer4 else "no")
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