import os
from pathlib import Path
import re
import asyncio

import pandas as pd

from typing import Any, List, Optional

from moral_lens.models import ModelFactory, load_model_config
from moral_lens.data_models import ChatMessage, MessageRole, Prompt
from moral_lens.config import ModelConfig, PathConfig
from moral_lens.utils import load_yaml_file, parse_keyword_text

class RankerRunner:
    def __init__(
        self,
        ranker_model_id: str,
        ranker_run_name: Optional[str] = None,
        ranker_cot: bool = False,
        dataset_name: Optional[str] = None,
        results_dir: str = "data/results",
        validation_fn: Optional[Any] = None,
        override_ranker_temperature: Optional[float] = None,
    ):
        self.validation_fn = validation_fn

        # Setup the path configuration
        path_config = PathConfig(results_dir=results_dir)

        # Setup the model configurations
        self.ranker_model_cfg = load_model_config(ranker_model_id)
        if override_ranker_temperature is not None:
            self.ranker_model_cfg.override_temperature(override_ranker_temperature)

        # Set the output file paths
        self.responses_output_dir = path_config.responses_output_dir
        self.responses_output_files = path_config.get_all_response_output_files()
        # self.decision_model_ids = decision_model_ids

        self.ranker_input_file = path_config.get_ranker_input_file(
            name="ranker-input",
            # dataset_name=dataset_name,
        )

        self.ranker_output_file = path_config.get_ranker_output_file(
            ranker_save_id=self.ranker_model_cfg.save_id,
            run_name=ranker_run_name,
            # dataset_name=dataset_name,
        )
        if self.ranker_output_file.exists():
            print(f"Output file already exists at {self.ranker_output_file}. Use `overwrite=True` in .run() to overwrite it.")

        # Load the ranker template and system prompts
        ranker_template_path = path_config.get_file("prompts_ranker.yaml" if not ranker_cot else "prompts_ranker_cot.yaml")
        template_yaml_obj = load_yaml_file(ranker_template_path)

        self.ranker_template = template_yaml_obj.get("ranker_classification_template")
        system_prompts = template_yaml_obj.get("system_prompts")
        self.system_prompt = system_prompts['ranking']

        self.data: Optional[pd.DataFrame] = None


    def create_ranker_input_file(
        self,

    ) -> None:
        all_data = []
        for file in self.responses_output_files:
            model_id = file.stem.split("_")[0]
            if model_id in self.decision_model_ids:
                df = pd.read_csv(file, keep_default_na=False)
                all_data.append(df)

        combined_df = pd.concat(all_data, ignore_index=True)

        comparisons = []


    async def run(
        self, overwrite: bool = False
    ) -> None:
        if not self.ranker_input_file.exists():
            raise FileNotFoundError(f"Ranker input file not found at {self.ranker_input_file}.")

        file_exists = self.ranker_output_file.exists()

        if file_exists and not overwrite:
            self.load_data()
            print(f"Ranker output file already exists at {self.ranker_output_file}. Use `overwrite=True` in .run() to overwrite it.")
            return

        self.load_data()

        prompts = []

        for _, row in self.data.iterrows():
            scenario = re.search(
                r'===BEGIN SCENARIO===\s*(.*?)\s*===END SCENARIO===',
                row['dilemma_prompt'], re.DOTALL
            ).group(1).strip()
            model_a_response = row['model_a_reasoning']
            model_b_response = row['model_b_reasoning']
            decision = row['model_a_decision'] # arbitrary as both decisions must be the same

            system_prompt = self.system_prompt
            prompt_text = self.ranker_template.format(
                scenario=scenario,
                response_a=model_a_response,
                response_b=model_b_response,
                decision=decision,
            )

            prompts.append(Prompt([
                ChatMessage(role=MessageRole.system, content=system_prompt),
                ChatMessage(role=MessageRole.user, content=prompt_text),
            ]))

        # Create and run the model
        model = ModelFactory.get_model(self.ranker_model_cfg)
        responses = await model.ask_async_with_retry(prompts, validation_fn=self.validation_fn)

        self.data['raw_ranker_response'] = [r.content for r in responses]

        for i, response in enumerate(responses):
            ranker_decision = parse_keyword_text(response.content, "answer").lower()
            self.data.loc[i, "ranker_decision"] = ranker_decision

        os.makedirs(self.ranker_output_file.parent, exist_ok=True)
        self.data.to_csv(self.ranker_output_file, index=False)
        print(f"[INFO] Ranker output saved to {self.ranker_output_file}")


    def load_data(self):
        if self.ranker_output_file.exists():
            self.data = pd.read_csv(self.ranker_output_file, keep_default_na=False)
        elif self.ranker_input_file.exists():
            self.data = pd.read_csv(self.ranker_input_file, keep_default_na=False)
        else:
            raise FileNotFoundError(f"Neither ranker output file nor response output file found.")


if __name__ == "__main__":
    runner = RankerRunner(
        ranker_model_id="gemini-2.0-flash-001",
        decision_run_name=None,
        ranker_run_name=None,
        dataset_name=None,
        overwrite=True,
    )
    asyncio.run(runner.run())
    runner.process()