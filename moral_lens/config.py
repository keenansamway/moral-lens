from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Dict, Optional

from datetime import datetime
from dotenv import load_dotenv

load_dotenv(override=True)  # Load environment variables from .env file if it exists

@dataclass
class ApiConfig:
    """Loads API keys from environment variables."""
    HF_TOKEN: Optional[str] = os.getenv("HF_TOKEN")
    OPENAI_API_KEY: Optional[str] = os.getenv("OPENAI_API_KEY")
    ANTHROPIC_API_KEY: Optional[str] = os.getenv("ANTHROPIC_API_KEY")
    GOOGLE_API_KEY: Optional[str] = os.getenv("GOOGLE_API_KEY")
    OPENROUTER_API_KEY: Optional[str] = os.getenv("OPENROUTER_API_KEY")

    def summary(self):
        """Returns a summary of the API configuration."""
        keys = [k for k, v in self.__dict__.items() if v is not None]
        if not keys:
            return "[INFO] No API keys are set."
        return f"[INFO] Configured API keys: {', '.join(keys)}"

@dataclass
class RateLimitConfig:
    concurrency: int = 10
    max_rate: int = 5
    period: int = 1

@dataclass
class ModelConfig:
    """Configuration for a specific model used in the inference API."""
    model_id: str
    save_id: str
    model_name: str
    provider: str
    release_date: datetime.date

    reasoning_model: bool
    accepts_system_message: bool
    temperature: float
    max_completion_tokens: int
    provider_routing: str
    reasoning_effort: str

    # function to convert the model configuration to a dictionary for kwargs
    def to_dict(self) -> Dict[str, Optional[Any]]:
        """
        Converts the ModelConfig instance to a dictionary for use in API calls.
        """
        return self.__dict__

    def override_temperature(self, temperature: float):
        """
        Override the temperature for this model configuration. Use with caution.
        """
        self.temperature = temperature


class PathConfig:
    """
    Configuration for paths used in the application.
    """

    def __init__(
        self,
        datasets_dir: Path = Path("moral_lens/config"),
        results_dir: Path | str = Path("data/results"),
    ) -> None:
        if isinstance(results_dir, str):
            results_dir = Path(results_dir)

        self.datasets_dir = datasets_dir  # Path to the MultiTP datasets

        self.results_dir = results_dir  # Directory to save results
        self.responses_output_dir = self.results_dir / "responses"
        self.judge_output_dir = self.results_dir / "judge"
        self.ranker_input_dir = self.results_dir / "ranker"
        self.ranker_output_dir = self.results_dir / "ranker"

    def get_file(self, file_name) -> Path:
        """
        Returns the path to the file.
        """
        file = self.datasets_dir / file_name
        if not os.path.exists(file):
            raise FileNotFoundError(f"Missing file: {file}")
        return file

    def get_response_output_file(
        self,
        decision_save_id: str,
        run_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Path:
        """
        Returns the path for the response output file.
        """
        # file_name = f"{decision_save_id}_{run_name}_{language}_{dataset_name}.cvs"

        file_name = f"{decision_save_id}"
        file_name += f"_{run_name}" if run_name else ""
        # file_name += f"_{language}" if language else ""
        # file_name += f"_{dataset_name}" if dataset_name else ""
        file_name += ".csv"

        return self.responses_output_dir / file_name

    def get_all_response_output_files(self) -> list[Path]:
        """
        Returns all response output files.
        """
        return list(self.responses_output_dir.glob("*.csv"))

    def get_judge_output_file(
        self,
        decision_save_id: str,
        judge_save_id: str,
        run_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Path:
        """
        Returns the path for the judge output file.
        """
        # file_name = f"{decision_save_id}_{judge_save_id}_{run_name}_{language}_{dataset_name}.cvs"

        file_name = f"{decision_save_id}_{judge_save_id}"
        file_name += f"_{run_name}" if run_name else ""
        # file_name += f"_{language}" if language else ""
        # file_name += f"_{dataset_name}" if dataset_name else ""
        file_name += ".csv"

        return self.judge_output_dir / file_name

    def get_ranker_input_file(
        self,
        name: str,
        run_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Path:
        """
        Returns the path for the ranker input file.
        """
        # file_name = f"{judge_save_id}_{run_name}_{language}_{dataset_name}.cvs"

        file_name = f"{name}"
        file_name += f"_{run_name}" if run_name else ""
        # file_name += f"_{language}" if language else ""
        # file_name += f"_{dataset_name}" if dataset_name else ""
        file_name += ".csv"

        return self.ranker_input_dir / file_name

    def get_ranker_output_file(
        self,
        ranker_save_id: str,
        run_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        language: Optional[str] = None,
    ) -> Path:
        """
        Returns the path for the ranker output file.
        """
        # file_name = f"{judge_save_id}_{run_name}_{language}_{dataset_name}.cvs"

        file_name = f"{ranker_save_id}"
        file_name += f"_{run_name}" if run_name else ""
        # file_name += f"_{language}" if language else ""
        # file_name += f"_{dataset_name}" if dataset_name else ""
        file_name += ".csv"

        return self.ranker_output_dir / file_name