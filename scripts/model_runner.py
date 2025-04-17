import sys
import os
import argparse
import asyncio

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from moral_lens.dilemma import DilemmaRunner

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the DilemmaRunner with specified parameters.')
    parser.add_argument('--model_id', type=str, required=True, help='ID of the decision model')
    parser.add_argument('--decision_run_name', type=str, required=True, help='Name for the decision run')
    parser.add_argument('--results_dir', type=str, required=True, help='Directory for storing results')
    parser.add_argument('--temperature', type=float, default=None, help='Override decision temperature (default: None)')
    parser.add_argument('--choices_filename', type=str, default=None, help='Filename for choices (default: None)')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for the runner (default: 1)')

    args = parser.parse_args()

    # Common runner configuration
    runner_kwargs = {
        "model_id": args.model_id,
        "results_dir": args.results_dir,
        "batch_size": args.batch_size,
    }

    temperature = getattr(args, "temperature", None)
    choices_filename = getattr(args, "choices_filename", None)

    if temperature is not None:
        runner_kwargs["override_decision_temperature"] = temperature
    if choices_filename is not None:
        runner_kwargs["choices_filename"] = choices_filename

    if args.decision_run_name == "sampler" and temperature and temperature > 0:
        # Query the model for three samples
        for i in range(1,4):
            runner_kwargs["decision_run_name"] = f"s{i}"
            runner = DilemmaRunner(**runner_kwargs)
            asyncio.run(runner.run())
    else:
        runner_kwargs["decision_run_name"] = args.decision_run_name
        runner = DilemmaRunner(**runner_kwargs)
        asyncio.run(runner.run())
