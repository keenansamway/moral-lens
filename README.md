# Moral Lens
 Are Language Models Consequentialist or Deontological Moral Reasoners?

## Abstract
> As AI systems increasingly navigate applications in healthcare, law, and governance, understanding how they handle ethically complex scenarios becomes critical. Previous work has mainly examined the moral judgments in large language models (LLMs), rather than their underlying moral reasoning process. In contrast, we focus on a large-scale analysis of the moral reasoning traces provided by LLMs. Furthermore, unlike prior work that attempted to draw inferences from only a handful of moral dilemmas, our study leverages over 600 distinct trolley problems as probes for revealing the reasoning patterns that emerge within different LLMs. We introduce and test a taxonomy of moral rationales to systematically classify reasoning traces according to two main normative ethical theories: consequentialism and deontology. Our analysis reveals that LLM chains-of-thought tend to favor deontological principles based on moral obligations, while post-hoc explanations shift notably toward consequentialist rationales that emphasize utility. Our framework provides a foundation for understanding how LLMs process and articulate ethical considerations, an important step toward safe and interpretable deployment of LLMs in high-stakes decision-making environments.


![Figure 1](assets/figure1.png)

## Installation
Install locally for API usage and data evaluation.
```bash
git clone https://github.com/keenansamway/moral-lens.git
cd moral-lens
```

Run the setup script to create a virtual environment and install dependencies.
```bash
# Install locally for general use.
bash setup.sh

# (OR) Install on DRAC for HuggingFace model inference (no API packages).
# Note that this can take a few minutes to run.
bash setup_DRAC.sh
```

Activate the virtual environment.
```bash
source .venv/bin/activate
```

## Usage

> **A Quick Note**: Currently, this repository consists of research code and as such, there is much work that could be done to improve its usability and readability. If you have any questions, please feel free to open an issue or contact me directly, and I would be happy to help out as best I can.

### Data
You can download the full compiled data from the following google drive link and save it in the `data/` directory:
- all_model_runs.parquet (https://drive.google.com/file/d/1Y56QBjnw5Y-E7FjHi0isSRPdtOX4-xtt/view?usp=share_link)

### Run Models via API
> TODO: Update this section with the latest usage instructions.

See `notebooks/decisions_and_judge.ipynb` for usage examples with OpenAI and OpenRouter APIs (uses functionality in the `dilemma.py` and `judge.py` files).

See `plots/compile_data.ipynb` for how to compile the data from the API and generate various plots.

### Run Models via Compute Canada
Use `salloc` to test out model inference and determine appropriate resources.

1. Begin by requesting a GPU node with the following command:
```bash
salloc --time=03:00:00 --gpus-per-node=2 --cpus-per-task=1 --mem=60G
```

2. Let's set some variables to make it easier to run the model (adjust these as needed):
```bash
model_id="allenai/OLMo-2-0325-32B-DPO"
results_dir=$SLURM_TMPDIR/workspace/data
source_repo=$project/workspace/moral-lens
source_model_path=$HOME/scratch/models
```

3. Copy any necessary files, directories, and models to the compute node:
```bash
files=(
    moral_lens/
    scripts/
)
cd $source_repo
tar -cf $source_repo/tmp_files.tar "${files[@]}"
mkdir -p "$SLURM_TMPDIR/workspace"
cp tmp_files.tar "$SLURM_TMPDIR/workspace"
rm tmp_files.tar
cd "$SLURM_TMPDIR/workspace"
tar -xf tmp_files.tar
rm tmp_files.tar

mkdir -p $SLURM_TMPDIR/models
save_id="${model_id#*/}"
cp -r $source_model_path/$save_id $SLURM_TMPDIR/models
model_path=$SLURM_TMPDIR/models/$save_id
```

4. Load the necessary modules and activate the virtual environment:
```bash
module load python/3.11 cuda cudnn
source $source_repo/.venv/bin/activate
```

5. Run the model inference script:
```bash
cd $SLURM_TMPDIR/workspace
python scripts/model_runner.py --model_id $model_path --decision_run_name "s1" --results_dir $results_dir --temperature 0.7 --batch_size 32
```
(Use `srun --jobid <job_id> --pty nvidia-smi` to check the GPU usage.)

6. After the script finishes, copy the results back to your home directory:
```bash
mkdir -p $source_repo/data
cp -r $SLURM_TMPDIR/workspace/data/ "$source_repo/data/${SLURM_JOB_ID}_${SLURM_JOB_NAME}"
```

After dialing in the necessary resources, we can submit an `sbatch` job.

1. Edit and launch the following script:
```bash
sbatch scripts/launch_job.sh
```


## Citation
If you find this work useful, please cite the following paper:
```bibtex
@misc{samway2025language,
      title={Are Language Models Consequentialist or Deontological Moral Reasoners?},
      author={Keenan Samway and Max Kleiman-Weiner and David Guzman Piedrahita and Rada Mihalcea and Bernhard Schölkopf and Zhijing Jin},
      year={2025},
      eprint={2505.21479},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2505.21479},
}
```