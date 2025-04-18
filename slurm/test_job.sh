#!/bin/bash

#SBATCH --job-name=inference_test
#SBATCH --output=slurm/output/%j_%x.out

#SBATCH --time=0-1:00:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G

# Override echo to include timestamp
echo() {
  builtin echo "[$(date +"%T")] $*"
}

echo "🚀 Starting job..."

# Define run variables
model_id="google/gemma-3-1b-it"
run_name="s1"
temperature=0.7
batch_size=32

# Logs
echo "📝 Prepare run with model_id: $model_id"
echo "📝 Prepare run with run_name: $run_name"
echo "📝 Prepare run with temperature: $temperature"
echo "📝 Prepare run with batch_size: $batch_size"


# Slurm automatically creates a temporary directory for each job as $SLURM_TMPDIR
echo "🗂️ Local compute node directory: $SLURM_TMPDIR"


# Load the necessary modules
echo "🔧 Loading modules..."
module load python/3.11 cuda cudnn
source $project/workspace/moral-lens/.venv/bin/activate
echo "✅ Loaded modules"


# Run the Python script
echo "🚀 Running model_runner.sh..."
bash scripts/model_runner.sh $model_id $run_name $temperature $batch_size
echo "✅ Finished running model_runner.sh"


echo "🎉 Job finished successfully!"