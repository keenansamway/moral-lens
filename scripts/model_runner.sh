#!/bin/bash

# Override echo to include timestamp
echo() {
  builtin echo "[$(date +"%T")] $*"
}

# Define run variables from args
model_id=$1
run_name=$2
temperature=$3
batch_size=$4

# Define a variable to point to the project directory
source_repo=$project/workspace/moral-lens
source_model_path=$HOME/scratch/models


# Copy any necessary files and directories to temp directory using tar and cp
echo "📦 Copying files to $SLURM_TMPDIR/workspace..."
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
echo "✅ Copied files to $SLURM_TMPDIR/workspace"


# Copy the model to the temporary directory
echo "🤖 Copying model to $SLURM_TMPDIR/models..."
mkdir -p $SLURM_TMPDIR/models
save_id="${model_id#*/}"
cp -r $source_model_path/$save_id $SLURM_TMPDIR/models
model_path=$SLURM_TMPDIR/models/$save_id
echo "✅ Copied $model_id to $model_path"


# Set up a trap to call the cleanup function on exit or error
cleanup() {
    echo "🧹 Job is finishing. Start clearning up..."
    if [ -d "$SLURM_TMPDIR/workspace/data" ]; then
        echo "📍 Results found in $SLURM_TMPDIR/workspace/data"
        echo "📦 Copying results to $source_repo/data/$SLURM_JOB_ID..."
        mkdir -p $source_repo/data
        cp -r $SLURM_TMPDIR/workspace/data $source_repo/data/$SLURM_JOB_ID
        echo "✅ Copied results to $source_repo/data/$SLURM_JOB_ID"
    else
        echo "❌ No results found in $SLURM_TMPDIR/workspace/data"
    fi

    echo "🗑️ Removing downloaded models..."
    rm -rf $SLURM_TMPDIR/models
    echo "✅ Removed downloaded models"
}
trap cleanup SIGTERM SIGINT SIGHUP EXIT


# Run the decision runner script
echo "🚀 Running model_runner.py..."
cd "$SLURM_TMPDIR/workspace"
python scripts/model_runner.py \
    --model_id $model_path \
    --decision_run_name $run_name \
    --results_dir $SLURM_TMPDIR/workspace/data \
    --temperature $temperature \
    --batch_size $batch_size \
echo "✅ Finished running model_runner.py"