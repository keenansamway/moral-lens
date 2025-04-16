#! /bin/bash

echo "🚀 Starting setup process..."

# Install submodules
# echo "🔄 Initializing git submodules..."
# git submodule update --init --recursive
# echo "✅ Submodules initialized successfully"

# Create and activate virtual environment
echo "🌍 Creating Python virtual environment..."
module load python/3.11
virtualenv --no-download .venv
echo "  → Activating virtual environment..."
source .venv/bin/activate
echo "  → Upgrading pip..."
pip install --no-index --upgrade pip
echo "✅ Virtual environment created and activated"

##### TODO: This has not been tested on DRAC yet #####
# Install dependencies
echo "📚 Installing dependencies..."
echo "  → Installing project requirements..."
echo "  → This may take a few minutes..."
# pip install --no-index -r pyproject.toml
pip install --no-index -r ./requirements_drac.txt
echo "✅ All dependencies installed"

# Setup wandb
# echo "🔑 Setting up Weights & Biases..."
# echo -n "Would you like to set up Weights & Biases? (y/n): "
# read -r SETUP_WANDB
# if [[ "$SETUP_WANDB" =~ ^[Yy]$ ]]; then
#     echo "Please get your API key from: https://wandb.ai/authorize"
#     echo -n "Enter your wandb API key: "
#     read -r WANDB_KEY
#     wandb login "$WANDB_KEY"
#     echo "✅ Successfully logged into wandb"
# else
#     echo "⏩ Skipping wandb setup"
# fi

# Setup Hugging Face
echo "🤗 Setting up Hugging Face..."
echo -n "Would you like to set up Hugging Face? (y/n): "
read -r SETUP_HF
if [[ "$SETUP_HF" =~ ^[Yy]$ ]]; then
    echo "You'll be prompted to enter your Hugging Face token"
    huggingface-cli login
    echo "✅ Hugging Face setup completed"
else
    echo "⏩ Skipping Hugging Face setup"
fi

echo "🎉 Setup completed successfully!"
echo "💡 Virtual environment is now created. Activate it with: source .venv/bin/activate"