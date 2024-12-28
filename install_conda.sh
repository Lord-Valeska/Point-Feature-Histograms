#!/bin/bash
set -e

echo "Starting conda environment installation..."

# Check if conda is installed
if ! command -v conda &>/dev/null; then
    echo "Conda is not installed. Please install Anaconda or Miniconda and try again."
    exit 1
fi

# Create a new conda environment (named "myenv") with Python 3.10
# (Modify "myenv" and "3.10" as needed)
echo "Creating conda environment 'myenv' with Python 3.10..."
conda create -y -n myenv python=3.10

# Activate the environment
echo "Activating the 'myenv' environment..."
# NOTE: For script-based activation, use 'source' with the conda.sh script, or
# an explicit path to conda's 'activate' script. One approach:
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate myenv

# (Optional) Upgrade pip within the conda environment
echo "Upgrading pip..."
pip install --upgrade pip

# Install dependencies from requirements.txt, if present
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Skipping dependency installation."
fi

echo "Conda environment 'myenv' setup complete."

python3 demo.py

conda deactivate