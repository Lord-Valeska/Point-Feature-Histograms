#!/bin/bash

# Exit immediately if a command exits with a non-zero status
set -e

# Step 1: Print a message
echo "Starting installation process..."

# Step 2: Check if Python is installed
if ! command -v python3 &>/dev/null; then
    echo "Python3 is not installed. Please install Python3 and try again."
    exit 1
fi

# Step 3: Create a virtual environment (optional but recommended)
if [ ! -d "venv" ]; then
    echo "Creating a virtual environment..."
    python3 -m venv venv
fi

# Step 4: Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

# Step 5: Install required dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Skipping dependency installation."
fi

# Step 6: Make the Python script executable (optional)
chmod +x your_script.py

# Step 7: Create a symbolic link for global usage (optional)
echo "Creating a symbolic link for easier access..."
ln -sf "$(pwd)/your_script.py" /usr/local/bin/your_script

# Step 8: Print success message
echo "Installation complete! You can now run the script using './your_script.py' or 'your_script'."
