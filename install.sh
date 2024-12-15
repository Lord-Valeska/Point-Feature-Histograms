set -e

# Step 1: Print a message
echo "Starting installation process..."

# Detect the OS and install Python 3.10 accordingly
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    echo "Detected Linux. Installing Python 3.10..."
    sudo apt update
    sudo apt install -y python3.10 python3.10-venv
elif [[ "$OSTYPE" == "darwin"* ]]; then
    echo "Detected macOS. Installing Python 3.10..."
    brew install python@3.10
else
    echo "Unsupported OS. Please install Python 3.10 manually."
    exit 1
fi
# fi

# Step 3: Create a virtual environment using Python 3.10
if [ ! -d "venv" ]; then
    echo "Creating a virtual environment with Python 3.10..."
    python3.10 -m venv venv
fi

# Step 5: Activate the virtual environment
echo "Activating the virtual environment..."
source venv/bin/activate

./venv/bin/python3 -m pip install --upgrade pip

# Step 6: Install required dependencies
if [ -f "requirements.txt" ]; then
    echo "Installing dependencies from requirements.txt..."
    pip install -r requirements.txt
else
    echo "No requirements.txt found. Skipping dependency installation."
fi
