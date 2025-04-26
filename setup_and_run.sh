#!/bin/bash
# setup_and_run.sh

# Create virtual environment
echo "Creating virtual environment..."
python -m venv rag_env

# Activate virtual environment
echo "Activating virtual environment..."
source rag_env/bin/activate

# Create directories
echo "Creating necessary directories..."
mkdir -p files
mkdir -p vectordb

# Install requirements
echo "Installing requirements..."
pip install -r requirements.txt

# Check if CUDA is available
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Run the application
echo "Starting the application..."
python app.py