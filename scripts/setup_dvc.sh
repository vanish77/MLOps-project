#!/bin/bash
# Setup script for DVC initialization

set -e

echo "Setting up DVC for MLOps project..."

# Check if DVC is installed
if ! command -v dvc &> /dev/null; then
    echo "DVC not found. Installing..."
    pip install dvc
fi

# Initialize DVC if not already initialized
if [ ! -d ".dvc" ]; then
    echo "Initializing DVC..."
    dvc init --no-scm
fi

# Create default local remote (can be changed later)
if ! dvc remote list | grep -q "myremote"; then
    echo "Setting up local remote storage..."
    mkdir -p ~/dvc-storage/mlops
    
    dvc remote add -d myremote ~/dvc-storage/mlops
    
    echo "Default remote 'myremote' created at: ~/dvc-storage/mlops"
    echo "You can change this later with: dvc remote modify myremote url <new-url>"
fi

echo ""
echo "DVC setup complete!"
echo ""
echo "Next steps:"
echo "1. Add data to DVC: dvc add data/processed/dataset_info.json"
echo "2. Add model to DVC: dvc add models/distilbert-imdb/"
echo "3. Run pipeline: dvc repro"
echo "4. Push to remote: dvc push"
echo "5. Commit DVC files: git add *.dvc .dvcignore && git commit -m 'Add DVC tracking'"

