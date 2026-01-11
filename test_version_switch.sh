#!/bin/bash
# Test script to verify DVC version switching functionality

set -e

echo "========================================="
echo "Testing DVC Version Switching"
echo "========================================="

# Get current commit
CURRENT_COMMIT=$(git rev-parse HEAD)
echo "Current commit: $CURRENT_COMMIT"

# Check if data/model files exist
echo ""
echo "Checking current files..."
if [ -f "data/processed/processed_train.jsonl" ]; then
    echo "? data/processed/processed_train.jsonl exists"
    TRAIN_SIZE=$(wc -l < data/processed/processed_train.jsonl)
    echo "  Size: $TRAIN_SIZE lines"
else
    echo "? data/processed/processed_train.jsonl missing"
fi

if [ -d "models/baseline" ]; then
    echo "? models/baseline/ exists"
    if [ -f "models/baseline/metrics.json" ]; then
        echo "? models/baseline/metrics.json exists"
    fi
else
    echo "? models/baseline/ missing"
fi

# Test dvc pull (should restore files if they were removed)
echo ""
echo "Testing dvc pull..."
dvc pull
echo "? dvc pull completed successfully"

# Check if files are accessible
echo ""
echo "Verifying files after dvc pull..."
if [ -f "data/processed/processed_train.jsonl" ] && [ -d "models/baseline" ]; then
    echo "? All files accessible after dvc pull"
else
    echo "? Some files missing after dvc pull"
    exit 1
fi

echo ""
echo "========================================="
echo "Version switching test completed!"
echo "========================================="
echo ""
echo "To test switching between Git versions:"
echo "1. Make a commit: git add . && git commit -m 'Test commit'"
echo "2. Switch to previous: git checkout HEAD~1"
echo "3. Pull DVC data: dvc pull"
echo "4. Switch back: git checkout main && dvc pull"
