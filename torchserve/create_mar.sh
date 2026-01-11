#!/bin/bash
# Script to create TorchServe model archive (.mar file)

set -e

echo "========================================="
echo "Creating TorchServe Model Archive"
echo "========================================="

# Configuration
MODEL_NAME="imdb-sentiment"
MODEL_VERSION="1.0"
HANDLER_FILE="handler.py"
MODEL_FILE="model-artifacts/model.pt"
EXTRA_FILES="model-artifacts/config.json,model-artifacts/tokenizer.json,model-artifacts/vocab.txt,model-artifacts/special_tokens_map.json,model-artifacts/tokenizer_config.json"
EXPORT_PATH="model-store"
REQUIREMENTS_FILE="requirements.txt"

# Check if model artifacts exist
if [ ! -f "$MODEL_FILE" ]; then
    echo "Error: Model file not found: $MODEL_FILE"
    echo "Please run export_model.py first:"
    echo "  python torchserve/export_model.py"
    exit 1
fi

# Create model-store directory
mkdir -p "$EXPORT_PATH"

# Create MAR file
echo "Creating .mar archive..."
torch-model-archiver \
    --model-name "$MODEL_NAME" \
    --version "$MODEL_VERSION" \
    --serialized-file "$MODEL_FILE" \
    --handler "$HANDLER_FILE" \
    --extra-files "$EXTRA_FILES" \
    --requirements-file "$REQUIREMENTS_FILE" \
    --export-path "$EXPORT_PATH" \
    --force

echo "========================================="
echo "Model archive created successfully!"
echo "Location: $EXPORT_PATH/$MODEL_NAME.mar"
echo "========================================="
echo ""
echo "To test locally:"
echo "  torchserve --start --model-store $EXPORT_PATH --models $MODEL_NAME=$MODEL_NAME.mar"
echo ""
echo "To test with Docker:"
echo "  docker build -t imdb-sentiment-serve:v1 -f torchserve/Dockerfile ."
echo "  docker run -d -p 8080:8080 -p 8081:8081 imdb-sentiment-serve:v1"
