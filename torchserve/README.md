# TorchServe Deployment for IMDb Sentiment Classification

This directory contains all files needed to deploy the IMDb sentiment classification model using TorchServe.

## Quick Start

### 1. Export Model

```bash
# From project root
source .venv/bin/activate
python torchserve/export_model.py
```

### 2. Create Model Archive

```bash
cd torchserve
bash create_mar.sh
```

### 3. Build and Run Docker Container

```bash
# From project root
docker build -t imdb-sentiment-serve:v1 -f torchserve/Dockerfile .
docker run -d -p 8080:8080 -p 8081:8081 --name imdb-sentiment imdb-sentiment-serve:v1
```

### 4. Test API

```bash
# Health check
curl http://localhost:8080/ping

# Prediction
curl -X POST http://localhost:8080/predictions/imdb-sentiment \
  -H "Content-Type: application/json" \
  -d '{"text": "This movie was fantastic!"}'
```

## Files

- `export_model.py` - Export Hugging Face model to TorchScript
- `handler.py` - Custom TorchServe handler for preprocessing/postprocessing
- `config.properties` - TorchServe server configuration
- `requirements.txt` - Handler dependencies
- `create_mar.sh` - Script to create .mar archive
- `Dockerfile` - Docker image definition
- `sample_request.json` - Example API request

## Generated Files

After running the scripts, you'll have:

- `model-artifacts/` - Exported model files (TorchScript, tokenizer, config)
- `model-store/` - Model archives (.mar files)

## API Endpoints

- **Inference**: `http://localhost:8080/predictions/imdb-sentiment`
- **Management**: `http://localhost:8081/models`
- **Metrics**: `http://localhost:8082/metrics`

See main README.md for detailed API documentation.
