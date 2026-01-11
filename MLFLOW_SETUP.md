# MLflow Configuration Guide

## Overview

MLflow tracks experiment parameters, metrics, and artifacts automatically.

## Storage Options

### Option 1: Local Storage (Default)

Stores all runs locally in `mlruns/` directory.

```bash
# No additional configuration needed
# MLflow will use file://./mlruns by default
python scripts/train.py --config configs/baseline.yaml
```

### Option 2: Remote Storage (PostgreSQL + S3/MinIO)

For production use with PostgreSQL backend and S3/MinIO artifact storage.

#### Setup PostgreSQL Backend

```bash
# Install PostgreSQL driver
pip install psycopg2-binary

# Set tracking URI
export MLFLOW_TRACKING_URI="postgresql://user:password@localhost:5432/mlflow_db"
```

#### Setup S3/MinIO Artifact Storage

```bash
# Install S3 support
pip install boto3

# Configure AWS credentials
export AWS_ACCESS_KEY_ID="your-access-key"
export AWS_SECRET_ACCESS_KEY="your-secret-key"
export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"  # for MinIO

# Or use AWS S3 directly (no endpoint URL needed)
```

#### Start MLflow Server

```bash
# Start MLflow tracking server
mlflow server \
    --backend-store-uri postgresql://user:password@localhost:5432/mlflow_db \
    --default-artifact-root s3://mlflow-artifacts \
    --host 0.0.0.0 \
    --port 5000
```

```bash
# Then set tracking URI in your environment
export MLFLOW_TRACKING_URI="http://localhost:5000"
```

## Environment Variables

```bash
# Tracking URI (where runs are stored)
export MLFLOW_TRACKING_URI="file:./mlruns"  # Local (default)
# or
export MLFLOW_TRACKING_URI="http://localhost:5000"  # Remote server
# or
export MLFLOW_TRACKING_URI="postgresql://user:pass@host:5432/mlflow_db"  # Direct DB

# Experiment name
export MLFLOW_EXPERIMENT_NAME="imdb-sentiment-classification"

# For S3/MinIO artifact storage
export AWS_ACCESS_KEY_ID="your-key"
export AWS_SECRET_ACCESS_KEY="your-secret"
export MLFLOW_S3_ENDPOINT_URL="http://localhost:9000"  # MinIO only
```

## Running Experiments

### Basic Usage

```bash
# Default (local mlruns/)
python scripts/train.py --config configs/baseline.yaml

# With custom experiment name
MLFLOW_EXPERIMENT_NAME="my-experiment" python scripts/train.py --config configs/baseline.yaml

# With remote server
MLFLOW_TRACKING_URI="http://localhost:5000" python scripts/train.py --config configs/baseline.yaml
```

### Viewing Results

```bash
# Start MLflow UI (local)
mlflow ui

# Or specify mlruns directory
mlflow ui --backend-store-uri file:./mlruns

# Or connect to remote server
mlflow ui --backend-store-uri http://localhost:5000

# Open browser: http://localhost:5000
```

## Tracked Information

MLflow automatically logs:

### Parameters
- seed
- dataset_name, val_size, max_length
- pretrained_name, num_labels
- learning_rate, batch_size, num_epochs
- weight_decay, warmup_ratio, fp16
- train/val/test sample counts

### Metrics
- Training metrics (loss, learning rate per step)
- Validation metrics per epoch (accuracy, precision, recall, f1)
- Test metrics (final evaluation)

### Artifacts
- Config file (`config/baseline.yaml`)
- Training config (`config/training_config.yaml`)
- Test metrics (`metrics/metrics_test.json`)
- Training logs (`logs/train.log`)
- DVC lock file (`dvc/dvc.lock`)
- Trained model (via autolog)

## Docker Setup for Remote Storage

### PostgreSQL

```yaml
# docker-compose.yml
version: '3.8'
services:
  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: mlflow
      POSTGRES_PASSWORD: mlflow
      POSTGRES_DB: mlflow_db
    ports:
      - "5432:5432"
    volumes:
      - postgres_data:/var/lib/postgresql/data

volumes:
  postgres_data:
```

### MinIO (S3-compatible)

```yaml
# docker-compose.yml
version: '3.8'
services:
  minio:
    image: minio/minio
    command: server /data --console-address ":9001"
    environment:
      MINIO_ROOT_USER: minioadmin
      MINIO_ROOT_PASSWORD: minioadmin
    ports:
      - "9000:9000"
      - "9001:9001"
    volumes:
      - minio_data:/data

volumes:
  minio_data:
```

### MLflow Server

```yaml
# docker-compose.yml
version: '3.8'
services:
  mlflow:
    image: python:3.10-slim
    command: >
      bash -c "pip install mlflow psycopg2-binary boto3 &&
      mlflow server
      --backend-store-uri postgresql://mlflow:mlflow@postgres:5432/mlflow_db
      --default-artifact-root s3://mlflow-artifacts
      --host 0.0.0.0
      --port 5000"
    environment:
      AWS_ACCESS_KEY_ID: minioadmin
      AWS_SECRET_ACCESS_KEY: minioadmin
      MLFLOW_S3_ENDPOINT_URL: http://minio:9000
    ports:
      - "5000:5000"
    depends_on:
      - postgres
      - minio
```

## Best Practices

1. **Use Experiment Names**: Group related runs
   ```bash
   MLFLOW_EXPERIMENT_NAME="experiment-lr-sweep" python scripts/train.py ...
   ```

2. **Tag Runs**: Add custom tags
   ```python
   mlflow.set_tag("description", "Baseline with DistilBERT")
   mlflow.set_tag("version", "v1.0")
   ```

3. **Remote Storage for Teams**: Use PostgreSQL + S3 for sharing

4. **Version Control**: DVC for data, MLflow for experiments

5. **Clean Old Runs**: Periodically clean `mlruns/` or use retention policies
