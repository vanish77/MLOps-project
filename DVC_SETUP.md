# DVC Setup Guide

## Initial Setup

### 1. Install DVC

```bash
pip install dvc
```

### 2. Initialize DVC

```bash
# Initialize DVC repository
dvc init

# Commit DVC initialization files
git add .dvc .dvcignore dvc.yaml dvc.lock
git commit -m "Initialize DVC"
```

### 3. Add Data to DVC

Since we're using Hugging Face datasets (downloaded on-the-fly), we'll track the processed data:

```bash
# Add processed dataset to DVC
dvc add data/processed/

# Add trained model to DVC
dvc add models/baseline/

# Commit .dvc files
git add data/processed.dvc models/baseline.dvc .gitignore
git commit -m "Add data and models to DVC"
```

### 4. Set Up Remote Storage (Optional)

For local development, you can use local storage:

```bash
# Use local storage (recommended for development)
dvc remote add -d storage ./dvc_storage
dvc remote default storage

# Or use cloud storage (S3, GCS, etc.)
# dvc remote add -d storage s3://bucket-name/path
# dvc remote default storage
```

### 5. Push Data to Remote

```bash
# Push data to remote storage
dvc push
```

## Using DVC Pipeline

### Run Full Pipeline

```bash
# Run all stages in order
dvc repro

# Run specific stage
dvc repro prepare
dvc repro train
dvc repro evaluate
```

### Pull Data

```bash
# Pull all tracked data
dvc pull

# Pull specific file
dvc pull data/processed.dvc
dvc pull models/baseline.dvc
```

## Project Structure with DVC

```
MLOps/
|
+-- data/
|   +-- raw/                      # Raw dataset (HF downloads here)
|   +-- processed/                # DVC tracked: processed dataset
|       +-- processed_train.jsonl
|       +-- processed_test.jsonl
|       +-- raw_train.jsonl
|       +-- raw_test.jsonl
|       +-- processed.dvc         # DVC metadata file
|
+-- models/
|   +-- baseline/                 # DVC tracked: trained model
|       +-- model.safetensors
|       +-- config.json
|       +-- metrics.json
|       +-- baseline.dvc          # DVC metadata file
|
+-- metrics/
|   +-- eval.json                 # DVC tracked: evaluation metrics
|
+-- dvc.yaml                      # DVC pipeline definition
+-- dvc.lock                      # DVC pipeline lock file (auto-generated)
+-- .dvc/                         # DVC internal files
+-- .dvcignore                    # DVC ignore patterns
```

## Workflow

### Initial Setup (First Time)

```bash
# Clone repository
git clone <repo-url>
cd MLOps

# Pull data and models from DVC
dvc pull

# Or run full pipeline from scratch
dvc repro
```

### Regular Development

```bash
# Make changes to code/config
# ...

# Run pipeline
dvc repro

# If data/model changed, commit .dvc files
git add data/processed.dvc models/baseline.dvc dvc.lock
git commit -m "Update data/model versions"
dvc push
```

### Switching Versions

```bash
# Switch to specific git commit
git checkout <commit-hash>

# Pull corresponding data/model versions
dvc pull
```

## File Locations

- **Raw Dataset**: Downloaded from Hugging Face Hub (not stored, re-downloaded)
- **Processed Dataset**: `data/processed/` (DVC tracked)
- **Trained Model**: `models/baseline/` (DVC tracked)
- **Metrics**: `metrics/eval.json` (DVC tracked)

## DVC Remote Storage

By default, DVC uses local storage (`./dvc_storage`). For production, configure remote storage:

- **AWS S3**: `dvc remote add -d storage s3://bucket/path`
- **Google Cloud Storage**: `dvc remote add -d storage gs://bucket/path`
- **Azure Blob**: `dvc remote add -d storage azure://container/path`
- **Local**: `dvc remote add -d storage ./dvc_storage` (default)
