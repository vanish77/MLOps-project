# DVC Setup Instructions

## Quick Setup Guide

### 1. Initialize DVC

```bash
# Install DVC (if not already installed)
pip install dvc

# Initialize DVC repository
dvc init

# Commit DVC initialization files
git add .dvc .dvcignore dvc.yaml .gitignore
git commit -m "Initialize DVC for data and model versioning"
```

### 2. Set Up Remote Storage

**Option 1: Local Storage (Recommended for Development)**

```bash
# Create local storage directory
mkdir -p ./dvc_storage

# Add as default remote
dvc remote add -d storage ./dvc_storage

# Verify remote
dvc remote list
```

**Option 2: Cloud Storage (For Production/Sharing)**

```bash
# AWS S3
dvc remote add -d storage s3://bucket-name/mlops

# Google Cloud Storage
dvc remote add -d storage gs://bucket-name/mlops

# Azure Blob Storage
dvc remote add -d storage azure://container-name/mlops
```

### 3. Add Data and Models to DVC

After running the pipeline (or if you have existing data/model):

```bash
# Add processed dataset
dvc add data/processed/

# Add trained model
dvc add models/baseline/

# Commit .dvc files
git add data/processed.dvc models/baseline.dvc .gitignore
git commit -m "Add data and models to DVC versioning"
```

### 4. Push to Remote Storage

```bash
# Push data and models to remote
dvc push
```

## Running DVC Pipeline

### Full Pipeline

```bash
# Run all stages in order: prepare -> train -> evaluate
dvc repro

# This will:
# 1. Download and preprocess IMDb dataset (prepare stage)
# 2. Train the model (train stage)
# 3. Evaluate the model (evaluate stage)
```

### Individual Stages

```bash
# Run only data preparation
dvc repro prepare

# Run only training (requires prepared data)
dvc repro train

# Run only evaluation (requires trained model)
dvc repro evaluate

# Run train and evaluate (skip prepare if data exists)
dvc repro train evaluate
```

## Restoring Versions

### Complete Reproduction

```bash
# Clone repository
git clone <repo-url> && cd MLOps

# Install dependencies
pip install -r requirements.txt

# Initialize DVC and set remote
dvc init
dvc remote add -d storage ./dvc_storage

# Pull data and models
dvc pull

# Or rebuild from scratch
dvc repro
```

### Single Command

```bash
git clone <repo-url> && cd MLOps && \
  pip install -r requirements.txt && \
  dvc init && dvc remote add -d storage ./dvc_storage && \
  dvc pull && dvc repro
```

### Switch to Specific Version

```bash
# Switch Git commit
git checkout <commit-hash>

# Pull corresponding DVC data/models
dvc pull

# The .dvc files in that commit will automatically pull the correct versions
```

## File Locations

### In Git Repository

- `.dvc/` - DVC internal files (tracked)
- `dvc.yaml` - Pipeline definition (tracked)
- `dvc.lock` - Pipeline lock file (tracked)
- `data/processed.dvc` - Dataset metadata file (tracked)
- `models/baseline.dvc` - Model metadata file (tracked)
- `metrics/eval.json.dvc` - Metrics metadata file (tracked)

### In DVC Remote Storage

- Actual data files: `data/processed/processed_*.jsonl` (not in Git)
- Model files: `models/baseline/model.safetensors`, etc. (not in Git)
- Metrics: `metrics/eval.json` (not in Git)

### After `dvc pull`

Files are downloaded to:
- `data/processed/` - Processed dataset files
- `models/baseline/` - Trained model files
- `metrics/` - Evaluation metrics

## Pipeline Stages

### Stage 1: prepare

- **Script**: `scripts/prepare.py`
- **Purpose**: Download and preprocess IMDb dataset
- **Inputs**: None (downloads from Hugging Face)
- **Outputs**:
  - `data/processed/processed_train.jsonl`
  - `data/processed/processed_test.jsonl`
  - `data/processed/raw_train.jsonl`
  - `data/processed/raw_test.jsonl`

### Stage 2: train

- **Script**: `scripts/train_dvc.py`
- **Purpose**: Train sentiment classification model
- **Inputs**:
  - `data/processed/processed_train.jsonl`
  - `data/processed/processed_test.jsonl`
  - `configs/baseline.yaml`
- **Outputs**:
  - `models/baseline/` (model files)
  - `models/baseline/metrics.json` (training metrics)

### Stage 3: evaluate

- **Script**: `scripts/evaluate.py`
- **Purpose**: Evaluate trained model
- **Inputs**:
  - `models/baseline/`
- **Outputs**:
  - `metrics/eval.json` (evaluation summary)

## Common Commands

```bash
# Check pipeline status
dvc status

# View pipeline graph
dvc dag

# Check pipeline changes
dvc diff

# Remove tracked files (keep metadata)
dvc remove data/processed.dvc

# Update DVC files after changes
dvc add data/processed/

# Show metrics
dvc metrics show

# Compare metrics
dvc metrics diff
```

## Troubleshooting

### Issue: "Remote storage not configured"

```bash
# Add remote storage
dvc remote add -d storage ./dvc_storage

# Or for cloud
dvc remote add -d storage s3://bucket-name/mlops
```

### Issue: "Data files not found"

```bash
# Pull data from remote
dvc pull

# Or rebuild
dvc repro
```

### Issue: "Pipeline dependencies changed"

```bash
# Rebuild pipeline
dvc repro

# Or force rebuild
dvc repro --force
```

## Best Practices

1. **Always commit `.dvc` files** - They contain metadata about data/model versions
2. **Push to remote regularly** - After adding new data/models: `dvc push`
3. **Use meaningful commit messages** - Describe what changed in data/model
4. **Tag important versions** - `git tag v1.0` for important model versions
5. **Backup remote storage** - For production, use cloud storage with backups

## Next Steps

1. Initialize DVC: `dvc init`
2. Set remote: `dvc remote add -d storage ./dvc_storage`
3. Run pipeline: `dvc repro`
4. Add to DVC: `dvc add data/processed/ models/baseline/`
5. Commit: `git add *.dvc dvc.lock && git commit -m "Add versioned data/model"`
6. Push: `dvc push`
