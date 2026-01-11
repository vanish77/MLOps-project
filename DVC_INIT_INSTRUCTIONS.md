# DVC Initialization Instructions

## Quick Setup Guide

### Step 1: Install DVC

```bash
pip install dvc
# or
pip install -r requirements.txt  # includes dvc
```

### Step 2: Initialize DVC

```bash
# Initialize DVC repository
dvc init

# This creates:
# - .dvc/ directory
# - .dvc/config file
# - .dvcignore file
```

### Step 3: Configure Remote Storage

**Option A: Local Storage (Recommended for first time)**

```bash
# Create storage directory
mkdir -p ~/dvc-storage/mlops

# Add as default remote
dvc remote add -d myremote ~/dvc-storage/mlops

# Verify
dvc remote list
```

**Option B: S3 Storage**

```bash
# Configure AWS credentials first
export AWS_ACCESS_KEY_ID=your-key
export AWS_SECRET_ACCESS_KEY=your-secret

# Add S3 remote
dvc remote add -d myremote s3://your-bucket-name/mlops

# Verify
dvc remote list
```

### Step 4: Data and Models Already Tracked via Pipeline

**Important:** When using DVC pipeline (defined in `dvc.yaml`), outputs are automatically tracked through the pipeline. You don't need to run `dvc add` separately!

The pipeline already tracks:
- `data/processed/processed_*.jsonl` (outputs of `prepare` stage)
- `models/baseline/` (output of `train` stage)
- `metrics/eval.json` (output of `evaluate` stage)

All tracking information is stored in `dvc.lock` file.

### Step 5: Commit DVC Files to Git

```bash
# Add DVC tracking files
git add .dvc/
git add dvc.yaml dvc.lock .dvcignore .gitignore

# Commit
git commit -m "Add DVC pipeline tracking for data and models"

# Push to Git
git push origin main
```

### Step 6: Push Data to Remote

```bash
# Push all tracked data to remote
dvc push

# Or push specific files
dvc push data/processed/dataset_info.json
dvc push models/distilbert-imdb/
```

## Verification

### Check DVC Status

```bash
# Check what's tracked by DVC
dvc status

# Check pipeline
dvc pipeline show dvc.yaml

# Check DAG
dvc dag
```

### Test Reproducibility

```bash
# Remove local data (simulating fresh clone)
rm -rf data/processed/dataset_info.json
rm -rf models/distilbert-imdb/

# Pull from remote
dvc pull

# Verify files are back
ls -la data/processed/
ls -la models/distilbert-imdb/
```

## Using the Pipeline

### Run Entire Pipeline

```bash
# Run all stages in order
dvc repro

# This will:
# 1. Run prepare (if needed)
# 2. Run train (if dependencies changed)
# 3. Run evaluate (if model changed)
```

### Run Specific Stage

```bash
# Run only prepare stage
dvc repro prepare

# Run only train stage (will run prepare if needed)
dvc repro train

# Run train and evaluate
dvc repro train evaluate
```

## Switching Versions

### Method 1: Using Git

```bash
# Checkout specific commit
git checkout <commit-hash>

# Pull corresponding DVC data
dvc pull

# Reproduce if needed
dvc repro
```

### Method 2: Direct DVC

```bash
# List available versions
git log --oneline

# Checkout and pull
git checkout <commit-hash>
dvc pull
```

## Troubleshooting

### Issue: "dvc init" fails

**Solution:**
```bash
# Make sure DVC is installed
pip install dvc

# If .dvc exists, remove and reinit
rm -rf .dvc
dvc init
```

### Issue: "Remote not configured"

**Solution:**
```bash
# Setup local remote
dvc remote add -d myremote ~/dvc-storage/mlops

# Or check existing remotes
dvc remote list
```

### Issue: "File not found" after clone

**Solution:**
```bash
# Pull data from remote
dvc pull

# If remote not configured, setup first
dvc remote add -d myremote ~/dvc-storage/mlops
dvc pull
```

---

**After initialization, your project is fully versioned with DVC!**

