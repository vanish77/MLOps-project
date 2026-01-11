# DVC Setup - Next Steps

## What Has Been Done

1. ? Created DVC pipeline definition (`dvc.yaml`) with three stages:
   - `prepare` - Data preparation and preprocessing
   - `train` - Model training
   - `evaluate` - Model evaluation

2. ? Created DVC stage scripts:
   - `src/mlops_imdb/dvc_prepare.py`
   - `src/mlops_imdb/dvc_train.py`
   - `src/mlops_imdb/dvc_evaluate.py`

3. ? Updated `.gitignore` for DVC files
4. ? Added DVC to `requirements.txt`
5. ? Created documentation (DVC_SETUP.md, REPRODUCE.md, DVC_INIT_INSTRUCTIONS.md)

## What You Need to Do

### Step 1: Install DVC

```bash
# Install DVC
pip install -r requirements.txt

# Verify installation
dvc --version
```

### Step 2: Initialize DVC

```bash
# Initialize DVC repository
dvc init

# This creates .dvc/ directory and config files
```

### Step 3: Configure Remote Storage

**Option A: Local Storage (for development/testing)**

```bash
# Create storage directory
mkdir -p ~/dvc-storage/mlops

# Add as default remote
dvc remote add -d myremote ~/dvc-storage/mlops

# Verify
dvc remote list
```

**Option B: Cloud Storage (for production/sharing)**

```bash
# S3 example
dvc remote add -d myremote s3://your-bucket-name/mlops

# Google Cloud Storage example
dvc remote add -d myremote gs://your-bucket-name/mlops

# Azure Blob Storage example
dvc remote add -d myremote azure://your-container/mlops
```

### Step 4: Run Pipeline First Time

```bash
# Run entire pipeline to generate data and model
dvc repro

# This will:
# 1. Run prepare stage (download and preprocess dataset)
# 2. Run train stage (train model, saves to models/distilbert-imdb/)
# 3. Run evaluate stage (evaluate model)
```

### Step 5: Add Data and Model to DVC

After pipeline runs successfully:

```bash
# Add processed dataset metadata
dvc add data/processed/dataset_info.json

# Add trained model
dvc add models/distilbert-imdb/

# This creates .dvc files that track the actual data
# Example: data/processed/dataset_info.json.dvc
# Example: models/distilbert-imdb.dvc
```

### Step 6: Commit DVC Files to Git

```bash
# Add DVC tracking files to Git
git add data/processed/*.dvc
git add models/*.dvc
git add .dvc/
git add dvc.yaml dvc.lock .dvcignore

# Commit
git commit -m "Add DVC tracking for data and models"

# Push to Git
git push origin main
```

### Step 7: Push Data to Remote

```bash
# Push all tracked data to remote storage
dvc push

# Verify files are in remote
dvc status
```

## Verify Everything Works

### Test Reproducibility

```bash
# Remove local data (simulating fresh clone)
rm -rf data/processed/dataset_info.json
rm -rf models/distilbert-imdb/

# Pull from remote
dvc pull

# Verify files are restored
ls -la data/processed/
ls -la models/distilbert-imdb/

# Test full reproduction
rm -rf data/processed/*.json
rm -rf models/distilbert-imdb/
rm -rf models/*.json

dvc pull
dvc repro
```

### Test Version Switching

```bash
# Make some changes
# ... modify code or config ...

# Run pipeline (only changed stages will run)
dvc repro

# Commit changes
git add .
git commit -m "Update training config"

# Switch to previous version
git checkout HEAD~1
dvc pull
dvc repro
```

## Summary

After completing these steps:

1. ? DVC is initialized and configured
2. ? Remote storage is set up
3. ? Data and models are versioned via DVC
4. ? Files are stored in remote (not in Git)
5. ? Full reproducibility works: `git clone && dvc pull && dvc repro`

## Important Notes

- **Large files stay out of Git**: Only `.dvc` metadata files are in Git
- **Actual data in remote**: Large files are stored in DVC remote storage
- **Full reproducibility**: Anyone can clone repo and run `dvc pull && dvc repro` to get exact same results
- **Version control**: Switch between versions using `git checkout` + `dvc pull`

## Troubleshooting

If you encounter issues, see:
- [DVC_INIT_INSTRUCTIONS.md](DVC_INIT_INSTRUCTIONS.md) - Detailed initialization guide
- [DVC_SETUP.md](DVC_SETUP.md) - DVC setup and configuration
- [REPRODUCE.md](REPRODUCE.md) - How to reproduce the project

---

**Ready to version your data and models with DVC!** ??

