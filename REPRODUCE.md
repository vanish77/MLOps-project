# How to Reproduce Project

## Complete Setup from Scratch

### Step 1: Clone Repository

```bash
git clone https://github.com/vanish77/MLOps-project.git
cd MLOps-project
```

### Step 2: Setup Environment

```bash
# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### Step 3: Initialize DVC

```bash
# Initialize DVC
dvc init

# Setup remote storage (choose one option)

# Option A: Local storage (for development)
mkdir -p ~/dvc-storage/mlops
dvc remote add -d myremote ~/dvc-storage/mlops

# Option B: S3 (for production)
# dvc remote add -d myremote s3://your-bucket/mlops

# Option C: Google Cloud Storage
# dvc remote add -d myremote gs://your-bucket/mlops
```

### Step 4: Pull Data and Models

```bash
# Pull all versioned data and models from remote
dvc pull

# Or pull specific files
dvc pull data/processed/dataset_info.json
dvc pull models/distilbert-imdb/
```

### Step 5: Reproduce Pipeline

```bash
# Reproduce entire pipeline
dvc repro

# Or run specific stages
dvc repro prepare      # Data preparation
dvc repro train        # Model training
dvc repro evaluate     # Model evaluation
```

## Alternative: Run Pipeline from Scratch

If you want to run everything from scratch without pulling existing data:

```bash
# Run entire pipeline (will download dataset and train model)
dvc repro

# This will:
# 1. Prepare stage: Download IMDb dataset, preprocess, validate
# 2. Train stage: Train DistilBERT model
# 3. Evaluate stage: Evaluate and generate metrics
```

## Verify Reproducibility

### Check Pipeline Status

```bash
# Check pipeline dependencies
dvc pipeline show dvc.yaml

# Check what needs to be reproduced
dvc status

# Check pipeline structure
dvc dag
```

### Check Data and Model Locations

**Physical Storage:**
- **Dataset metadata:** `data/processed/dataset_info.json` (in Git)
- **Trained model:** `models/distilbert-imdb/` (pulled via `dvc pull`)
- **Actual data files:** Stored in DVC remote storage, not in Git

**How to Verify:**

```bash
# Check if model exists (after dvc pull)
ls -la models/distilbert-imdb/

# Check dataset info (always in Git)
cat data/processed/dataset_info.json

# Check DVC tracking
dvc status
```

## Switching Versions

### Method 1: Using Git + DVC

```bash
# Switch to specific Git commit
git checkout <commit-hash>

# Pull corresponding DVC data
dvc pull

# Reproduce pipeline if needed
dvc repro
```

### Method 2: Using DVC Versions

```bash
# List DVC file versions
dvc list . models/distilbert-imdb/ --rev <rev>

# Get specific version
dvc get . models/distilbert-imdb/ --rev <rev> -o models/distilbert-imdb-v1
```

## Common Workflow

```bash
# Daily workflow

# 1. Clone/pull latest code
git pull origin main

# 2. Pull latest data and models
dvc pull

# 3. Make changes to code/config
# ... edit files ...

# 4. Run pipeline (only changed stages will run)
dvc repro

# 5. Add/commit changes
dvc add models/distilbert-imdb/  # If model changed
git add *.dvc dvc.yaml dvc.lock
git commit -m "Update model training"

# 6. Push to DVC remote
dvc push

# 7. Push to Git
git push origin main
```

## Troubleshooting

### Issue: "File not found" after clone

**Solution:**
```bash
dvc pull  # Download data and models from remote
```

### Issue: "DVC remote not configured"

**Solution:**
```bash
# Setup local remote
dvc remote add -d myremote ~/dvc-storage/mlops

# Or configure cloud storage
# See DVC_SETUP.md for details
```

### Issue: "Pipeline dependencies changed"

**Solution:**
```bash
# Force reproduce specific stage
dvc repro -f prepare

# Or reproduce from specific stage
dvc repro train evaluate
```

## Files Versioned by DVC

- `data/processed/dataset_info.json` - Dataset metadata
- `models/distilbert-imdb/` - Complete trained model directory
- `models/evaluation_summary.json` - Evaluation results

Actual large files are stored in DVC remote and pulled on demand.

---

**Full reproduction command:**
```bash
git clone <repo-url> && cd MLOps-project && dvc pull && dvc repro
```

