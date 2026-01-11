#!/usr/bin/env python3
"""
DVC train stage: Train sentiment classification model.
"""
import argparse
import json
import logging
import os
import sys
from pathlib import Path

# Add root directory to PYTHONPATH
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from datasets import load_dataset
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, TrainingArguments, Trainer, set_seed
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import numpy as np

from src.mlops_imdb.data import tokenize_dataset
from src.mlops_imdb.validation import run_all_validations, validate_tokenized_data

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    """Compute classification metrics."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    return {"accuracy": acc, "precision": p, "recall": r, "f1": f1}


def main():
    """Train model."""
    parser = argparse.ArgumentParser(description="Train IMDb sentiment classification model")
    parser.add_argument("--data-dir", required=True, help="Directory with processed data")
    parser.add_argument("--model-dir", required=True, help="Output directory for trained model")
    parser.add_argument("--config", required=True, help="Path to training config")
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    # Load config
    import yaml
    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    set_seed(cfg.get("seed", 42))

    logger.info("=" * 60)
    logger.info("DVC Train Stage: Training sentiment classification model")
    logger.info("=" * 60)

    # Load processed data
    logger.info("Loading processed dataset...")
    train_ds = load_dataset("json", data_files=str(data_dir / "processed_train.jsonl"), split="train")
    test_ds = load_dataset("json", data_files=str(data_dir / "processed_test.jsonl"), split="train")

    # Split train into train/val
    val_size = cfg.get("data", {}).get("val_size", 0.1)
    # Note: stratify_by_column not available for JSON-loaded datasets (label is Value, not ClassLabel)
    split = train_ds.train_test_split(test_size=val_size, seed=42)

    from datasets import DatasetDict
    ds = DatasetDict(train=split["train"], validation=split["test"], test=test_ds)

    # Validate data
    logger.info("Validating dataset...")
    run_all_validations(ds, num_labels=2)

    # Load model and tokenizer
    pretrained_name = cfg.get("model", {}).get("pretrained_name", "distilbert-base-uncased")
    logger.info(f"Loading model: {pretrained_name}")
    config = AutoConfig.from_pretrained(pretrained_name, num_labels=2)  # nosec B615
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name, use_fast=True)  # nosec B615
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_name, config=config)  # nosec B615

    # Tokenize
    max_length = cfg.get("data", {}).get("max_length", 256)
    tokenized = tokenize_dataset(ds, tokenizer, max_length=max_length)
    validate_tokenized_data(tokenized, max_length=max_length)

    # Training args
    train_cfg = cfg.get("training", {})
    targs = TrainingArguments(
        output_dir=str(model_dir),
        per_device_train_batch_size=train_cfg.get("per_device_train_batch_size", 16),
        per_device_eval_batch_size=train_cfg.get("per_device_eval_batch_size", 32),
        learning_rate=float(train_cfg.get("learning_rate", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.01)),
        num_train_epochs=float(train_cfg.get("num_train_epochs", 2)),
        warmup_ratio=float(train_cfg.get("warmup_ratio", 0.1)),
        fp16=bool(train_cfg.get("fp16", False)),
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        metric_for_best_model="f1",
        load_best_model_at_end=True,
        save_total_limit=2,
        seed=cfg.get("seed", 42),
    )

    # Train
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    logger.info("Starting training...")
    trainer.train()
    logger.info("Training completed!")

    # Save model
    logger.info("Saving model...")
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))

    # Evaluate on test set
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=tokenized["test"])

    # Save metrics
    metrics_file = model_dir / "metrics.json"
    with open(metrics_file, "w") as f:
        json.dump(test_metrics, f, indent=2)

    logger.info("Test metrics:")
    for key, value in test_metrics.items():
        logger.info(f"  {key}: {value:.4f}")

    logger.info("=" * 60)
    logger.info("Train stage completed successfully!")
    logger.info(f"Model saved to: {model_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
