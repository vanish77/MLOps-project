"""
Module for training IMDb sentiment classification model.
"""
import json
import logging
import os
import random
from typing import Dict, Optional

import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from transformers import (
    TrainingArguments,
    Trainer,
    set_seed,
)

from .config import load_config
from .data import load_and_prepare_dataset, tokenize_dataset
from .model import build_model_and_tokenizer

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred):
    """
    Compute model quality metrics.
    
    Args:
        eval_pred: Tuple (predictions, labels)
        
    Returns:
        Dictionary with metrics
    """
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    
    acc = accuracy_score(labels, preds)
    p, r, f1, _ = precision_recall_fscore_support(
        labels, preds, average="binary", zero_division=0
    )
    
    return {
        "accuracy": acc,
        "precision": p,
        "recall": r,
        "f1": f1
    }


def run_training(config_path: str, overrides: Optional[Dict[str, str]] = None) -> None:
    """
    Run model training process.
    
    Args:
        config_path: Path to configuration file
        overrides: Config parameter overrides
    """
    # Load configuration
    cfg = load_config(config_path, overrides)
    
    # Set seed for reproducibility
    set_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    # Create directories for artifacts
    out_dir = cfg.training.get("output_dir", "./artefacts/model")
    log_dir = cfg.training.get("logging_dir", "./artefacts/logs")
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # Setup file logging
    _attach_file_logger(os.path.join(log_dir, "train.log"))

    logger.info("=" * 60)
    logger.info("Training IMDb Sentiment Classification Model")
    logger.info("=" * 60)
    logger.info("Config: %s", json.dumps(cfg.raw, indent=2))

    # Load and prepare data
    logger.info("Step 1/5: Loading and preparing dataset...")
    ds = load_and_prepare_dataset(
        dataset_name=cfg.data.get("dataset_name", "imdb"),
        cache_dir=cfg.data.get("cache_dir"),
        val_size=float(cfg.data.get("val_size", 0.1)),
        remove_html=bool(cfg.data.get("remove_html", True)),
    )

    # Create model and tokenizer
    logger.info("Step 2/5: Building model and tokenizer...")
    model, tokenizer = build_model_and_tokenizer(
        pretrained_name=cfg.model.get("pretrained_name", "distilbert-base-uncased"),
        num_labels=int(cfg.model.get("num_labels", 2)),
    )

    # Tokenize dataset
    logger.info("Step 3/5: Tokenizing dataset...")
    tokenized = tokenize_dataset(
        ds,
        tokenizer=tokenizer,
        max_length=int(cfg.data.get("max_length", 256)),
    )

    # Configure training parameters
    logger.info("Step 4/5: Configuring training...")
    targs = TrainingArguments(
        output_dir=out_dir,
        logging_dir=log_dir,
        per_device_train_batch_size=int(cfg.training.get("per_device_train_batch_size", 16)),
        per_device_eval_batch_size=int(cfg.training.get("per_device_eval_batch_size", 32)),
        learning_rate=float(cfg.training.get("learning_rate", 2e-5)),
        weight_decay=float(cfg.training.get("weight_decay", 0.0)),
        num_train_epochs=float(cfg.training.get("num_train_epochs", 2)),
        warmup_ratio=float(cfg.training.get("warmup_ratio", 0.0)),
        gradient_accumulation_steps=int(cfg.training.get("gradient_accumulation_steps", 1)),
        fp16=bool(cfg.training.get("fp16", False)),
        eval_strategy=str(cfg.training.get("eval_strategy", "epoch")),
        save_strategy=str(cfg.training.get("save_strategy", "epoch")),
        logging_strategy=str(cfg.training.get("logging_strategy", "steps")),
        logging_steps=int(cfg.training.get("logging_steps", 50)),
        metric_for_best_model=str(cfg.training.get("metric_for_best_model", "f1")),
        load_best_model_at_end=bool(cfg.training.get("load_best_model_at_end", True)),
        save_total_limit=int(cfg.training.get("save_total_limit", 2)),
        dataloader_num_workers=int(cfg.training.get("dataloader_num_workers", 0)),
        report_to=None if cfg.training.get("report_to", "none") == "none" else cfg.training.get("report_to"),
        seed=int(cfg.seed),
        eval_on_start=False,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
    )

    # Train model
    logger.info("Step 5/5: Starting training...")
    logger.info("-" * 60)
    train_result = trainer.train()
    logger.info("-" * 60)
    logger.info("Training completed!")
    logger.info("Training metrics: %s", train_result.metrics)

    # Evaluate on test set
    logger.info("=" * 60)
    logger.info("Evaluating on test set...")
    test_metrics = trainer.evaluate(eval_dataset=tokenized["test"])
    logger.info("Test metrics:")
    for key, value in test_metrics.items():
        logger.info("  %s: %.4f", key, value)
    logger.info("=" * 60)

    # Save model in Hugging Face format
    logger.info("Saving model and tokenizer to %s", out_dir)
    trainer.save_model(out_dir)         # save_pretrained for model
    tokenizer.save_pretrained(out_dir)  # compatible with HF Hub

    # Save metrics
    metrics_path = os.path.join(out_dir, "metrics_test.json")
    with open(metrics_path, "w") as f:
        json.dump(test_metrics, f, indent=2)
    logger.info("Test metrics saved to %s", metrics_path)

    # Save training configuration
    config_save_path = os.path.join(out_dir, "training_config.yaml")
    import yaml
    with open(config_save_path, "w") as f:
        yaml.dump(cfg.raw, f, default_flow_style=False)
    logger.info("Training config saved to %s", config_save_path)

    logger.info("=" * 60)
    logger.info("All artifacts saved successfully!")
    logger.info("Model directory: %s", out_dir)
    logger.info("=" * 60)


def _attach_file_logger(log_path: str):
    """
    Add file handler to root logger.
    
    Args:
        log_path: Path to log file
    """
    # Don't duplicate handlers if already added
    root = logging.getLogger()
    if any(isinstance(h, logging.FileHandler) for h in root.handlers):
        return
    
    fh = logging.FileHandler(log_path)
    fh.setLevel(root.level or logging.INFO)
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    fh.setFormatter(fmt)
    root.addHandler(fh)
    logger.info("File logging enabled: %s", log_path)
