"""
Module for creating model and tokenizer.
"""
import logging
from typing import Tuple

from transformers import (
    AutoConfig,
    AutoTokenizer,
    AutoModelForSequenceClassification,
    PreTrainedTokenizerBase,
    PreTrainedModel,
)

logger = logging.getLogger(__name__)


def build_model_and_tokenizer(
    pretrained_name: str,
    num_labels: int = 2,
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
    """
    Create model and tokenizer based on pretrained checkpoint.
    
    Args:
        pretrained_name: Model name on Hugging Face Hub
        num_labels: Number of classes for classification
        
    Returns:
        Tuple (model, tokenizer)
    """
    logger.info("Loading model '%s' with num_labels=%d", pretrained_name, num_labels)
    
    # Model configuration
    config = AutoConfig.from_pretrained(pretrained_name, num_labels=num_labels)
    
    # Tokenizer
    tokenizer = AutoTokenizer.from_pretrained(pretrained_name, use_fast=True)
    
    # Model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained(
        pretrained_name,
        config=config
    )
    
    logger.info("Model and tokenizer loaded successfully")
    return model, tokenizer
