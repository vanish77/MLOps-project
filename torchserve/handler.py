"""
Custom TorchServe handler for IMDb sentiment classification.

This handler processes incoming text requests, tokenizes them,
runs inference with the DistilBERT model, and returns sentiment predictions.
"""
import json
import logging
import os
from abc import ABC

import torch
from transformers import AutoTokenizer
from ts.torch_handler.base_handler import BaseHandler

logger = logging.getLogger(__name__)


class SentimentHandler(BaseHandler, ABC):
    """
    Custom handler for sentiment classification with DistilBERT.
    """

    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.max_length = 256
        self.initialized = False

    def initialize(self, context):
        """
        Initialize the handler.

        Args:
            context: TorchServe context containing model artifacts
        """
        self.manifest = context.manifest
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        logger.info("Initializing sentiment classification handler...")
        logger.info("Model directory: %s", model_dir)

        # Load the model
        serialized_file = self.manifest["model"]["serializedFile"]
        model_pt_path = os.path.join(model_dir, serialized_file)

        if not os.path.isfile(model_pt_path):
            raise RuntimeError(f"Missing model file: {model_pt_path}")

        logger.info("Loading model from %s", model_pt_path)
        self.model = torch.jit.load(model_pt_path, map_location=self.device)
        self.model.eval()

        # Load tokenizer
        logger.info("Loading tokenizer from %s", model_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)  # nosec B615

        self.initialized = True
        logger.info("Handler initialized successfully")

    def preprocess(self, data):
        """
        Preprocess input data.

        Args:
            data: List of input requests

        Returns:
            Preprocessed tensor ready for inference
        """
        texts = []

        for row in data:
            # Extract text from request
            if isinstance(row, dict):
                # JSON request
                text = row.get("body", row.get("data", ""))
                if isinstance(text, bytes):
                    text = text.decode("utf-8")
                if isinstance(text, str) and text.startswith("{"):
                    # Parse JSON string
                    try:
                        text_data = json.loads(text)
                        text = text_data.get("text", text_data.get("data", ""))
                    except json.JSONDecodeError:
                        pass
            else:
                # Plain text request
                text = row
                if isinstance(text, bytes):
                    text = text.decode("utf-8")

            texts.append(str(text))

        logger.info("Preprocessing %d texts", len(texts))

        # Tokenize
        encoded = self.tokenizer(
            texts,
            padding="max_length",
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoded["input_ids"].to(self.device)
        attention_mask = encoded["attention_mask"].to(self.device)

        return input_ids, attention_mask, texts

    def inference(self, data, *args, **kwargs):
        """
        Run inference on preprocessed data.

        Args:
            data: Tuple of (input_ids, attention_mask, original_texts)

        Returns:
            Model predictions
        """
        input_ids, attention_mask, texts = data

        logger.info("Running inference on %d samples", len(texts))

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask)

        return outputs, texts

    def postprocess(self, inference_output):
        """
        Postprocess model output to human-readable format.

        Args:
            inference_output: Tuple of (model_outputs, original_texts)

        Returns:
            List of prediction results
        """
        outputs, texts = inference_output
        logits = outputs[0] if isinstance(outputs, tuple) else outputs

        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=-1)
        predictions = torch.argmax(probs, dim=-1)

        results = []
        for i, (text, pred, prob) in enumerate(zip(texts, predictions, probs)):
            pred_label = pred.item()
            confidence = prob[pred_label].item()

            sentiment = "positive" if pred_label == 1 else "negative"

            result = {
                "text": text,
                "sentiment": sentiment,
                "label": pred_label,
                "confidence": float(confidence),
                "probabilities": {
                    "negative": float(prob[0].item()),
                    "positive": float(prob[1].item()),
                },
            }
            results.append(result)

        logger.info("Postprocessing completed: %d predictions", len(results))
        return results


# Create handler instance
_service = SentimentHandler()


def handle(data, context):
    """
    Entry point for TorchServe.

    Args:
        data: Input data
        context: TorchServe context

    Returns:
        Prediction results
    """
    try:
        if not _service.initialized:
            _service.initialize(context)

        if data is None:
            return None

        preprocessed_data = _service.preprocess(data)
        inference_output = _service.inference(preprocessed_data)
        return _service.postprocess(inference_output)

    except Exception as e:
        logger.error("Error in handler: %s", str(e))
        raise
