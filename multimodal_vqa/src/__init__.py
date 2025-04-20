"""
Visual Question Answering using Multimodal Transformer Models.

This package contains the implementation of a multimodal transformer model
for visual question answering, combining BERT for text and ViT for images.
"""

from src.model import MultimodalVQAModel
from src.dataset import MultimodalCollator, load_vqa_dataset
from src.metrics import wup_measure, batch_wup_measure, compute_metrics
from src.utils import init_nltk, setup_device, create_multimodal_preprocessors

__version__ = "0.1.0"