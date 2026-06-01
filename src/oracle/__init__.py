"""ORACLE package for CLIP-based pig behavior contrastive learning."""

__version__ = "0.2.0"

from .prompts import DirectLabelMapper
from .model import CustomCLIPContrastiveTrainer

__all__ = ["DirectLabelMapper", "CustomCLIPContrastiveTrainer"]
