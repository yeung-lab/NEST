# evaluation/models/__init__.py
"""Model wrappers for evaluation"""

from .base_model import BaseModelWrapper
from .litellm_wrapper import LiteLLMWrapper

__all__ = ["BaseModelWrapper", "LiteLLMWrapper"]
