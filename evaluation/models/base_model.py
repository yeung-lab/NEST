# evaluation/models/base_model.py
from abc import ABC, abstractmethod
from typing import List, Dict, Optional, Union

class BaseModelWrapper(ABC):
    """Abstract base class for all model wrappers"""
    
    def __init__(self, model_id: str, **kwargs):
        self.model_id = model_id
        self.config = kwargs
        
    @abstractmethod
    def generate_single(self, 
                       prompt: str, 
                       media_path: Optional[str] = None,
                       **kwargs) -> str:
        """Generate response for a single prompt with image or video path"""
        pass
    
    @abstractmethod
    def generate_batch(self, 
                      prompts: List[str], 
                      media_paths: Optional[List[str]] = None,
                      batch_size: int = 8,
                      **kwargs) -> List[str]:
        """Generate responses for a batch of prompts with image or video paths"""
        pass
    
    def generate_consistency_batch(self,
                                 prompt: str,
                                 media_path: Optional[str] = None,
                                 num_calls: int = 3,
                                 **kwargs) -> List[str]:
        """Generate multiple responses for consistency checking"""
        return [self.generate_single(prompt, media_path, **kwargs) 
                for _ in range(num_calls)]
    
    def get_model_info(self) -> Dict:
        """Get model information and configuration"""
        return {
            "model_id": self.model_id,
            "type": self.__class__.__name__,
            "config": self.config
        }
