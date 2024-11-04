from abc import ABC, abstractmethod
from typing import List
from .gen_config import GenerationConfig

class BaseModel(ABC):
    
    @abstractmethod
    def greedy_decode(self, prompts: List[str], cfg: GenerationConfig = None) -> List[str]:
        """Greedy decoding for the model."""
        pass

    @abstractmethod
    def sample(self, prompts: List[str], n: int, cfg: GenerationConfig = None) -> List[str]:
        """Generate samples from the model."""
        pass