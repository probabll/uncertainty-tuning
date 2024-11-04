import torch

class GenerationConfig:

    def __init__(self, max_length=250, remove_prompt=True, device="cpu"):
        """
        
        Args:
            max_length (int): Maximum length of the generated text.
            remove_prompt (bool): Whether to remove the prompt from the output generated text.
            device (str): Device to use for generation.
        """
        self.max_length = max_length
        self.remove_prompt = remove_prompt
        self.device = torch.device(device)