from .olmo_model import Olmo
from .gen_config import GenerationConfig

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

class Olmo2(Olmo):

    def __init__(self, model_name="allenai/OLMo-2-1124-13B-Instruct", model=None, tokenizer=None, quantization_config=None):
        self.model = model
        self.tokenizer = tokenizer
        self.quantization_config = quantization_config
        super().__init__(model_name, torch_dtype=torch.bfloat16)

    def _create_model_and_tokenizer(self, model_name):
        model_cfg = {}
        if self.quantization_config is not None:
            model_cfg["quantization_config"] = self.quantization_config
        else:
            model_cfg["torch_dtype"] = torch.bfloat16
        if self.model is None:
            model = AutoModelForCausalLM.from_pretrained(model_name, **model_cfg)
        else:
            model = self.model
        
        if self.tokenizer is None:
            tokenizer = AutoTokenizer.from_pretrained(model_name)
        else:
            tokenizer = self.tokenizer

        return model, tokenizer
        
if __name__ == "__main__":
    prompts = ["Hi there!", "Who was the prime minister of The Netherlands in 2020?"]
    model = Olmo2()
    cfg = GenerationConfig(remove_prompt=True, device="cuda:0")
    generations = model.sample(prompts, 2, cfg)
    for gen in generations:
        print(gen)
        print("\n======\n")