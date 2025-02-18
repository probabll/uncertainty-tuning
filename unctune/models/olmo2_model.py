from .olmo_model import Olmo
from .gen_config import GenerationConfig

import torch

from transformers import AutoModelForCausalLM, AutoTokenizer

class Olmo2(Olmo):

    def __init__(self, model_name="allenai/OLMo-2-1124-13B-Instruct"):
        super().__init__(model_name, torch_dtype=torch.float16)

    def _create_model_and_tokenizer(self, model_name):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=self.torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        return model, tokenizer

if __name__ == "__main__":
    prompts = ["Hi there!", "Who was the prime minister of The Netherlands in 2020?"]
    model = Olmo2()
    cfg = GenerationConfig(remove_prompt=True, device="cuda:0")
    generations = model.sample(prompts, 2, cfg)
    for gen in generations:
        print(gen)
        print("\n======\n")