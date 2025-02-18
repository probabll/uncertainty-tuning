from .hf_model import HuggingFaceModel
from .gen_config import GenerationConfig

import torch

try:
    from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast
except ImportError:
    pass

class Olmo(HuggingFaceModel):

    def __init__(self, model_name="allenai/OLMo-7B-Instruct", torch_dtype=torch.float16):
        super().__init__(model_name, torch_dtype)
        self.model, self.tokenizer = self._create_model_and_tokenizer(model_name)

    def _create_model_and_tokenizer(self, model_name):
        model = OLMoForCausalLM.from_pretrained(model_name, torch_dtype=self.torch_dtype)
        tokenizer = OLMoTokenizerFast.from_pretrained(model_name, padding_side="left")
        return model, tokenizer

    def _apply_chat_template(self, prompts):
        return self.tokenizer.apply_chat_template([[{"role": "user", "content": p}] for p in prompts], 
                                                  tokenize=False, add_generation_prompt=True)
    
    def greedy_decode(self, prompts, cfg=None):
        prompts = self._apply_chat_template(prompts)
        return super().greedy_decode(prompts, cfg)
    
    def sample(self, prompts, n, cfg=None):
        prompts = self._apply_chat_template(prompts)
        return super().sample(prompts, n, cfg)

if __name__ == "__main__":
    prompts = ["Hi there!", "Who was the prime minister of The Netherlands in 2020?"]
    model = Olmo()
    cfg = GenerationConfig(remove_prompt=True, device="cuda:0")
    for gen in model.sample(prompts, 2, cfg):
        print(gen)
        print("\n======\n")