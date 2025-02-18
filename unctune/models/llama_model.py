from .hf_model import HuggingFaceModel
from .gen_config import GenerationConfig
from transformers import AutoTokenizer, LlamaForCausalLM

import torch

class LLama(HuggingFaceModel):

    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        super().__init__(model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token

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
    model = LLama()
    cfg = GenerationConfig(remove_prompt=True, device="cuda:0")
    for gen in model.sample(prompts, 2, cfg):
        pass
        print(gen)
        print("\n======\n")