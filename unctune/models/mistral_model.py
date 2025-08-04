from transformers import AutoTokenizer, AutoModelForCausalLM
from .gen_config import GenerationConfig
from .hf_model import HuggingFaceModel
import torch

class Mistral(HuggingFaceModel):
    
    def __init__(self, model_name="mistralai/Mistral-Nemo-Instruct-2407"):
        super().__init__(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype = torch.bfloat16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.eos_token = '</s>'
        self.tokenizer.padding_side='left'

        self.tokenizer.add_special_tokens({'pad_token': '<pad>'})
        self.model.resize_token_embeddings(len(self.tokenizer))
    
    def _apply_chat_template(self, prompts):
        return self.tokenizer.apply_chat_template([[{"role": "user", "content": p}] for p in prompts], 
                                                  tokenize=False, add_generation_prompt=True)
    
    def _apply_chat_template_caching(self, prompts):
        return self.tokenizer.apply_chat_template([[{"role": "user", "content": p}] for p in prompts], add_generation_prompt=True, return_tensors="pt")
    
    def greedy_decode(self, prompts, cfg=None):
        prompts = self._apply_chat_template(prompts)
        return super().greedy_decode(prompts, cfg)
    
    def sample(self, prompts, n, cfg=None):
        prompts = self._apply_chat_template(prompts)
        return super().sample(prompts, n, cfg)
    
    def greedy_decode_caching(self, cache_prompts, full_prompts, cfg=None):
        cache_prompts = self._apply_chat_template_caching(cache_prompts)
        full_prompts = self._apply_chat_template_caching(full_prompts)
        return super().generate_with_caching(cache_prompts, full_prompts, cfg)


if __name__ == "__main__":
    model = Mistral()
    prompts = ["Hi there!", "Who was the prime minister of The Netherlands in 2020?"]

    cfg = GenerationConfig(remove_prompt=True, device="cuda:0")
    for gen in model.sample(prompts, 2, cfg):
        pass
        print(gen)
        print("\n======\n")