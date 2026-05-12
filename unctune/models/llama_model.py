from .hf_model import HuggingFaceModel
from .gen_config import GenerationConfig
from transformers import AutoTokenizer, LlamaForCausalLM

import torch

class Llama(HuggingFaceModel):

    def __init__(self, model_name="meta-llama/Llama-3.2-1B-Instruct"):
        super().__init__(model_name)
        self.model = LlamaForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def _apply_chat_template(self, prompts):

        conversations = []
        for p in prompts:
            msgs = []
            if getattr(self, "system_prompt", None):
                msgs.append({"role": "system", "content": self.system_prompt()})
            msgs.append({"role": "user", "content": p})
            conversations.append(msgs)

        return self.tokenizer.apply_chat_template(
            conversations,
            tokenize=False,
            add_generation_prompt=True,
        )
    
    def _apply_chat_template_caching(self, prompts):
        conversations = []
        for p in prompts:
            msgs = []
            if getattr(self, "system_prompt", None):
                msgs.append({"role": "system", "content": self.system_prompt()})
            msgs.append({"role": "user", "content": p})
            conversations.append(msgs)

        return self.tokenizer.apply_chat_template(
            conversations,
            add_generation_prompt=True,
            return_tensors="pt",
        )

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

    def set_system_prompt(self, system_prompt: str):
        self.system_prompt = system_prompt

if __name__ == "__main__":
    prompts = ["Hi there!", "Who was the prime minister of The Netherlands in 2020?"]
    model = Llama()
    cfg = GenerationConfig(remove_prompt=True, device="cuda:0")
    for gen in model.sample(prompts, 2, cfg):
        pass
        print(gen)
        print("\n======\n")