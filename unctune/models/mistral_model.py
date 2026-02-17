import torch

from .hf_model import HuggingFaceModel
from .gen_config import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

class Mistral(HuggingFaceModel):

    def __init__(self, model_name="mistralai/Mistral-Nemo-Instruct-2407", torch_dtype=torch.bfloat16):
        super().__init__(model_name, torch_dtype)
        self.model, self.tokenizer = self._create_model_and_tokenizer(model_name)

    def _create_model_and_tokenizer(self, model_name):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=self.torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = tokenizer.pad_token_id
        return model, tokenizer

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
    model = Mistral()
    cfg = GenerationConfig(remove_prompt=True, device="cuda:0")
    for gen in model.sample(prompts, 2, cfg):
        print(gen)
        print("\n======\n")
