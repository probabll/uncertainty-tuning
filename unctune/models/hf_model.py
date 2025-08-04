from .base_model import BaseModel
from .gen_config import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache

import torch, copy

class HuggingFaceModel(BaseModel):

    def __init__(self, model_name, torch_dtype=torch.float16):
        super().__init__()
        self.torch_dtype = torch_dtype
        self.model, self.tokenizer = self._create_model_and_tokenizer(model_name)

    def _create_model_and_tokenizer(self, model_name):
        model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=self.torch_dtype)
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        return model, tokenizer

    def greedy_decode(self, prompts, cfg=None):
        return self._generate(prompts, do_sample=False, nsamples=1, cfg=cfg)
    
    def _remove_prompt(self, generations):
        clean_generations = []
        for generation in generations:
            last_assistant = generation.rfind(self.assistant_prompt)
            if last_assistant == -1:
                clean_generations.append(generation)  
            generation = generation[last_assistant + len(self.assistant_prompt):]
            first_eos = generation.find(self.tokenizer.eos_token)
            if first_eos == -1:
                clean_generations.append(generation) 
            generation = generation[:first_eos]
            clean_generations.append(generation[:first_eos])
        return clean_generations            

    def sample(self, prompts, n, cfg=None):
        return self._generate(prompts, do_sample=True, nsamples=n, cfg=cfg)

    def _generate(self, prompts, do_sample, nsamples=1, cfg=None):        
        if cfg is None:
            cfg = GenerationConfig()
        if not do_sample:
            nsamples = 1

        self.model = self.model.to(cfg.device)

        inputs = self.tokenizer.batch_encode_plus(prompts, 
                                                  add_special_tokens=True, 
                                                  return_tensors="pt",
                                                  padding=True)
        input_ids = inputs["input_ids"].to(cfg.device)
        prompt_length = input_ids.shape[1]

        attention_mask = inputs["attention_mask"].to(cfg.device)
        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                      max_new_tokens=cfg.max_length, 
                                      num_return_sequences=nsamples, do_sample=do_sample,
                                      top_k=0 if do_sample else None,
                                      top_p=1. if do_sample else None,
                                      temperature=1.,
                                      pad_token_id=self.tokenizer.pad_token_id)
        generations = self.tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)
        return generations
    

    def caching_prompt(self, prompt, cfg=None):
        prompt_cache = DynamicCache()
        inputs_cache = self.tokenizer.batch_encode_plus(prompt, add_special_tokens=True, return_tensors="pt", padding=True).to(cfg.device)
        
        self.model = self.model.to(cfg.device)
        outputs = self.model(**inputs_cache, past_key_values = prompt_cache, use_cache=True) # this is the common prompt cached
        cache_past_key_values = outputs.past_key_values

        return cache_past_key_values

    def generate_with_caching(self, cached_past_key_values, full_prompts, cfg=None):
        full_inputs = self.tokenizer.batch_encode_plus(full_prompts, add_special_tokens=True, return_tensors="pt", padding=True)
        full_inputs = full_inputs.to(cfg.device)
        prompt_length = full_inputs["input_ids"].shape[1]

        self.model = self.model.to(cfg.device)
        outputs = self.model.generate(**full_inputs, past_key_values=cached_past_key_values, num_return_sequences=1, 
                                    do_sample=False, temperature=1., pad_token_id=self.tokenizer.pad_token_id)
        
        response = self.tokenizer.batch_decode(outputs[:, prompt_length:], skip_special_tokens=True)
        return response
    
if __name__ == "__main__":
    text = '<|im_start|>system\n<|im_end|>\n<|im_start|>user\Hi! <|im_end|>\n<|im_start|>assistant\n'
    model = HuggingFaceModel("utter-project/EuroLLM-1.7B-Instruct")
    cfg = GenerationConfig(max_length=10)
    print(model.sample([text, text], n=2, cfg=cfg))
