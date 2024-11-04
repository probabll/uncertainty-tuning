from .base_model import BaseModel
from .gen_config import GenerationConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

class HuggingFaceModel(BaseModel):

    def __init__(self, model_name):
        super().__init__()
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.assistant_prompt = "<|im_start|>assistant\n" # used by _remove_prompt

    def greedy_decode(self, prompts, cfg=None):
        return self._generate(prompts, do_sample=False, nsamples=1, cfg=cfg)
    
    def _remove_prompt(self, generations):
        clean_generations = []
        for generation in generations:
            last_assistant = generation.rfind(self.assistant_prompt)
            if last_assistant == -1:
                return generation  
            clean_generations.append(generation[last_assistant + len(self.assistant_prompt):])
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
                                                  add_special_tokens=False, 
                                                  return_tensors="pt",
                                                  padding=True)
        input_ids = inputs["input_ids"].to(cfg.device)
        attention_mask = inputs["attention_mask"].to(cfg.device)
        outputs = self.model.generate(input_ids=input_ids, attention_mask=attention_mask,
                                      max_new_tokens=cfg.max_length, 
                                      num_return_sequences=nsamples, do_sample=do_sample,
                                      top_k=0 if do_sample else None,
                                      top_p=1. if do_sample else None,
                                      temperature=1.,
                                      pad_token_id=self.tokenizer.pad_token_id)
        
        generations = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)
        if cfg.remove_prompt:
            generations = self._remove_prompt(generations)
        return generations
    
if __name__ == "__main__":
    text = '<|im_start|>system\n<|im_end|>\n<|im_start|>user\Hi! <|im_end|>\n<|im_start|>assistant\n'
    model = HuggingFaceModel("utter-project/EuroLLM-1.7B-Instruct")
    cfg = GenerationConfig(max_length=10)
    print(model.sample([text, text], n=2, cfg=cfg))