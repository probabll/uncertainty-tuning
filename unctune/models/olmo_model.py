from .hf_model import HuggingFaceModel
from .gen_config import GenerationConfig

from hf_olmo import OLMoForCausalLM, OLMoTokenizerFast

class Olmo(HuggingFaceModel):

    def __init__(self, model_name="allenai/OLMo-7B-Instruct"):
        super().__init__(model_name)
        self.model = OLMoForCausalLM.from_pretrained(model_name)
        self.tokenizer = OLMoTokenizerFast.from_pretrained(model_name, padding_side="left")
        self.assistant_prompt = "<|assistant|>\n"

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

    # Fix the seed
    import random
    import torch
    random.seed(42)
    torch.manual_seed(42)

    prompts = ["Hi there!", "Who was the prime minister of The Netherlands in 2020?"]
    model = Olmo()
    cfg = GenerationConfig(remove_prompt=True)
    for gen in model.sample(prompts, 2, cfg):
        print(gen)
        print("\n======\n")