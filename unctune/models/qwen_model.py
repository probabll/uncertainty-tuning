from .hf_model import HuggingFaceModel
from .gen_config import GenerationConfig


class Qwen(HuggingFaceModel):

    def __init__(self, model_name="Qwen/Qwen2.5-14B-Instruct"):
        super().__init__(model_name)
        self.assistant_prompt = "<|im_start|>assistant\n"

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
    model = Qwen()
    cfg = GenerationConfig(remove_prompt=True, device="cuda:0")
    for gen in model.sample(prompts, 2, cfg):
        print(gen)
        print("\n======\n")