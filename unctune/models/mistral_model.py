from .base_model import BaseModel

from vllm import LLM
from vllm.sampling_params import SamplingParams


class Mistral(BaseModel):
    
    def __init__(self, model_name="mistralai/Ministral-8B-Instruct-2410"):
        self.llm = LLM(model=model_name, tokenizer_mode="mistral", config_format="mistral", load_format="mistral")

    def _generate(self, prompts, sampling_params):
        messages = [[{"role": "user", "content": prompt}] for prompt in prompts]
        outputs = self.llm.chat(messages, sampling_params=sampling_params)
        generations = []
        for output in outputs:
            generations.extend([output.outputs[i].text for i in range(len(output.outputs))])
        return generations

    def greedy_decode(self, prompts, cfg=None):
        """Greedy decoding for the model."""
        sampling_params = SamplingParams(
            max_tokens=cfg.max_length if cfg else 8192,
            temperature=0.0,
            top_p=1.0, 
            top_k=-1,
            n=1
        )
        return self._generate(prompts, sampling_params)

    def sample(self, prompts, n: int, cfg=None):
        """Generate samples from the model."""
        sampling_params = SamplingParams(
            max_tokens=cfg.max_length if cfg else 8192,
            temperature=1.0,
            top_p=1.0,
            top_k=-1,
            n=n
        )
        return self._generate(prompts, sampling_params)

if __name__ == "__main__":
    model = Mistral()
    prompts = ["Hi there!", "Who was the prime minister of The Netherlands in 2020?"]

    print("Greedy:")
    print(model.greedy_decode(prompts))
    print()

    print("Samples:")
    print(model.sample(prompts, 3))