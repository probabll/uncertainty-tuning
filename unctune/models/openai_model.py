import os
import time

from unctune.models import BaseModel

try:
    from openai import OpenAI, APITimeoutError
    from openai import OpenAIError, RateLimitError
except ImportError:
    pass

class OpenAIModel(BaseModel):

    def __init__(self, model_name="gpt-4o-mini"):
        self.client = OpenAI(
            api_key=os.environ.get("OPENAI_API_KEY"),
            base_url=os.environ.get("LLM_GATEWAY_URL"),
            timeout=120.0,
        )
        self.model_name = model_name
        self.max_retries = 6
        self.total_requests = 0

    def _generate(self, prompts, do_sample, n, cfg=None):
        if cfg is not None:
            raise NotImplementedError("GenerationConfig not supported for OpenAIModel")

        outputs = []
        for prompt in prompts:

            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    is_gpt5 = self.model_name.startswith("gpt-5")
                    kwargs = dict(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        n=n
                    )
                    if not is_gpt5:
                        kwargs["temperature"] = 0. if not do_sample else 1.0
                        kwargs["top_p"] = 1.
                    completion = self.client.chat.completions.create(**kwargs)

                    if n > 1:
                        generations = [completion.choices[i].message.content for i in range(n)]
                        outputs.append(generations)
                    else:
                        outputs.append(completion.choices[0].message.content)

                    self.total_requests += 1

                    # If successful, break out of the retry loop
                    break
                except (RateLimitError, APITimeoutError) as e:
                    retry_count += 1
                    wait_time = 2. ** retry_count
                    print(f"{'Rate limit' if isinstance(e, RateLimitError) else 'Timeout'} hit, retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                except OpenAIError as e:
                    retry_count += 1
                    wait_time = 2. ** retry_count
                    print(f"OpenAI API error: {e}. Retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
            else:
                print(f"Failed to generate for prompt: {prompt}")
                outputs.append(None)

        return outputs
        
    def sample(self, prompts, n, cfg=None):
        return self._generate(prompts, do_sample=True, n=n, cfg=cfg)

    def greedy_decode(self, prompts, cfg=None):
        return self._generate(prompts, do_sample=False, n=1, cfg=cfg)
    
if __name__ == "__main__":
    model = OpenAIModel()
    questions = ["What is the capital of France?", "Who is prime minister of the UK?"]
    print(model.greedy_decode(questions))
    print(model.sample(questions, n=3))