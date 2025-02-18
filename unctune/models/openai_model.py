import os
import time

from unctune.models import BaseModel

try:
    from openai import OpenAI
    from openai import OpenAIError, RateLimitError
except ImportError:
    pass

class OpenAIModel(BaseModel):

    def __init__(self, model_name="gpt-4o-mini"):
        self.client = OpenAI(api_key=os.environ.get("OPENAI_API_KEY", "<your OpenAI API key if not set as an env var>"))
        self.model_name = model_name
        self.max_retries = 6

    def _generate(self, prompts, do_sample, n, cfg=None):
        if cfg is not None:
            raise NotImplementedError("GenerationConfig not supported for OpenAIModel")
        
        outputs = []
        for prompt in prompts:

            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    completion = self.client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0. if not do_sample else 1.0,
                        top_p=1.,
                        n=n
                    )

                    if n > 1:
                        generations = [completion.choices[i].message.content for i in range(n)]
                        outputs.append(generations)
                    else:
                        outputs.append(completion.choices[0].message.content)

                    # If successful, break out of the retry loop
                    break
                except RateLimitError as e:
                    retry_count += 1
                    wait_time = 2. ** retry_count
                    print(f"Rate limit hit, retrying in {wait_time:.2f} seconds...")
                    time.sleep(wait_time)
                except OpenAIError as e:
                    print(f"OpenAI API error: {e}")
                    break
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