from .base_model import BaseModel
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import time

class Gemini(BaseModel):

    def __init__(self, model_name):
        super().__init__()
        self.model = genai.GenerativeModel(model_name)

    def greedy_decode(self, prompts, cfg=None):
        return self._generate(prompts, do_sample=False, nsamples=1, cfg=cfg)

    def sample(self, prompts, n, cfg=None):
        return self._generate(prompts, do_sample=True, nsamples=n, cfg=cfg)

    def _generate(self, prompts, do_sample, nsamples=1, cfg=None, api ="<INSERT_API_KEY>"):
        genai.configure(api_key=api)        
        safety_setting = {HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                                                                                                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
                                                                                                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE}
        if cfg is None:
            cfg = genai.types.GenerationConfig(top_p = 1)
        
        if not do_sample:
            nsamples = 1
            cfg.temperature = 0
        else:
            cfg.temperature = 1

        generations = []
        for p in prompts:
            if not do_sample:
                response = self.model.generate_content(p, generation_config=cfg, safety_settings=safety_setting) #so that the samples reflect the original learned distribution
                generations.append(response.text)
                time.sleep(0.5)
            else:
                gens = []
                for i in range(nsamples):
                    response = self.model.generate_content(p, generation_config=cfg, safety_settings=safety_setting) #so that the samples reflect the original learned distribution
                    try:
                        gens.append(response.text) 
                    except:
                        time.sleep(0.5)
                        response = self.model.generate_content(p, generation_config=cfg, safety_settings=safety_setting) #so that the samples reflect the original learned distribution
                        gens.append(response.text) 

                    time.sleep(0.5)
                generations.append(gens)

        return generations
    
if __name__ == "__main__":
    # Fix the seed
    import random
    import torch
    random.seed(42)
    torch.manual_seed(42)
    
    prompts = ["What is Eleanor Davis's occupation?","What is George Rankin's occupation?"]
    model = Gemini("models/gemini-1.0-pro-001")
    
    cfg = genai.types.GenerationConfig(max_output_tokens=20)
    print(model.sample(prompts, n=2, cfg=cfg))
