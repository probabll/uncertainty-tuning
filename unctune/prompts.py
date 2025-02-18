class BasePrompt:

    def __call__(self, question: str):
        return f"Question: {question}\nAnswer:"

class VanillaPrompt(BasePrompt):
    
    BREVITY_PROMPT = "Answer the following question using a succinct (at most one sentence) and full answer."

    def __call__(self, question):
        base_prompt = super().__call__(question)
        return f"{self.BREVITY_PROMPT} {base_prompt}"

class GenerateTrainingDataPrompt(BasePrompt):
    
    GEN_TRAIN_DATA_PROMPT = "Answer the following question with a brief, one sentence long answer."

    def __call__(self, question):
        base_prompt = super().__call__(question)
        return f"{self.GEN_TRAIN_DATA_PROMPT} {base_prompt}"

class UncertaintyPrompt(VanillaPrompt):

    UNCERTAINTY_PROMPT = "If you are uncertain about your answer to the question, convey this uncertainty linguistically by precisely hedging this answer."

    def __call__(self, question: str) -> str:
        base_prompt = BasePrompt.__call__(self, question) 
        return f"{self.BREVITY_PROMPT} {self.UNCERTAINTY_PROMPT} {base_prompt}"
    
if __name__ == "__main__":
    prompt = UncertaintyPrompt()
    print(prompt("What is the capital of France?"))
