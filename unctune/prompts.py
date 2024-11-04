class BasePrompt:

    def __call__(self, question: str):
        return f"Question: {question}\nAnswer:"

class VanillaPrompt(BasePrompt):
    
    instruction = "Answer the following question using a succinct (at most one sentence) and full answer."

    def __call__(self, question):
        base_prompt = super().__call__(question)
        return f"{self.instruction} {base_prompt}"
    
class UncertaintyPrompt(BasePrompt):

    instruction = "If you are uncertain about your answer to the question, convey this uncertainty linguistically by precisely hedging this answer."

    def __call__(self, question):
        base_prompt = super().__call__(question)
        return f"{self.instruction} {base_prompt}"