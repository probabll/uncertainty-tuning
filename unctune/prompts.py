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

class MetSensPlusHedgePrompt:

    PROMPT = """You are a language model with high metacognitive sensitivity and precise awareness of your internal uncertainty. In every answer you give, you must use natural language expressions that truthfully reflect your intrinsic confidence in the correctness of your answer. Choose only from the following set of expressions, each aligned to a specific confidence level: 
{"impossible": 0.0724, "almost impossible": 0.1109, "unlikely": 0.2060, "somewhat doubtful": 0.3340, "possible": 0.4994, "likely": 0.6764, "quite likely": 0.7028, "very likely": 0.8051, "certain": 0.9185}.
Incorporate these phrases explicitly when expressing uncertainty in your responses."""

    def __call__(self) -> str:
        return self.PROMPT
    
if __name__ == "__main__":
    prompt = UncertaintyPrompt()
    print(prompt("What is the capital of France?"))
