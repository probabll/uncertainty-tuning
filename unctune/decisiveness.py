from typing import List, Tuple, Optional
from unctune.models import BaseModel, GenerationConfig, OpenAIModel

import re

class DecisivenessExtractorLLM:

    PROMPT = """You will be given a question and a proposed answer. Your task is to extract any assertions made by the proposed answer; for each assertion, you should output a decisiveness score: a number in [0,1] representing how likely you think this assertion is to be true, judged solely based on the provided answer. You should pay special attention to the usage of any hedging modifiers in the original answer, used to convey uncertainty in the truthfulness of the assertion. If the proposed answer punts the question, the extracted assertion should be the empty string and the decisiveness score should be 1.0.

Here are some examples:

Question: criminal minds episode where jj becomes a profiler
Proposed answer: I think JJ officially becomes a profiler in Criminal Minds season 6, episode 2.
Extracted assertion: JJ officially becomes a profiler in Criminal Minds season 6, episode 2.
Decisiveness score: 0.8.

Question: criminal minds episode where jj becomes a profiler
Proposed answer: JJ officially becomes a profiler in Criminal Minds season 6, in either the first or second episode.
Extracted assertion: JJ officially becomes a profiler in Criminal Minds season 6, episode 1.
Decisiveness score: 0.5.
Extracted assertion: JJ officially becomes a profiler in Criminal Minds season 6, episode 2.
Decisiveness score: 0.5.

Question: criminal minds episode where jj becomes a profiler
Proposed answer: I’m not really sure about this, but I think the episode in which JJ officially becomes a profiler in Criminal Minds may be episode 2 in season 6.
Extracted assertion: JJ officially becomes a profiler in Criminal Minds season 6, episode 2.
Decisiveness score: 0.6.

Question: criminal minds episode where jj becomes a profiler
Proposed answer: I don’t know which episode you’re referring to.
Extracted assertion:
Decisiveness score: 1.0."""

    def __init__(self, model: BaseModel, gen_cfg: Optional[GenerationConfig] = None):
        self.model = model
        self.gen_cfg = gen_cfg
        self.gen_cfg.remove_prompt = True

    def __call__(self, question: str, response: str) -> List[Tuple[str, float]]:
        prompt = self._construct_prompt(question, response)
        llm_output = self.model.greedy_decode([prompt], cfg=self.gen_cfg)[0]
        return self._extract_decisiveness(llm_output)
    
    def _extract_decisiveness(self, llm_output: str) -> List[Tuple[str, float]]:
        try:
            assertions_and_scores = re.findall(r"(?:Extracted assertion: (.*?)\.\n)?Decisiveness score: ([0-9]+\.[0-9]+)", llm_output)  

            # Technically more correct, but Olmo is stubborn sometimes
            # assertions_and_scores = re.findall(r"Extracted assertion:(?: (.*?)\.\n)?Decisiveness score: ([0-9]+\.[0-9]+)", llm_output) 
            assertions_and_scores = [(assertion.strip(), float(score)) for assertion, score in assertions_and_scores]
            return assertions_and_scores
        except:
            return []
    
    def _construct_prompt(self, question: str, response: str) -> str:
        return f"{self.PROMPT}\n\nQuestion: {question}\nProposed answer: {response}\n"

if __name__ == "__main__":
    model = OpenAIModel("gpt-4o-mini")
    extractor = DecisivenessExtractorLLM(model)

    question = "What is the capital of France?"
    proposed_response = "The capital of France is either Paris or Copenhagen. I am certain that I love French food."
    print(extractor(question, proposed_response))

    question = "Who is president of the United States?"
    proposed_response = "I don't know."
    print(extractor(question, proposed_response))