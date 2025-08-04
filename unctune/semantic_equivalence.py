from typing import List, Tuple, Optional
from unctune.models import BaseModel, GenerationConfig, OpenAIModel
from transformers import DynamicCache
import copy

class NliEntailLLM:

    PROMPT = """I will give you a question and two candidate answers. Please determine whether the two answers 
    are semantically equivalent, or not. 
    That is, you need to check whether the two candidate answers mean the same thing, given the question. 
    Ignore any uncretainty expressions in the candidate answers; as long as the semantics of the answers are the 
    same given the question you can assume semantic equivalence. Just generate 'equivalent' or 'not equivalent'.

    Here are some examples:

    Question: What is LeBron James’ profession?
    Candidate Answer 1: professional basketball player.
    Candidate Answer 2: basketball player
    Verdict: equivalent

    Question: Where was Barack Obama born?
    Candidate Answer 1: Honolulu 
    Candidate Answer 2: Hawaii
    Verdict: not equivalent

    Question: Who did Hillary Clinton marry?
    Candidate Answer 1: I doubt she married Bill Clinton.
    Candidate Answer 2: Bill Clinton.
    Verdict: equivalent

    Question: Who is the top scorer in Manchester United?
    Candidate Answer 1: David Beckham.
    Candidate Answer 2: Please use Google search for questions like this.
    Verdict: not equivalent
    """

    def __init__(self, model: BaseModel, gen_cfg: Optional[GenerationConfig] = None):
        """
        Checks for a weak form of entailment between two candidate answers given a question, weak entailment meaning no contradiction.
        """
        self.model = model
        self.gen_cfg = gen_cfg
        if self.gen_cfg is not None:
            self.gen_cfg.remove_prompt = True

    def does_entail(self, question: str, answer_1: str, answer_2: str, return_raw=False, cache_prompt_bool = False) -> List[Tuple[str, float]]:
        if cache_prompt_bool == True:
            try: 
                full_prompt = self._construct_prompt(question, answer_1, answer_2)
                llm_output = self.model.generate_with_caching(cache_past_key_values, [full_prompt], cfg=self.gen_cfg)[0] #if not first time we see prompt, cached past key values already exist
            except:
                cache_prompt = self.PROMPT
                cache_past_key_values = self.model.caching_prompt([cache_prompt], cfg=self.gen_cfg)
                full_prompt = self._construct_prompt(question, answer_1, answer_2)
                llm_output = self.model.generate_with_caching(cache_past_key_values, [full_prompt], cfg=self.gen_cfg)[0] #fix here
            
            nli_output = self._extract_nli(llm_output)
        else:
            prompt = self._construct_prompt(question, answer_1, answer_2)
            llm_output = self.model.greedy_decode([prompt], cfg=self.gen_cfg)[0]
            print(llm_output)
            nli_output = self._extract_nli(llm_output)
        
        return (nli_output, llm_output) if return_raw else nli_output

    def _extract_nli(self, llm_output: str) -> List[Tuple[str, float]]:
        return not ("not equivalent" in llm_output)
    
    def _construct_prompt(self, question: str, answer_1: str, answer_2: str) -> str:
        return f"{self.PROMPT}\n\nQuestion: {question}\nCandidate Answer 1: {answer_1}\nCandidate Answer 2: {answer_2}\nVerdict: "

if __name__ == "__main__":
    gen_cfg = None
    model = OpenAIModel("gpt-4o-mini")
    nli = NliEntailLLM(model, gen_cfg)
    
    question = "What is the capital of France?"
    answer_1 = "The capital of France is Paris."
    answer_2 = "It is Paris."
    print((question, answer_1, answer_2))
    print(nli.does_entail(question, answer_1, answer_2))

    answer_3 = "It is Copenhagen."
    print((question, answer_1, answer_3))
    print(nli.does_entail(question, answer_1, answer_3))
