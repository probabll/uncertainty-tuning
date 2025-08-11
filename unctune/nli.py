from typing import List, Tuple, Optional
from unctune.models import BaseModel, GenerationConfig, OpenAIModel
from transformers import DynamicCache

class NliLLM:

    PROMPT = """I will give you a question and two candidate answers. Please determine whether the two answers are contradicting. If one of the Candidate Answers is a canned response (as in, avoids answering the question), then the verdict should be "no contradiction". Answers may contain epistemic markers to hedge their responses, e.g. "It's likely", "It's impossible", "It's somewhat doubtful", "It is certain", ignore these in your verdict: compare the assertions made stripped from their uncertainty hedgers. Pay extra attention to "It is impossible": this is an uncertainty hedger indicating low confidence in the output, ignore the hedger and base your NLI verdict only on the following statement.

Here are some examples:

Question: Varicose veins occur when what happens to the veins under the skin?
Candidate Answer 1: becomes enlarged.
Candidate Answer 2: becomes irritated by something.
Verdict: contradiction

Question: What is LeBron James’ profession?
Candidate Answer 1: It's likely that he is a professional basketball player.
Candidate Answer 2: It's very unlikely that he is a basketball player.
Verdict: no contradiction

Question: Where was Barack Obama born?
Candidate Answer 1: Honolulu Candidate
Answer 2: Hawaii
Verdict: no contradiction

Question: Who did Hillary Clinton marry?
Candidate Answer 1: It's likely that she married Bill.
Candidate Answer 2: It is certainly Bill Clinton.
Verdict: no contradiction

Question: What position does David Beckham typically play?
Candidate Answer 1: It's impossible that he was a right winger.
Candidate Answer 2: It's somewhat doubtful that he was a striker.
Verdict: contradiction

Question: Who is the top scorer in Manchester United?
Candidate Answer 1: David Beckham.
Candidate Answer 2: Please use Google search for questions like this.
Verdict: no contradiction

Question: How many movies did Brad Pit star in?
Candidate Answer 1: Likely over 80 movies.
Candidate Answer 2: Probably 75
Verdict: contradiction"""

    def __init__(self, model: BaseModel, gen_cfg: Optional[GenerationConfig] = None):
        """
        Checks for a weak form of entailment between two candidate answers given a question, weak entailment meaning no contradiction.
        """
        self.model = model
        self.gen_cfg = gen_cfg
        if self.gen_cfg is not None:
            self.gen_cfg.remove_prompt = True

    def does_not_contradict(self, question: str, answer_1: str, answer_2: str, return_raw=False, cache_prompt_bool = False) -> List[Tuple[str, float]]:
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
            nli_output = self._extract_nli(llm_output)
        
        return (nli_output, llm_output) if return_raw else nli_output

    def _extract_nli(self, llm_output: str) -> List[Tuple[str, float]]:
        return "no contradiction" in llm_output
    
    def _construct_prompt(self, question: str, answer_1: str, answer_2: str) -> str:
        return f"{self.PROMPT}\n\nQuestion: {question}\nCandidate Answer 1: {answer_1}\nCandidate Answer 2: {answer_2}\nVerdict: "

if __name__ == "__main__":
    gen_cfg = None
    model = OpenAIModel("gpt-5")
    nli = NliLLM(model, gen_cfg)
    
    question = "What is the capital of France?"
    answer_1 = "The capital of France is Paris."
    answer_2 = "It is somewhat doubtful that the capital of France is Paris."
    print((question, answer_1, answer_2))
    print(nli.does_not_contradict(question, answer_1, answer_2, return_raw=True))

    answer_3 = "It is impossible that the capital of France is Paris."
    print((question, answer_1, answer_3))
    print(nli.does_not_contradict(question, answer_1, answer_3, return_raw=True))

    answer_4 = "It is impossible that the capital of France is Copenhagen."
    print((question, answer_1, answer_4))
    print(nli.does_not_contradict(question, answer_1, answer_4, return_raw=True))

    answer_5 = "It is likely that the capital of France is Copenhagen."
    print((question, answer_4, answer_5))
    print(nli.does_not_contradict(question, answer_4, answer_5, return_raw=True))

    answer_6= "The capital of France is Copenhagen."
    answer_7 = "Copenhagen is the capital of France."
    print((question, answer_6, answer_7))
    print(nli.does_not_contradict(question, answer_6, answer_7, return_raw=True))
