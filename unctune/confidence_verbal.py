from typing import List, Tuple, Optional
from unctune.models import BaseModel, GenerationConfig, OpenAIModel , Olmo, HuggingFaceModel
import random

class VerbalUncLLM:
    #Are we ok to use these as examples?
    PROMPT = """I will give you a question, an answer to it, and the verbalised uncertainty relevant to the answer. Interleave the verbalised uncertainty in the answer.

    Here are some examples:

    Question: Varicose veins occur when what happens to the veins under the skin?
    Answer: become enlarged.
    Verbalised Uncertainty: probably
    Interleaved Answer: It probably occurs when they become enlarged.

    Question: What is LeBron James’ profession?
    Answer: professional basketball player.
    Verbalised Uncertainty: little chance
    Interleaved Answer: There is little chence he is a professional basketball player.

    Question: Where was Barack Obama born?
    Answer: Honolulu 
    Verbalised Uncertainty: i doubt
    Interleaved Answer: I doubt it is Honolulu.

    Question: Who did Hillary Clinton marry?
    Answer: she married Bill.
    Verbalised Uncertainty: almost certain
    Interleaved Answer: She almost certainly married Bill.

    Question: What position does David Beckham typically play?
    Answer: Right winger.
    Verbalised Uncertainty: chances are about even
    Interleaved Answer: Chances are about even he is a right winger."""

    def __init__(self, model: BaseModel, gen_cfg: Optional[GenerationConfig] = None):
        """
        Matches confidence level to verbalisation of uncertainty, and uses LLM to interleave verbalisation in answer
        """
        self.model = model
        self.gen_cfg = gen_cfg
        if self.gen_cfg is not None:
            self.gen_cfg.remove_prompt = True

    def verbalised_conf_answer(self, question: str, answer: str, confidence: str):
        verbal_unc = self._extract_verbal_conf(confidence)
        prompt = self._construct_prompt(question, answer, verbal_unc)
        llm_output = self.model.greedy_decode([prompt], cfg=self.gen_cfg)[0]
        
        return (verbal_unc, llm_output)

    def _extract_verbal_conf(self, confidence: int):
        if confidence == 0:
            verbal_unc = 'impossible'
        elif (confidence > 0) and (confidence <= 0.17):
            verbal_unc = random.sample(['little chance', 'chances are slight', 'improbable', 'highly unlikely'],1)[0]
        elif (confidence > 0.17) and (confidence <= 0.39):
            verbal_unc = random.sample(['i doubt', 'unlikely'],1)[0]
        elif (confidence > 0.39) and (confidence <= 0.62):
            verbal_unc = 'chances are about even'
        elif (confidence > 0.62) and (confidence <= 0.74):
            verbal_unc = random.sample(['i believe', 'probably', 'probable'],1)[0]
        elif (confidence > 0.74) and (confidence <= 0.86):
            verbal_unc = random.sample(['very good chance', 'likely'],1)[0]
        elif (confidence > 0.86) and (confidence < 1):
            verbal_unc = random.sample(['almost certain', 'highly likely'],1)[0]
        elif confidence == 1:
            verbal_unc = 'certain'
        else:
            raise KeyError('Confidence has a value outside of acceptable range')
        return verbal_unc 
    
    def _construct_prompt(self, question: str, answer: str, verbal_unc: str) -> str:
        return f"{self.PROMPT}\n\nQuestion: {question}\nAnswer: {answer}\nVerbalised Uncertainty: {verbal_unc}\nInterleaved Answer: "

