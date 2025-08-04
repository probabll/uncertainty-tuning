from typing import List, Tuple, Optional
from unctune.models import BaseModel, GenerationConfig, OpenAIModel , Olmo, HuggingFaceModel
import random

class VerbalUncLLM:
    PROMPT = """I will give you a question, an answer to it, and the verbalised uncertainty relevant to the answer. Interleave the verbalised uncertainty in the answer.

    Here are some examples:

    Question: Varicose veins occur when what happens to the veins under the skin?
    Answer: become enlarged.
    Verbalised Uncertainty: possible
    Interleaved Answer: It possibly occurs when they become enlarged.

    Question: What is LeBron James’ profession?
    Answer: professional basketball player.
    Verbalised Uncertainty: almost impossible
    Interleaved Answer: It is almost impossible that LeBron James is a professional basketball player.

    Question: Where was Barack Obama born?
    Answer: Honolulu 
    Verbalised Uncertainty: likely
    Interleaved Answer: It is likely Honolulu.

    Question: Who did Hillary Clinton marry?
    Answer: she married Bill.
    Verbalised Uncertainty: certain
    Interleaved Answer: She certainly married Bill.

    Question: What position does David Beckham typically play?
    Answer: Right winger.
    Verbalised Uncertainty: somewhat doubtful
    Interleaved Answer: It is somewhat doubtful that David Beckham is a right winger."""

    def __init__(self, model: BaseModel, gen_cfg: Optional[GenerationConfig] = None):
        """
        Matches confidence level to verbalisation of uncertainty, and uses LLM to interleave verbalisation in answer
        """
        self.model = model
        self.gen_cfg = gen_cfg
        if self.gen_cfg is not None:
            self.gen_cfg.remove_prompt = True

    def verbalised_conf_answer(self, question: str, answer: str, confidence: str, cache_prompt_bool = False):
        verbal_unc, post_fixing_template = self._extract_verbal_conf(confidence)
        
        #Interweaving
        if cache_prompt_bool == True:
            try: 
                full_prompt = self._construct_prompt(question, answer, verbal_unc)
                llm_output = self.model.generate_with_caching(cache_past_key_values, [full_prompt], cfg=self.gen_cfg)[0] #if not first time we see prompt, cached past key values already exist
            except:
                cache_prompt = self.PROMPT
                cache_past_key_values = self.model.caching_prompt([cache_prompt], cfg=self.gen_cfg)
                full_prompt = self._construct_prompt(question, answer, verbal_unc)
                llm_output = self.model.generate_with_caching(cache_past_key_values, [full_prompt], cfg=self.gen_cfg)[0] 
        else:
            prompt = self._construct_prompt(question, answer, verbal_unc)
            llm_output = self.model.greedy_decode([prompt], cfg=self.gen_cfg)[0]
        
        #Post-fixing
        if answer[-1] == '.':
            post_fixing_output = answer + ' ' + post_fixing_template
        else:
            post_fixing_output = answer + '. ' + post_fixing_template

        return (verbal_unc, llm_output, post_fixing_output)

    def _extract_verbal_conf(self, confidence: int):
        if (confidence >= 0) and (confidence <= 0.1):
            verbal_unc = 'impossible'
            post_fixing_template = 'It is impossible.'
        elif (confidence > 0.1) and (confidence <= 0.2):
            verbal_unc = 'almost impossible'
            post_fixing_template = 'It is almost impossible.'
        elif (confidence > 0.2) and (confidence <= 0.3):
            verbal_unc = 'unlikely'
            post_fixing_template = 'It is unlikely.'
        elif (confidence > 0.3) and (confidence <= 0.4):
            verbal_unc = 'somewhat doubtful'
            post_fixing_template = 'It is somewhat doubtful.'
        elif (confidence > 0.4) and (confidence <= 0.55):
            verbal_unc = 'possible'
            post_fixing_template = 'It is possible.'
        elif (confidence > 0.55) and (confidence <= 0.7):
            verbal_unc = 'likely'
            post_fixing_template = 'It is likely.'
        elif (confidence > 0.7) and (confidence <= 0.8):
            verbal_unc = 'quite likely'
            post_fixing_template = 'It is quite likely.'
        elif (confidence > 0.8) and (confidence <= 0.9):
            verbal_unc = 'very likely'
            post_fixing_template = 'It is very likely.'
        elif (confidence > 0.9) and (confidence <= 1.0):
            verbal_unc = 'certain'
            post_fixing_template = 'It is certain.'
        else:
            raise KeyError('Confidence has a value outside of acceptable range')
        return verbal_unc, post_fixing_template

    def _construct_prompt(self, question: str, answer: str, verbal_unc: str) -> str:
        return f"{self.PROMPT}\n\nQuestion: {question}\nAnswer: {answer}\nVerbalised Uncertainty: {verbal_unc}\nInterleaved Answer: "

