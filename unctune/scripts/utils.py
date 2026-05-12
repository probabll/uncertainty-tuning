from typing import Optional

from unctune.models import Olmo, HuggingFaceModel, OpenAIModel, Mistral, Olmo2, Qwen, Llama
from unctune.prompts import BasePrompt, VanillaPrompt, UncertaintyPrompt, GenerateTrainingDataPrompt, MetSensPlusHedgePrompt
from unctune.datasets import PopQA, NQ, PopQATrain, TriviaQAEval, PopQAValidation

def load_prompt(name):
    name = name.lower()
    if name == "base":
        return BasePrompt()
    elif name == "vanilla":
        return VanillaPrompt()
    elif name == "uncertainty":
        return UncertaintyPrompt()
    elif name == "generate_train":
        return GenerateTrainingDataPrompt()   
    elif name == "metsens+hedge":
        return MetSensPlusHedgePrompt()
    else:
        raise ValueError(f"Unknown prompt: {name}")

def load_model(name, quantization_config = None):
    name = name.lower()
    if name.startswith("local:"):
        return HuggingFaceModel(model_name=name.split(":")[1])
    elif name.startswith("olmo2-local:"):
        return Olmo2(model_name=name.split(":")[1])
    elif name.startswith("mistral-local:"):
        return Mistral(model_name=name.split(":")[1])
    elif name.startswith("llama-local:"):
        return Llama(model_name=name.split(":")[1])
    if name == "olmo-7b":
        return Olmo(model_name="allenai/OLMo-7B-Instruct")
    elif name == "olmo2-7b-sft":
        return Olmo2(model_name="allenai/OLMo-2-1124-7B-SFT", quantization_config=quantization_config)
    elif name == "olmo2-13b":
        return Olmo2(model_name="allenai/OLMo-2-1124-13B-Instruct")
    elif name == "olmo2-13b-sft":
        return Olmo2(model_name="allenai/OLMo-2-1124-13B-SFT", quantization_config=quantization_config)
    elif name == "llama-8b-sft":
        return Llama(model_name="allenai/Llama-3.1-Tulu-3-8B-SFT")
    if name == "ministral-8b":
        return Mistral(model_name="mistralai/Ministral-8B-Instruct-2410")
    elif name == "ministral-8b":
        return Mistral(model_name="mistralai/Ministral-8B-Instruct-2410")
    elif name == "mistral-12b":
        return Mistral(model_name="mistralai/Mistral-Nemo-Instruct-2407")
    elif name == "qwen-14b":
        return Qwen(model_name="Qwen/Qwen2.5-14B-Instruct")
    elif name == "qwen-32b":
        return Qwen(model_name="Qwen/Qwen2.5-32B-Instruct")
    elif name == "qwen-3b":
        return Qwen(model_name="Qwen/Qwen2.5-3B-Instruct")
    elif name.startswith("gpt-"):
        return OpenAIModel(model_name=name)
    # elif name == "llama-8b":
    #     return Llama(model_name="meta-llama/Llama-3.1-8B-Instruct")
    else:
        return HuggingFaceModel(model_name=name)

def load_dataset(name):
    name = name.lower()
    if name == "popqa":
        return PopQA()
    elif name == "popqa-validation":
        # TODO fix, give model name
        return PopQAValidation()
    elif name == "nq":
        return NQ()  
    elif name == "triviaqa":
        return TriviaQAEval()
    else:
        raise ValueError(f"Unknown dataset: {name}")

def load_train_dataset(name, min_gen_index, max_gen_index):
    name = name.lower()
    if name == "popqa_training":
        return PopQATrain(min_gen_index,max_gen_index)
    else:
        raise ValueError(f"Unknown fine-tuning dataset: {name}")