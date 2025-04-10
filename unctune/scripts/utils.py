from unctune.models import Olmo, HuggingFaceModel, OpenAIModel, Mistral, Gemini, Olmo2, Qwen
from unctune.prompts import BasePrompt, VanillaPrompt, UncertaintyPrompt, GenerateTrainingDataPrompt
from unctune.datasets import PopQA, NQ, PopQATrain

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
    else:
        raise ValueError(f"Unknown prompt: {name}")

def load_model(name):
    name = name.lower()
    if name.startswith("local:"):
        return HuggingFaceModel(model_name=name.split(":")[1])
    if name == "olmo-7b":
        return Olmo(model_name="allenai/OLMo-7B-Instruct")
    elif name == "olmo2-13b":
        return Olmo2(model_name="allenai/OLMo-2-1124-13B-Instruct")
    elif name == "olmo2-13b-sft":
        return Olmo2(model_name="allenai/OLMo-2-1124-13B-SFT")
    if name == "ministral-8b":
        return Mistral(model_name="mistralai/Ministral-8B-Instruct-2410")
    elif name == "qwen-14b":
        return Qwen(model_name="Qwen/Qwen2.5-14B-Instruct")
    elif name == "qwen-32b":
        return Qwen(model_name="Qwen/Qwen2.5-32B-Instruct")
    elif name.startswith("gpt-"):
        return OpenAIModel(model_name=name)
    elif name.startswith("gemini"):
        return Gemini(model_name=name)
    else:
        return HuggingFaceModel(model_name=name)

def load_dataset(name):
    name = name.lower()
    if name == "popqa":
        return PopQA()
    elif name == "nq":
        return NQ()
    elif name == "popqa_training":
        return PopQATrain()    
    else:
        raise ValueError(f"Unknown dataset: {name}")