from unctune.models import Olmo, HuggingFaceModel, GenerationConfig
from unctune.datasets import PopQA, NQ
from unctune.prompts import BasePrompt, VanillaPrompt, UncertaintyPrompt

from pathlib import Path
from tqdm import tqdm

import argparse
import torch
import json
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="popqa")
    parser.add_argument("--model", type=str, default="olmo")
    parser.add_argument("--prompt", type=str, default="vanilla")
    args = parser.parse_args()

    model = load_model(args.model)
    dataset = load_dataset(args.dataset)
    prompt = load_prompt(args.prompt)
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    gen_cfg = GenerationConfig(max_length=250, remove_prompt=True, device=device)

    # Generate the data necessary for evaluation (20 samples and a greedy decoding)
    eval_data = []
    for idx, QA in tqdm(enumerate(dataset)):
        question, _ = QA
        model_input = prompt(question)
        greedy_generation = model.greedy_decode([model_input], cfg=gen_cfg)
        samples = model.sample([model_input], 20, cfg=gen_cfg)
        eval_data.append({
            "idx": idx,
            "question": question,
            "greedy": greedy_generation[0],
            "samples": samples
        })

    # Convert the data into a dataframe 
    dataframe = pd.DataFrame(eval_data)
    dataframe.attrs["dataset"] = args.dataset
    dataframe.attrs["model"] = args.model
    dataframe.attrs["prompt"] = args.prompt

    # Store the data
    project_root = Path(__file__).resolve().parent.parent / ".."
    output_dir = project_root / "outputs" / "eval_data"
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / f"{args.model}-{args.dataset}-{args.prompt}.parquet"
    dataframe.to_parquet(output_file, engine="pyarrow", index=False)
    print(f"Data saved to {output_file}.")

def load_prompt(name):
    name = name.lower()
    if name == "base":
        return BasePrompt()
    elif name == "vanilla":
        return VanillaPrompt()
    elif name == "uncertainty":
        return UncertaintyPrompt()
    else:
        raise ValueError(f"Unknown prompt: {name}")

def load_model(name):
    name = name.lower()
    if name == "olmo":
        return Olmo()
    else:
        return HuggingFaceModel(model_name=name)

def load_dataset(name):
    name = name.lower()
    if name == "popqa":
        return PopQA()
    elif name == "nq":
        return NQ()
    else:
        raise ValueError(f"Unknown dataset: {name}")