from unctune.models import Olmo, HuggingFaceModel, GenerationConfig, OpenAIModel
from unctune.datasets import PopQA, NQ
from unctune.prompts import BasePrompt, VanillaPrompt, UncertaintyPrompt
from unctune.decisiveness import DecisivenessExtractorLLM

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
    parser.add_argument("--judge_model", type=str, default="gpt-4o-mini")
    parser.add_argument("--sample_batch_size", type=int, default=10)
    parser.add_argument("--override_files", action="store_true")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    prompt = load_prompt(args.prompt)

    project_root = Path(__file__).resolve().parent.parent / ".."
    output_dir = project_root / "outputs" / "eval_data"
    output_dir.mkdir(exist_ok=True)
    generations_file = output_dir / f"{args.model}-{args.dataset}-{args.prompt}.parquet"
    
    ### Generate the data necessary for evaluation (20 samples and a greedy decoding)
    if not generations_file.exists() or args.override_files:

        model = load_model(args.model)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        gen_cfg = GenerationConfig(max_length=250, remove_prompt=True, device=device) if not isinstance(model, OpenAIModel) else None

        generation_data = []
        for idx, question, answers in tqdm(dataset):
            model_input = prompt(question)
            greedy_generation = model.greedy_decode([model_input], cfg=gen_cfg)

            samples = []
            while len(samples) < 20:
                n = min(args.sample_batch_size, 20-len(samples))
                sampled_generations = model.sample([model_input], n, cfg=gen_cfg)
                samples.extend(sampled_generations)

            generation_data.append({
                "idx": idx,
                "question": question,
                "greedy": greedy_generation[0],
                "samples": samples
            })

        # Convert the data into a dataframe 
        dataframe = pd.DataFrame(generation_data)
        dataframe.attrs["dataset"] = args.dataset
        dataframe.attrs["model"] = args.model
        dataframe.attrs["prompt"] = args.prompt

        # Store the data
        dataframe.to_parquet(generations_file, engine="pyarrow", index=False)
        print(f"Data saved to {generations_file}.")
    
    ### Extract the decisiveness scores
    decisiveness_file = output_dir / f"{args.model}-{args.dataset}-{args.prompt}-{args.judge_model}.decisiveness.json"
    if not decisiveness_file.exists() or args.override_files:
        dataframe = pd.read_parquet(generations_file, engine="pyarrow")

        judge_model = load_model(args.judge_model)
        extractor = DecisivenessExtractorLLM(judge_model)

        decisiveness_data = []
        for x, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            question = row["question"]
            row_data = {
                "idx": row["idx"]
            }

            # Do the greedy response first
            greedy_response = row["greedy"]
            decisiveness, raw_output = extractor(question, greedy_response, return_raw=True)
            row_data["greedy"] = {
                "llm_judge_output": raw_output,
                "decisiveness": decisiveness
            }

            # As many answers are the same in short-form QA, we can save some time and model calls by caching outputs.
            cache = {
                greedy_response: (decisiveness, raw_output)
            }

            # Then do the sampled responses, we technically only need the assertions here.
            sampled_responses = row["samples"]
            sampled_data = []
            for i, sampled_response in enumerate(sampled_responses):
                
                if sampled_response in cache:
                    decisiveness, raw_output = cache[sampled_response]
                else:
                    decisiveness, raw_output = extractor(question, sampled_response, return_raw=True)
                    cache[sampled_response] = (decisiveness, raw_output)

                sampled_data.append({
                    "sample_idx": i,
                    "llm_judge_output": raw_output,
                    "decisiveness": decisiveness # list of (assertion, deciviseness) tuples
                })
            row_data["samples"] = sampled_data
            
            decisiveness_data.append(row_data)

        # Save to a JSON file
        data = {}
        data["metadata"] = {
            "dataset": args.dataset,
            "model": args.model,
            "prompt": args.prompt,
            "judge_model": args.judge_model
        }
        data["data"] = decisiveness_data
        with open(decisiveness_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Decisiveness / assertions data saved to {decisiveness_file}.")

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
    elif name.startswith("gpt-"):
        return OpenAIModel(model_name=name)
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