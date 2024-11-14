from unctune.models import Olmo, HuggingFaceModel, GenerationConfig, OpenAIModel
from unctune.datasets import PopQA, NQ
from unctune.prompts import BasePrompt, VanillaPrompt, UncertaintyPrompt
from unctune.decisiveness import DecisivenessExtractorLLM
from unctune.nli import NliLLM

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
        print(f"Generating on the {args.dataset} dataset...")
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
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        gen_cfg = GenerationConfig(max_length=250, remove_prompt=True, device=device) if not isinstance(judge_model, OpenAIModel) else None
        
        extractor = DecisivenessExtractorLLM(judge_model, gen_cfg)

        print("Extracting assertions and decisiveness scores...")
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
                    raw_output = "<cache>"
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

    ### Compute confidence scores
    confidence_file = output_dir / f"{args.model}-{args.dataset}-{args.prompt}-{args.judge_model}.confidence.json"
    if not confidence_file.exists() or args.override_files:
        generations = pd.read_parquet(generations_file, engine="pyarrow")
        assertions_by_idx = read_assertions(decisiveness_file)
        
        judge_model = load_model(args.judge_model)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        gen_cfg = GenerationConfig(max_length=250, remove_prompt=True, device=device) if not isinstance(judge_model, OpenAIModel) else None

        nli = NliLLM(judge_model, gen_cfg)

        print("Computing confidence scores...")
        confidence_data = []
        for x, row in tqdm(generations.iterrows(), total=len(generations)):
            question = row["question"]
            idx = row["idx"]

            row_data = {
                "idx": idx
            }

            # Then do the sampled responses
            greedy_assertion = assertions_by_idx[idx]["assertion"]
            sampled_assertions = assertions_by_idx[idx]["sampled_assertions"]
            contradictions = []
            llm_outputs = []
            cache = {}
            for sampled_assertion in sampled_assertions:

                # If the greedy assertion is the same as the sampled assertion, we can skip the NLI call.
                # Also, we keep a cache of the NLI outputs to avoid redundant calls.
                if greedy_assertion == sampled_assertion:
                    dnc = True
                    llm_output = "<equivalent>"
                elif sampled_assertion in cache:
                    dnc, llm_output = cache[sampled_assertion]
                    llm_output = "<cache>"
                else:
                    dnc, llm_output = nli.does_not_contradict(question, greedy_assertion, sampled_assertion, return_raw=True)
                    cache[sampled_assertion] = (dnc, llm_output)

                contradicts = 0.0 if dnc else 1.0
                contradictions.append(contradicts)
                llm_outputs.append(llm_output)
            
            confidence = 1.0 - (1.0/len(sampled_assertions)) * sum(contradictions)

            row_data["confidence"] = confidence
            row_data["contradicitions"] = contradictions
            row_data["llm_outputs"] = llm_outputs
            confidence_data.append(row_data)

        # Save to a JSON file
        data = {}
        data["metadata"] = {
            "dataset": args.dataset,
            "model": args.model,
            "prompt": args.prompt,
            "judge_model": args.judge_model
        }
        data["data"] = confidence_data
        with open(confidence_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Confidence / NLI data saved to {confidence_file}.")

    ### Compute faithfulness scores
    with open(confidence_file, "r") as f:
        confidence_data = json.load(f)["data"]
    assertions = read_assertions(decisiveness_file)

    confidences = []
    decisivenesses = []
    faithfulnesses = []
    bin_ids = []

    confidence_bins = [i/10 for i in range(11)]
    
    for row in confidence_data:
        confidence = row["confidence"]
        idx = row["idx"]
        decisiveness = assertions[idx]["decisiveness"]
        faithfulness = 1.0 - abs(confidence - decisiveness)

        # Assign a bin based on the confidence score
        bin_idx = 0
        for i, bin_val in enumerate(confidence_bins):
            if confidence <= bin_val:
                bin_idx = i
                break

        bin_ids.append(bin_idx)
        confidences.append(confidence)
        decisivenesses.append(decisiveness)
        faithfulnesses.append(faithfulness)

    # Count the number of samples in each bin
    bin_counts = [confidence_bins.count(i) for i in range(11)]
    print("Bin counts:")
    for i, count in enumerate(bin_counts):
        print(f"{i}: {count}")
    print()

    # MFG per bin
    MFG_per_bin = [0.0 for _ in range(11)]
    for i in range(11):
        bin_faithfulnesses = [faithfulnesses[j] for j in range(len(faithfulnesses)) if confidence_bins[j] == i]
        MFG_per_bin[i] = sum(bin_faithfulnesses)/len(bin_faithfulnesses)
    print("Mean faithfulness per bin:")
    for i, mean_faithfulness in enumerate(MFG_per_bin):
        print(f"{i}: {mean_faithfulness:.2f}")
    print()

    # Assumes uniform distribution of confidence scores
    cMFG = sum(MFG_per_bin)/len(MFG_per_bin)
    
    print(f"Mean confidence: {sum(confidences)/len(confidences):.2f}")
    print(f"Mean decisiveness: {sum(decisivenesses)/len(decisivenesses):.2f}")
    print(f"Mean faithfulness: {sum(faithfulnesses)/len(faithfulnesses):.2f}")
    print(f"cMFG: {cMFG:.2f}")

def read_assertions(decisiveness_file):
    with open(decisiveness_file, "r") as f:
        data = json.load(f)["data"]
    assertions_by_idx = {}
    for row in data:
        idx = row["idx"]
        greedy_assertions = row["greedy"]["decisiveness"]
        if len(greedy_assertions) > 1:
            print("Warning: assuming only one assertion for greedy decoding, but found multiple. Will drop all but the first.")
            print(greedy_assertions)
        greedy_assertion = greedy_assertions[0][0]
        greedy_decisiveness = greedy_assertions[0][1]
        sampled_assertions = []
        for sampled_row in row["samples"]:
            for assertion, _ in sampled_row["decisiveness"]:
                sampled_assertions.append(assertion)
        assertions_by_idx[idx] = {"assertion": greedy_assertion, "decisiveness": greedy_decisiveness, "sampled_assertions": sampled_assertions}
    return assertions_by_idx

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