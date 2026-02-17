from unctune.models import GenerationConfig, OpenAIModel
from unctune.decisiveness import DecisivenessExtractorLLM
from unctune.nli import NliLLM
from unctune.metrics import compute_lenient_em

from unctune.scripts.utils import load_model, load_prompt, load_dataset

from pathlib import Path
from tqdm import tqdm

import argparse
import torch
import json
import pandas as pd
import evaluate

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metric", type=str, default="faithfulness", 
                        choices=["faithfulness", "accuracy"])
    parser.add_argument("--dataset", "-d", type=str, default="popqa")
    parser.add_argument("--model", "-m", type=str, default="olmo") 
    parser.add_argument("--prompt", "-p", type=str, default="vanilla")
    parser.add_argument("--system_prompt", type=str, default=None)
    parser.add_argument("--decisiveness_model", "-dec", type=str)
    parser.add_argument("--nli_model", "-nli", type=str, default=None)
    parser.add_argument("--sample_batch_size", "-bs", type=int, default=10)
    parser.add_argument("--override_files", action="store_true")
    parser.add_argument("--decoding_method", type=str, default="greedy", choices=["greedy", "sampling", "nucleus"])
    parser.add_argument("--use_altmap", action="store_true")
    args = parser.parse_args()

    if args.metric == "faithfulness":
        evaluate_faithfulness(args)
    elif args.metric == "accuracy":
        evaluate_accuracy(args)
    else:
        print(f"Unknown metric: {args.metric}. Please choose either 'faithfulness' or 'accuracy'.")
    
def evaluate_accuracy(args):
    dataset = load_dataset(args.dataset)
    prompt = load_prompt(args.prompt)
    system_prompt = load_prompt(args.system_prompt) if args.system_prompt else None

    project_root = Path(__file__).resolve().parent.parent / ".."
    output_dir = project_root / "outputs" / "eval_data"
    output_dir.mkdir(exist_ok=True)
    model_name = args.model 
    if "local:" in model_name:
        path = Path(model_name.split(":")[1])
        model_name = path.name

    decoding_str = "" if args.decoding_method == "greedy" else f"-{args.decoding_method}"
    if args.system_prompt: decoding_str = decoding_str + f"-sys={args.system_prompt}"
    generations_file = output_dir / f"{model_name}-{args.dataset}-{args.prompt}{decoding_str}.json"
    
    ### Generate the data necessary for evaluation (20 samples and a greedy decoding)
    if not generations_file.exists() or args.override_files:
        print("No generations found.")

        model = load_model(args.model)
        model.set_system_prompt(system_prompt)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        gen_cfg = GenerationConfig(max_length=250, remove_prompt=True, device=device) if not isinstance(model, OpenAIModel) else None

        generation_data = []
        print(f"Generating on the {args.dataset} dataset...")
        for idx, question, answers in tqdm(dataset):
            model_input = prompt(question)

            if args.decoding_method == "greedy":
                greedy_generation = model.greedy_decode([model_input], cfg=gen_cfg)
            elif args.decoding_method == "sampling":
                greedy_generation = model.sample([model_input], 1, cfg=gen_cfg)
            elif args.decoding_method == "nucleus":
                model.top_p = 0.7
                greedy_generation = model.sample([model_input], 1, cfg=gen_cfg)
                model.top_p = 1.0

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
        dataframe.to_json(generations_file, orient="records", indent=4)
        print(f"Data saved to {generations_file}.")

        # Unload the generation model to avoid OOM errors
        del model
        torch.cuda.empty_cache()
    else:
        print(f"Found generations data at {generations_file}.")

    dataframe = pd.read_json(generations_file, dtype={"idx": str})
    reference_lists = [dataset[i][-1] for i in range(len(dataset))]
    preds = [dataframe.iloc[i]["greedy"] for i in range(len(dataframe))]
    em = compute_lenient_em(predictions=preds, reference_lists=reference_lists)
    print(f"Lenient Exact Match score on {args.dataset}: {em:.2f}%")

def evaluate_faithfulness(args):
    dataset = load_dataset(args.dataset)
    prompt = load_prompt(args.prompt)
    system_prompt = load_prompt(args.system_prompt) if args.system_prompt else None

    project_root = Path(__file__).resolve().parent.parent / ".."
    output_dir = project_root / "outputs" / "eval_data"
    output_dir.mkdir(exist_ok=True)
    model_name = args.model 
    if "local:" in model_name:
        path = Path(model_name.split(":")[1])
        model_name = path.name

    decoding_str = "" if args.decoding_method == "greedy" else f"-{args.decoding_method}"
    if args.system_prompt: decoding_str = decoding_str + f"-sys={args.system_prompt}"
    generations_file = output_dir / f"{model_name}-{args.dataset}-{args.prompt}{decoding_str}.json"
    
    ### Generate the data necessary for evaluation (20 samples and a greedy decoding)
    if not generations_file.exists() or args.override_files:
        print("No generations found.")

        model = load_model(args.model)
        model.set_system_prompt(system_prompt)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        gen_cfg = GenerationConfig(max_length=250, remove_prompt=True, device=device) if not isinstance(model, OpenAIModel) else None

        generation_data = []
        print(f"Generating on the {args.dataset} dataset...")
        for idx, question, answers in tqdm(dataset):
            model_input = prompt(question)

            if args.decoding_method == "greedy":
                greedy_generation = model.greedy_decode([model_input], cfg=gen_cfg)
            elif args.decoding_method == "sampling":
                greedy_generation = model.sample([model_input], 1, cfg=gen_cfg)
            elif args.decoding_method == "nucleus":
                model.top_p = 0.7
                greedy_generation = model.sample([model_input], 1, cfg=gen_cfg)
                model.top_p = 1.0

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
        dataframe.to_json(generations_file, orient="records", indent=4)
        print(f"Data saved to {generations_file}.")

        # Unload the generation model to avoid OOM errors
        del model
        torch.cuda.empty_cache()
    else:
        print(f"Found generations data at {generations_file}.")
    
    ### Extract the decisiveness scores
    
    decisiveness_file = output_dir / f"{model_name}-{args.dataset}-{args.prompt}{decoding_str}.decisiveness-{args.decisiveness_model}.json"
    if not decisiveness_file.exists() or args.override_files:
        print("No decisiveness data found.")
        dataframe = pd.read_json(generations_file, dtype={"idx": str})

        judge_model = load_model(args.decisiveness_model)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        gen_cfg = GenerationConfig(max_length=250, remove_prompt=True, device=device) if not isinstance(judge_model, OpenAIModel) else None
        
        extractor = DecisivenessExtractorLLM(judge_model, gen_cfg, use_altmap=args.use_altmap)

        print(f"Extracting assertions and decisiveness scores using {args.decisiveness_model}...")
        decisiveness_data = []
        for x, row in tqdm(dataframe.iterrows(), total=len(dataframe)):
            question = row["question"]
            row_data = {
                "idx": row["idx"]
            }

            # Extract assertions and decisiveness for the greedy response
            greedy_response = row["greedy"]
            decisiveness, raw_output = extractor(question, greedy_response, return_raw=True)
            row_data["greedy"] = {
                "llm_judge_output": raw_output,
                "decisiveness": decisiveness
            }

            decisiveness_data.append(row_data)

        # Save to a JSON file
        data = {}
        data["metadata"] = {
            "dataset": args.dataset,
            "model": args.model,
            "prompt": args.prompt,
            "judge_model": args.decisiveness_model
        }
        data["data"] = decisiveness_data
        with open(decisiveness_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Decisiveness / assertions data saved to {decisiveness_file}.")

        # Unload the judge model to avoid OOM errors
        del judge_model
        del extractor
        torch.cuda.empty_cache()
    else:
        print(f"Found decisiveness data at {decisiveness_file}.")

    ### Compute confidence scores
    nli_model_name = args.nli_model if args.nli_model is not None else args.decisiveness_model
    if nli_model_name == args.decisiveness_model:
        confidence_file = output_dir / f"{model_name}-{args.dataset}-{args.prompt}{decoding_str}.confidence-{nli_model_name}.json"
    else:
        confidence_file = output_dir / f"{model_name}-{args.dataset}-{args.prompt}{decoding_str}-decisiveness-{args.decisiveness_model}-confidence-{nli_model_name}.json"

    if not confidence_file.exists() or args.override_files:
        print("No confidence data found.")
        generations = pd.read_json(generations_file, dtype={"idx": str})
        assertions_by_idx = read_assertions(decisiveness_file)
        
        judge_model = load_model(nli_model_name)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        gen_cfg = GenerationConfig(max_length=250, remove_prompt=True, device=device) if not isinstance(judge_model, OpenAIModel) else None

        nli = NliLLM(judge_model, gen_cfg)

        print(f"Computing confidence scores using {nli_model_name}...")
        confidence_data = []
        for x, row in tqdm(generations.iterrows(), total=len(generations)):
            question = row["question"]
            idx = row["idx"]

            row_data = {
                "idx": idx
            }

            greedy_assertions = assertions_by_idx[idx]["assertions"]
            decisiveness = assertions_by_idx[idx]["decisiveness"]
            
            samples = row["samples"]

            llm_outputs = []
            contradictions = []
            cache = {}

            confidence_scores = []

            # For each greedy assertion, we compute a confidence score
            for greedy_assertion in greedy_assertions:

                contradictions_i = []
                llm_outputs_i = []
                for sample_generation in samples:

                    # If the greedy assertion is the same as the, we can skip the NLI call.
                    # Also, we keep a cache of the NLI outputs to avoid redundant calls.
                    if greedy_assertion == sample_generation:
                        dnc = True
                        llm_output = "<equivalent>"
                    elif sample_generation in cache:
                        dnc = cache[sample_generation]
                        llm_output = "<cache>"
                    else:
                        dnc, llm_output = nli.does_not_contradict(question, greedy_assertion, sample_generation, return_raw=True)
                        cache[sample_generation] = dnc

                    contradicts = 0.0 if dnc else 1.0
                    contradictions_i.append(contradicts)
                    llm_outputs_i.append(llm_output)

                print(llm_outputs_i)            
                confidence_i = 1.0 - ((1.0/len(samples)) * sum(contradictions_i))
                confidence_scores.append(confidence_i)
                llm_outputs.append(llm_outputs_i)
                contradictions.append(contradictions_i)
            
            row_data["confidence"] = confidence_scores
            row_data["contradictions"] = contradictions
            row_data["llm_outputs"] = llm_outputs
            confidence_data.append(row_data)

        # Save to a JSON file
        data = {}
        data["metadata"] = {
            "dataset": args.dataset,
            "model": args.model,
            "prompt": args.prompt,
            "judge_model": nli_model_name
        }
        data["data"] = confidence_data
        with open(confidence_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Confidence / NLI data saved to {confidence_file}.")
    else:
        print(f"Found confidence data at {confidence_file}.")

    ### Compute faithfulness scores
    with open(confidence_file, "r") as f:
        confidence_data = json.load(f)["data"]
    assertions = read_assertions(decisiveness_file)

    confidences = []
    decisivenesses = []
    faithfulnesses = []
    bin_ids = []

    confidence_bins = [i/10 for i in range(1, 11)]
    
    for row in confidence_data:
        confidence = row["confidence"]
        idx = row["idx"]
        decisiveness = assertions[idx]["decisiveness"]
        faithfulness = 1.0 - ((1.0 / len(confidence)) * sum([abs(c - d) for c, d in zip(confidence, decisiveness)]))
        
        # Assign a bin based on the average confidence score
        average_confidence = sum(confidence) / len(confidence)
        bin_idx = 0
        for i, bin_val in enumerate(confidence_bins):
            if average_confidence <= bin_val:
                bin_idx = i
                break

        bin_ids.append(bin_idx)
        confidences.append(average_confidence)
        decisivenesses.append(sum(decisiveness) / len(decisiveness))
        faithfulnesses.append(faithfulness)

    # Count the number of samples in each bin
    bin_counts = [bin_ids.count(i) for i in range(len(confidence_bins))]

    bin_idx_to_name = {i: f"({confidence_bins[i]-(1. / len(confidence_bins)):.1f}, {confidence_bins[i]:.1f}]" for i in range(1, len(confidence_bins))}
    bin_idx_to_name[0] = f"[0.0, {confidence_bins[0]:.1f}]"
    print("Bin counts:")
    for i, count in enumerate(bin_counts):
        print(f"{bin_idx_to_name[i]}: {count}")
    print()

    # MFG per bin
    MFG_per_bin = [0.0 for _ in range(len(confidence_bins))]
    for i in range(len(confidence_bins)):
        bin_faithfulnesses = [faithfulnesses[j] for j in range(len(faithfulnesses)) if bin_ids[j] == i]
        MFG_per_bin[i] = sum(bin_faithfulnesses)/len(bin_faithfulnesses)

    print("Mean faithfulness per bin:")
    for i, mean_faithfulness in enumerate(MFG_per_bin):
        print(f"{bin_idx_to_name[i]}: {mean_faithfulness:.2f}")
    print()

    print("Mean confidence per bin:")
    for i in range(len(confidence_bins)):
        bin_confidences = [confidences[j] for j in range(len(confidences)) if bin_ids[j] == i]
        mean_confidence = sum(bin_confidences)/len(bin_confidences)
        print(f"{bin_idx_to_name[i]}: {mean_confidence:.2f}")
    print()

    print("Mean decisiveness per bin:")
    for i in range(len(confidence_bins)):
        bin_decisivenesses = [decisivenesses[j] for j in range(len(decisivenesses)) if bin_ids[j] == i]
        mean_decisiveness = sum(bin_decisivenesses)/len(bin_decisivenesses)
        print(f"{bin_idx_to_name[i]}: {mean_decisiveness:.2f}")
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
        greedy = row["greedy"]["decisiveness"]
        assertions = [greedy[i][0] for i in range(len(greedy))]
        decisiveness = [greedy[i][1] for i in range(len(greedy))]
        assertions_by_idx[idx] = {"assertions": assertions, "decisiveness": decisiveness}
    return assertions_by_idx

