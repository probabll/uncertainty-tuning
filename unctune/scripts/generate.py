from unctune.models import Olmo, HuggingFaceModel, GenerationConfig, OpenAIModel, Mistral, Gemini
from unctune.datasets import PopQATrain
from unctune.prompts import BasePrompt, VanillaPrompt, UncertaintyPrompt
from unctune.nli import NliLLM
from unctune.confidence_verbal import VerbalUncLLM

from pathlib import Path
from tqdm import tqdm

import argparse
import torch
import json
import pandas as pd
import random
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="popqa_training")
    parser.add_argument("--model", type=str, default="olmo") 
    parser.add_argument("--prompt", type=str, default="vanilla")
    parser.add_argument("--nli_model", type=str, default="mistral") #check if we want to keep the gpt-4o as the NLI model since not free
    parser.add_argument("--unc_verbal_model", type=str, default="mistral") 
    parser.add_argument("--sample_batch_size", type=int, default=10)
    parser.add_argument("--num_sample_refs", type=int, default=3)
    parser.add_argument("--override_files", action="store_true")
    parser.add_argument("--skip_sample_assertions", action="store_true")
    args = parser.parse_args()

    random.seed(10)

    dataset = load_dataset(args.dataset)
    print(dataset.data)
    prompt = load_prompt(args.prompt)

    project_root = Path(__file__).resolve().parent.parent / ".."
    print(project_root)
    output_dir = project_root / "outputs" / "train_data"
    # output_dir.mkdir(exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    generations_file = output_dir / f"{args.model}-{args.dataset}-{args.prompt}.parquet"
    
    ### Generate the data necessary for training 
    # We need for each question, the greedy response, a reference response and some sampled responses (e.g. 3), 
    # as well as 20 samples to compute the confidence of each of these responses (which we want to include in the training data)
    # We can use a subset of the 20 samples as the sampled responses
    # Hence, regarding generations from the model we need: 1 greedy response and 20 samples
    if not generations_file.exists() or args.override_files:

        model = load_model(args.model)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        gen_cfg = GenerationConfig(max_length=250, remove_prompt=True, device=device) if not isinstance(model, OpenAIModel) else None

        generation_data = []
        print(f"Generating on the {args.dataset} dataset...")
        for idx, row in tqdm(dataset.data.iterrows(), total=len(dataset.data)):
            question = row["question"]
            answers = row["answers"]
            list_of_string = answers.replace("[","").replace("]","").split("\"") #list of answers as string - is there a better way to make it back to list to access elements?
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
                "references": answers, 
                "ref": next(s for s in list_of_string if s),
                "greedy": greedy_generation[0],
                "sample_refs": random.sample(samples, args.num_sample_refs),
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

    ### Compute confidence scores
    skip_assertions = "-skip-sample-assertions" if args.skip_sample_assertions else ""
    confidence_file = output_dir / f"{args.model}-{args.dataset}-{args.prompt}.confidence-{args.nli_model}{skip_assertions}.json"
    if not confidence_file.exists() or args.override_files:
        generations = pd.read_parquet(generations_file, engine="pyarrow")
        
        judge_model = load_model(args.nli_model) 
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        gen_cfg = GenerationConfig(max_length=250, remove_prompt=True, device=device) if not isinstance(judge_model, OpenAIModel) else None

        nli = NliLLM(judge_model, gen_cfg)

        print(f"Computing confidence scores using {args.nli_model}...")
        confidence_data = []
        for x, row in tqdm(generations.iterrows(), total=len(generations)):
            print(row)
            question = row["question"]
            sampled_assertions = row["samples"]
            greedy = row["greedy"]
            sample_refs = row["sample_refs"].tolist()
            gold_ref = row["ref"] #Without loss of generality, we choose the first available ref (they are paraphrases of each other)
            #CHECK HOW THIS LOOKS
            idx = row["idx"]

            row_data = {
                "idx": idx
            } 
            print(sample_refs)
            #All answer sentences for which we create training data points
            ref_answers = sample_refs.copy()
            ref_answers.append(gold_ref)
            ref_answers.append(greedy) 
            print(ref_answers)
            
            contradictions_answers = {}
            llm_outputs_answers = {}
            confidence_answers = {}
            for answer in ref_answers:
                contradictions, llm_outputs, confidence = get_confidence(nli, question, answer, sampled_assertions)    
                contradictions_answers[answer] = contradictions
                llm_outputs_answers[answer] = llm_outputs
                confidence_answers[answer] = confidence 

            row_data["question"] = question
            row_data["ref_answers"] = ref_answers
            row_data["confidence"] = confidence_answers
            row_data["contradictions"] = contradictions_answers
            row_data["llm_outputs"] = llm_outputs_answers
            confidence_data.append(row_data)

        # Save to a JSON file
        data = {}
        data["metadata"] = {
            "dataset": args.dataset,
            "model": args.model,
            "prompt": args.prompt,
            "judge_model": args.nli_model
        }
        data["data"] = confidence_data
        with open(confidence_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Confidence / NLI data saved to {confidence_file}.")

    ### Verbalise confidence and interleave in response
    training_data_file = output_dir / f"{args.model}-{args.dataset}-{args.prompt}.training_data-{args.unc_verbal_model}{skip_assertions}.json"
    if not training_data_file.exists() or args.override_files:
        with open(confidence_file, "r") as f:
            confidence_data = json.load(f)["data"]

        judge_model = load_model(args.unc_verbal_model) 
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        gen_cfg = GenerationConfig(max_length=250, remove_prompt=True, device=device) if not isinstance(judge_model, OpenAIModel) else None

        verbunc = VerbalUncLLM(judge_model, gen_cfg)
        training_data_points = []
        
        for row in confidence_data:
            print(row)
            confidence_answers = row["confidence"]
            ref_answers = row["ref_answers"]
            question = row["question"]
            idx = row["idx"]
            print(ref_answers)
            print(confidence_answers)
            for i, ans in enumerate(ref_answers):
                data_point = {}
                confidence = confidence_answers[ans]
                (verbal_unc, answer_unc) = verbunc.verbalised_conf_answer( question, ans, confidence) 
                data_point["question"] = question
                data_point["answer"] = ans
                data_point["confidence"] = confidence
                data_point["uncertainty_hedger"] = verbal_unc
                data_point["answer_unc_hedged"] = answer_unc
                training_data_points.append(data_point)

        # Save to a JSON file
        data = {}
        data["metadata"] = {
                "dataset": args.dataset,
                "model": args.model,
                "prompt": args.prompt,
                "judge_nli_model": args.nli_model,
                "judge_unc_verb_model": args.unc_verbal_model}
        data["data"] = training_data_points
        
        with open(training_data_file, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Training data points with verbalised uncertainty saved to {training_data_file}.")


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
    if name == "mistral":
        return Mistral()
    elif name.startswith("gpt-"):
        return OpenAIModel(model_name=name)
    elif name.startswith("gemini"):
        return Gemini(model_name=name)
    else:
        return HuggingFaceModel(model_name=name)

def load_dataset(name):
    name = name.lower()
    if name == "popqa_training":
        return PopQATrain()
    else:
        raise ValueError(f"Unknown training dataset: {name}")

def get_confidence(nli, question, ref_response, samples):
    contradictions = []
    llm_outputs = []
    
    for sample in samples:
        dnc, llm_output = nli.does_not_contradict(question, ref_response, sample, return_raw=True)

        contradicts = 0.0 if dnc else 1.0
        contradictions.append(contradicts)
        llm_outputs.append(llm_output)
    
    confidence = 1.0 - (1.0/len(samples)) * sum(contradictions)

    return contradictions, llm_outputs, confidence

if __name__ == "__main__":
    main()