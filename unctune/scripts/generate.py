from unctune.models import GenerationConfig, OpenAIModel
from unctune.nli import NliLLM
from unctune.nli_model import NLIModel
from unctune.confidence_verbal import VerbalUncLLM
from unctune.scripts.utils import load_model, load_prompt, load_dataset, load_train_dataset

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
    parser.add_argument("--model", type=str, default="llama-8b-sft") 
    parser.add_argument("--prompt", type=str, default="generate_train")
    parser.add_argument("--nli_model", type=str, default="deberta-large") 
    parser.add_argument("--unc_verbal_model", type=str, default="mistral-12b") 
    parser.add_argument("--sample_batch_size", type=int, default=10)
    parser.add_argument("--num_sample_refs", type=int, default=10)
    parser.add_argument("--override_files", action="store_true")
    parser.add_argument("--override_conf", action="store_true") #store_false (if NLI component should be skipped)
    parser.add_argument("--override_verbal", action="store_true") #store_false (if verbalising component should be skipped)
    parser.add_argument("--min_gen_index", type=int, default=4500) 
    parser.add_argument("--max_gen_index", type=int, default=5654) 
    parser.add_argument("--cache_prompts", type=bool, default=True)
    args = parser.parse_args()

    random.seed(10)

    dataset = load_train_dataset(args.dataset, args.min_gen_index, args.max_gen_index)
    prompt = load_prompt(args.prompt)

    project_root = Path(__file__).resolve().parent.parent / ".."
    output_dir = project_root / "outputs" / "train_data"
    # output_dir.mkdir(exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    generations_file = output_dir / f"{args.model}-{args.dataset}-{args.prompt}-{args.min_gen_index}-{args.max_gen_index}.parquet"
    
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
    confidence_file = output_dir / f"{args.model}-{args.dataset}-{args.prompt}.confidence-{args.nli_model}-{args.min_gen_index}-{args.max_gen_index}.json"
    if args.override_conf:
        print('Not running confidence')
    else:
        if not confidence_file.exists() or args.override_files:
            generations = pd.read_parquet(generations_file, engine="pyarrow")
            
            if args.nli_model == "deberta-large":
                nli = NLIModel("microsoft/deberta-large-mnli")
            else:
                judge_model = load_model(args.nli_model) 
                device = "cuda:0" if torch.cuda.is_available() else "cpu"
                gen_cfg = GenerationConfig(max_length=250, remove_prompt=True, device=device) if not isinstance(judge_model, OpenAIModel) else None
                nli = NliLLM(judge_model, gen_cfg)


            print(f"Computing confidence scores using {args.nli_model}...")
            confidence_data = []
            for x, row in tqdm(generations.iterrows(), total=len(generations)):
                question = row["question"]
                sampled_assertions = row["samples"]

                idx = row["idx"]

                row_data = {
                    "idx": idx
                } 
                #All answer sentences for which we create training data points
                samples = sampled_assertions.copy()
                ref_answers = random.sample(samples.tolist(), args.num_sample_refs)
                contradictions_answers = {}
                # llm_outputs_answers = {}
                confidence_answers = {}
                for answer in ref_answers:
                    contradictions, confidence = get_confidence(nli, question, answer, sampled_assertions, cache_prompt_bool=args.cache_prompts)    
                    contradictions_answers[answer] = contradictions
                    # llm_outputs_answers[answer] = llm_outputs
                    confidence_answers[answer] = confidence 

                row_data["question"] = question
                row_data["ref_answers"] = ref_answers
                row_data["confidence"] = confidence_answers
                row_data["contradictions"] = contradictions_answers
                # row_data["llm_outputs"] = llm_outputs_answers
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
    training_data_file = output_dir / f"{args.model}-{args.dataset}-{args.prompt}.training_data-{args.unc_verbal_model}-{args.min_gen_index}-{args.max_gen_index}.json"
    if args.override_verbal:
        print('Not running interleaving')
    else:
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
                for i, ans in enumerate(ref_answers):
                    data_point = {}
                    confidence = confidence_answers[ans]
                    if len(ans)>0:
                        (verbal_unc, answer_unc_interweave, answer_unc_postfix) = verbunc.verbalised_conf_answer(question, ans, confidence, cache_prompt_bool = args.cache_prompts) 
                        data_point["question"] = question
                        data_point["answer"] = ans
                        data_point["confidence"] = confidence
                        data_point["uncertainty_hedger"] = verbal_unc
                        data_point["answer_unc_hedged_interweave"] = answer_unc_interweave
                        data_point["answer_unc_hedged_postfix"] = answer_unc_postfix
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


def get_confidence(nli, question, ref_response, samples, cache_prompt_bool = True):
    contradictions = []
    
    for sample in samples:
        contradicts = nli.bidirectional_contra(question, ref_response, sample)
        contradictions.append(contradicts)
    
    confidence = 1.0 - (sum(contradictions)/len(samples))

    return contradictions, confidence, 

if __name__ == "__main__":
    main()
