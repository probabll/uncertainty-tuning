from unctune.models import GenerationConfig, OpenAIModel
from unctune.semantic_equivalence import NliEntailLLM

from unctune.scripts.utils import load_model, load_prompt, load_dataset

from pathlib import Path
from tqdm import tqdm

from collections import Counter
import random
import argparse
import torch
import json
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", type=str, default="popqa")
    parser.add_argument("--initial_model", "-im", type=str, default="olmo2-13b-sft") 
    parser.add_argument("--final_model", "-fm", type=str, default="olmo2-13b-sft-FT-pf") 
    parser.add_argument("--initial_prompt", "-ip", type=str, default="generate_train")
    parser.add_argument("--final_prompt", "-fp", type=str, default="generate_train")
    parser.add_argument("--se_model", "-se", type=str, default="olmo2-13b")
    parser.add_argument("--sample_batch_size", "-bs", type=int, default=10)
    parser.add_argument("--lower_ind", "-li", type=int, default=0)
    parser.add_argument("--upper_ind", "-ui", type=int, default=100)
    parser.add_argument("--override_files", action="store_true")
    parser.add_argument("--cache_prompts", type=bool, default=True)
    parser.add_argument("--bootstrap", type=str, default="different")
    args = parser.parse_args()

    project_root = Path(__file__).resolve().parent.parent / ".."
    output_dir = project_root / "outputs" / "eval_data"
    output_dir.mkdir(exist_ok=True)

    generations_file_initial_dist = output_dir / f"{args.initial_model}-{args.dataset}-{args.initial_prompt}_initial_generations.json"
    ### Generate the data necessary for evaluation (20 samples and a greedy decoding) for initial distribution
    if not generations_file_initial_dist.exists() or args.override_files:
        print("No generations found.")

        model = load_model(args.initial_model)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        gen_cfg = GenerationConfig(max_length=250, remove_prompt=True, device=device) if not isinstance(model, OpenAIModel) else None

        dataset = load_dataset(args.dataset)
        prompt = load_prompt(args.initial_prompt)

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
        dataframe.attrs["model"] = args.initial_model
        dataframe.attrs["prompt"] = args.initial_prompt

        # Store the data
        dataframe.to_json(generations_file_initial_dist, orient="records", indent=4)      
        print(f"Data saved to {generations_file_initial_dist}.")
    else:
        print(f"Found generations data at {generations_file_initial_dist}.")

    generations_file_final_dist = output_dir / f"{args.final_model}-{args.dataset}-{args.final_prompt}_final_generations.json"

    ### Generate the data necessary for evaluation (20 samples and a greedy decoding) for final distribution
    if not generations_file_final_dist.exists() or args.override_files:
        print("No generations found.")

        model = load_model(args.final_model)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        gen_cfg = GenerationConfig(max_length=250, remove_prompt=True, device=device) if not isinstance(model, OpenAIModel) else None

        dataset = load_dataset(args.dataset)
        prompt = load_prompt(args.final_prompt)

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
        dataframe.attrs["model"] = args.final_model
        dataframe.attrs["prompt"] = args.final_prompt

        # Store the data
        dataframe.to_json(generations_file_final_dist, orient="records", indent=4)      
        print(f"Data saved to {generations_file_final_dist}.")
    else:
        print(f"Found generations data at {generations_file_final_dist}.")

    se_model_name = args.se_model 

    clusters_file = output_dir / f"{args.dataset}-{args.initial_model}-{args.initial_prompt}-compare-{args.final_model}-{args.final_prompt}.clustering-{args.se_model}_{args.lower_ind}-{args.upper_ind}.json"
    if not clusters_file.exists() or args.override_files:
        print("No clustering data found.")
        
        generations_initial = pd.read_json(generations_file_initial_dist)
        generations_final = pd.read_json(generations_file_final_dist)

        generations_merged = generations_initial.merge(generations_final,  how='inner', on='idx')
        generations_merged = generations_merged.drop(columns=['question_y'])
        generations_merged = generations_merged.rename(columns={"question_x":"question", "greedy_x": "greedy_initial", "samples_x": "samples_initial", "greedy_y": "greedy_final", "samples_y": "samples_final"})

        generations_merged = generations_merged[args.lower_ind:args.upper_ind]

        judge_model = load_model(se_model_name)
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        gen_cfg = GenerationConfig(max_length=250, remove_prompt=True, device=device) if not isinstance(judge_model, OpenAIModel) else None

        nli = NliEntailLLM(judge_model, gen_cfg)
        
        print(f"Computing clusters using {se_model_name}...")
        clustering_data = []
        for x, row in tqdm(generations_merged.iterrows(), total=len(generations_merged)):
            question = row["question"]
            idx = row["idx"]
            samples_initial = row["samples_initial"]
            samples_final = row["samples_final"]

            row_data = {
                "idx": idx,
                "samples_initial": samples_initial,
                "samples_final": samples_final,
            }

            joint_set_samples = list(set(samples_initial + samples_final)) #unique responses
            clusters = {}
            sample_to_cluster = {}

            clusters[0] = [joint_set_samples[0]]
            sample_to_cluster[joint_set_samples[0]] = 0
            if len(joint_set_samples)>1:
                for sample in joint_set_samples[1:]:
                    cluster_assignment = None
                    for clus_num in clusters.keys():
                        compare_sample = random.sample(clusters[clus_num],1)
                        is_equivalent = nli.does_entail(question, sample, compare_sample[0], return_raw=True, cache_prompt_bool=args.cache_prompts)

                        if is_equivalent[0] == True:
                            cluster_assignment = clus_num #this is the cluster we assign to the sample
                            break
                    
                    if cluster_assignment == None: 
                        #if we iterated over all existing clusters, 
                        #but none of them is semantically equivalent to the sample
                        new_cluster = max(list(clusters.keys())) + 1
                        clusters[new_cluster] = [sample]
                        sample_to_cluster[sample] = new_cluster
                    else: #if we found a semantically equivalent cluster
                        clusters[cluster_assignment].append(sample)
                        sample_to_cluster[sample] = cluster_assignment

            row_data["clusters_to_sample_assignments"] = clusters
            row_data["sample_to_cluster_assignments"] = sample_to_cluster

            clustering_data.append(row_data)

            # Save to a JSON file
            data = {}
            data["metadata"] = {
                "dataset": args.dataset,
                "initial_model": args.initial_model,
                "final_model": args.final_model,
                "initial_prompt": args.initial_prompt,
                "final_prompt": args.final_prompt,
                "judge_model": se_model_name
            }
            data["data"] = clustering_data
            with open(clusters_file, "w") as f:
                json.dump(data, f, indent=2)
            print(f"Semantic equivalence / NLI data saved to {clusters_file}.")
    else:
        print(f"Found clustering data at {clusters_file}.")

    distribution_analysis_file = output_dir / f"{args.dataset}-{args.initial_model}-{args.initial_prompt}-compare-{args.final_model}-{args.final_prompt}.dist_analyses_{args.lower_ind}-{args.upper_ind}.json"
    if not distribution_analysis_file.exists() or args.override_files:
        ### Compute faithfulness scores
        with open(clusters_file, "r") as f:
            clustering_data = json.load(f)["data"]

        for row in clustering_data:
            samples_initial = row["samples_initial"]
            samples_final = row["samples_final"]
            # samples_initial_bootstrap = row["samples_initial_bootstrap"]
            idx = row["idx"]

            sample_to_cluster = row["sample_to_cluster_assignments"]
            cluster_to_responses = row["clusters_to_sample_assignments"]
            #Response to cluster re-expression
            clusters_initial = [sample_to_cluster[sample] for sample in samples_initial]
            clusters_final = [sample_to_cluster[sample] for sample in samples_final]            

            distribution_support = list(cluster_to_responses.keys())
            distribution_support = [int(x) for x in distribution_support]
            row["distribution_support"] = distribution_support

            distribution_probs_initial = get_dist_probs(distribution_support,clusters_initial)
            distribution_probs_final = get_dist_probs(distribution_support,clusters_final)
            row["distribution_probs_initial"] = distribution_probs_initial
            row["distribution_probs_final"] = distribution_probs_final
            
            tvd_initial_vs_final = get_tvd(distribution_probs_initial, distribution_probs_final)
            row["tvd_initial_vs_final"] = tvd_initial_vs_final

        with open(distribution_analysis_file, "w") as f:
            json.dump(clustering_data, f, indent=2)
    else:
        print(f"Found analyses data at {distribution_analysis_file}.")

def get_dist_probs(support, counts_support):
    """Receives the desired distribution support and a list with repeated counts for each of the elements in the support.
    e.g. support: [1,2,3,4] counts_support: [2,2,1,3,1,1,1]
    Returns the probability of each of the elements of the support given their counts.
    e.g. probs: [4/7, 2/7, 1/7, 0]"""
    total_counts = len(counts_support)
    counts_per_element = dict(Counter(counts_support))

    probs = [counts_per_element[element]/total_counts if element in list(counts_per_element.keys()) else 0 for element in support]
    return probs

def get_tvd(probs1, probs2):
    """Receives list of probabilities from 2 distributions with a shared support 
    and returns ther Total Variation Distance = 1/2 * Sum_{x} |P(x) - Q(x)|"""
    abs_differences = [abs(p - probs2[i]) for i,p in enumerate(probs1)]
    tvd = sum(abs_differences)/2
    return tvd

if __name__ == "__main__":
    main()
