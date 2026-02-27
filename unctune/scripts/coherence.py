from unctune.models import GenerationConfig, OpenAIModel
from unctune.scripts.utils import load_model, load_prompt

from pathlib import Path
from tqdm import tqdm
from collections import defaultdict

import argparse
import torch
import json
import numpy as np

from datasets import load_dataset as hf_load_dataset
from sentence_transformers import SentenceTransformer


def load_popqa_tp(filter_props=None, max_clusters=None):
    """Load PopQA-TP from HuggingFace and group into paraphrase clusters."""
    dataset = hf_load_dataset("ibm-research/popqa-tp", split="train")
    df = dataset.to_pandas()

    if filter_props:
        df = df[df["prop"].isin(filter_props)]

    clusters = []
    for cluster_id, group in df.groupby("id"):
        group = group.sort_values("template_id")
        clusters.append({
            "cluster_id": int(cluster_id),
            "prop": group["prop"].iloc[0],
            "questions": group["paraphrase"].tolist(),
            "possible_answers": group["possible_answers"].iloc[0],
        })

    clusters.sort(key=lambda c: c["cluster_id"])

    if max_clusters is not None:
        clusters = clusters[:max_clusters]

    return clusters


def compute_pairwise_cosine_similarity(embeddings):
    """Compute average pairwise cosine similarity for a set of embeddings."""
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    normalized = embeddings / norms

    n = len(embeddings)
    if n < 2:
        return 1.0

    similarity_matrix = normalized @ normalized.T

    # Mean of upper triangle (excluding diagonal)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += similarity_matrix[i, j]
            count += 1

    return total / count


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate semantic consistency (coherence) using PopQA-TP."
    )
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Model name (e.g. olmo2-13b-sft or olmo2-local:interweave-1e-5-july)")
    parser.add_argument("--prompt", "-p", type=str, default="vanilla",
                        help="Prompt variant (default: vanilla)")
    parser.add_argument("--embedding_model", type=str, default="all-mpnet-base-v2",
                        help="Sentence-transformers model for embeddings (default: all-mpnet-base-v2)")
    parser.add_argument("--filter_props", type=str, default=None,
                        help="Comma-separated props to filter (e.g. director,author,occupation)")
    parser.add_argument("--max_clusters", type=int, default=None,
                        help="Limit to first N clusters (for debugging)")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Batch size for embedding computation (default: 32)")
    parser.add_argument("--device_map", type=str, default=None,
                        help='HF device_map for multi-GPU (e.g. "auto")')
    parser.add_argument("--override_files", action="store_true",
                        help="Force recomputation even if outputs exist")
    args = parser.parse_args()

    # Resolve model name for output file naming
    model_name = args.model
    if "local:" in model_name:
        path = Path(model_name.split(":")[1])
        model_name = path.name

    # Output paths
    project_root = Path(__file__).resolve().parent.parent / ".."
    output_dir = project_root / "outputs" / "eval_data"
    output_dir.mkdir(parents=True, exist_ok=True)

    props_str = f"-props={args.filter_props.replace(',', '_')}" if args.filter_props else ""
    cluster_str = f"-n={args.max_clusters}" if args.max_clusters else ""

    generations_file = output_dir / f"{model_name}-popqa_tp-{args.prompt}{props_str}{cluster_str}.generations.json"
    coherence_file = output_dir / f"{model_name}-popqa_tp-{args.prompt}{props_str}{cluster_str}.coherence.json"

    # Parse filter_props
    filter_props = args.filter_props.split(",") if args.filter_props else None

    # --- Stage 1: Generate answers ---
    if not generations_file.exists() or args.override_files:
        print("Loading PopQA-TP dataset...")
        clusters = load_popqa_tp(filter_props=filter_props, max_clusters=args.max_clusters)
        print(f"Loaded {len(clusters)} clusters.")

        print(f"Loading model: {args.model}")
        wrapper = load_model(args.model, device_map=args.device_map)
        prompt = load_prompt(args.prompt)

        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        gen_cfg = GenerationConfig(max_length=250, remove_prompt=True, device=device) \
            if not isinstance(wrapper, OpenAIModel) else None

        generation_data = []
        print("Generating greedy answers for each paraphrased question...")
        for cluster in tqdm(clusters, desc="Clusters"):
            cluster_answers = []
            for question in cluster["questions"]:
                model_input = prompt(question)
                answer = wrapper.greedy_decode([model_input], cfg=gen_cfg)
                cluster_answers.append(answer[0])

            generation_data.append({
                "cluster_id": cluster["cluster_id"],
                "prop": cluster["prop"],
                "questions": cluster["questions"],
                "answers": cluster_answers,
                "possible_answers": cluster["possible_answers"],
            })

        with open(generations_file, "w") as f:
            json.dump({
                "metadata": {
                    "model": args.model,
                    "prompt": args.prompt,
                    "filter_props": filter_props,
                    "max_clusters": args.max_clusters,
                    "num_clusters": len(generation_data),
                },
                "data": generation_data
            }, f, indent=2)
        print(f"Generations saved to {generations_file}.")

        del wrapper
        torch.cuda.empty_cache()
    else:
        print(f"Found generations at {generations_file}.")

    # --- Stage 2: Embed and compute coherence ---
    if not coherence_file.exists() or args.override_files:
        print(f"Loading generations from {generations_file}...")
        with open(generations_file, "r") as f:
            gen_data = json.load(f)
        generation_data = gen_data["data"]

        print(f"Loading embedding model: {args.embedding_model}")
        embedder = SentenceTransformer(args.embedding_model)

        print("Computing semantic consistency per cluster...")
        coherence_data = []
        all_similarities = []

        for cluster in tqdm(generation_data, desc="Embedding"):
            answers = cluster["answers"]
            embeddings = embedder.encode(answers, batch_size=args.batch_size,
                                         show_progress_bar=False, convert_to_numpy=True)
            similarity = compute_pairwise_cosine_similarity(embeddings)
            all_similarities.append(similarity)
            coherence_data.append({
                "cluster_id": cluster["cluster_id"],
                "prop": cluster["prop"],
                "num_paraphrases": len(answers),
                "mean_pairwise_cosine_similarity": float(similarity),
                "questions": cluster["questions"],
                "answers": answers,
            })

        # Per-prop aggregates
        prop_similarities = defaultdict(list)
        for entry in coherence_data:
            prop_similarities[entry["prop"]].append(entry["mean_pairwise_cosine_similarity"])

        per_prop = {}
        for prop, sims in sorted(prop_similarities.items()):
            per_prop[prop] = {
                "mean": float(np.mean(sims)),
                "std": float(np.std(sims)),
                "count": len(sims),
            }

        # Summary
        sims_array = np.array(all_similarities)
        summary = {
            "mean_coherence": float(np.mean(sims_array)),
            "median_coherence": float(np.median(sims_array)),
            "std_coherence": float(np.std(sims_array)),
            "per_prop_coherence": per_prop,
        }

        results = {
            "metadata": {
                "model": args.model,
                "dataset": "popqa_tp",
                "prompt": args.prompt,
                "embedding_model": args.embedding_model,
                "num_clusters": len(coherence_data),
                "num_questions": sum(c["num_paraphrases"] for c in coherence_data),
                "filter_props": filter_props,
                "max_clusters": args.max_clusters,
            },
            "summary": summary,
            "data": coherence_data,
        }

        with open(coherence_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Coherence results saved to {coherence_file}.")

        # Print summary
        print(f"\n{'='*50}")
        print(f"Semantic Consistency (Coherence) Results")
        print(f"{'='*50}")
        print(f"Model: {args.model}")
        print(f"Clusters evaluated: {len(coherence_data)}")
        print(f"Mean coherence: {summary['mean_coherence']:.4f}")
        print(f"Median coherence: {summary['median_coherence']:.4f}")
        print(f"Std coherence: {summary['std_coherence']:.4f}")
        print(f"\nPer-property coherence:")
        for prop, stats in per_prop.items():
            print(f"  {prop}: {stats['mean']:.4f} +/- {stats['std']:.4f} (n={stats['count']})")
    else:
        print(f"Found coherence results at {coherence_file}.")
        with open(coherence_file, "r") as f:
            results = json.load(f)
        summary = results["summary"]
        print(f"Mean coherence: {summary['mean_coherence']:.4f}")


if __name__ == "__main__":
    main()
