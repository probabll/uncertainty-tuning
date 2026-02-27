from unctune.scripts.utils import load_model

from pathlib import Path
from tqdm import tqdm

import argparse
import torch
import json

from datasets import load_dataset as hf_load_dataset


def main():
    parser = argparse.ArgumentParser(description="Compute perplexity on WikiText using a strided sliding window.")
    parser.add_argument("--model", "-m", type=str, required=True,
                        help="Model name (e.g. olmo2-13b-sft or olmo2-local:interweave-1e-5-july)")
    parser.add_argument("--dataset", type=str, default="wikitext-103-v1",
                        choices=["wikitext-103-v1", "wikitext-2-v1"],
                        help="WikiText variant to evaluate on")
    parser.add_argument("--split", type=str, default="test",
                        help="Dataset split (default: test)")
    parser.add_argument("--stride", type=int, default=512,
                        help="Sliding window stride in tokens (default: 512)")
    parser.add_argument("--max_length", type=int, default=2048,
                        help="Context window size in tokens (default: 2048)")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Number of windows per forward pass (default: 1)")
    parser.add_argument("--device_map", type=str, default=None,
                        help='HF device_map for multi-GPU (e.g. "auto")')
    parser.add_argument("--override_files", action="store_true",
                        help="Force recomputation even if output exists")
    args = parser.parse_args()

    # Resolve model name for output file
    model_name = args.model
    if "local:" in model_name:
        path = Path(model_name.split(":")[1])
        model_name = path.name

    # Output path
    project_root = Path(__file__).resolve().parent.parent / ".."
    output_dir = project_root / "outputs" / "eval_data"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / f"{model_name}-{args.dataset}-{args.split}.perplexity.json"

    if output_file.exists() and not args.override_files:
        print(f"Found existing results at {output_file}. Use --override_files to recompute.")
        with open(output_file, "r") as f:
            results = json.load(f)
        print(f"Perplexity: {results['perplexity']:.4f}")
        return

    # Load model
    print(f"Loading model: {args.model}")
    wrapper = load_model(args.model, device_map=args.device_map)
    model = wrapper.model
    tokenizer = wrapper.tokenizer

    if args.device_map:
        device = next(model.parameters()).device
    else:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
        model = model.to(device)
    model.eval()

    # Clamp max_length to model's max position embeddings if available
    if hasattr(model.config, "max_position_embeddings"):
        args.max_length = min(args.max_length, model.config.max_position_embeddings)
    max_length = args.max_length
    stride = args.stride

    # Load and tokenize WikiText
    print(f"Loading dataset: wikitext/{args.dataset} ({args.split} split)")
    dataset = hf_load_dataset("wikitext", args.dataset, split=args.split)

    # Concatenate all text and tokenize as one long sequence
    text = "\n\n".join([t for t in dataset["text"] if t.strip()])
    encodings = tokenizer(text, return_tensors="pt")
    input_ids = encodings.input_ids  # shape: (1, seq_len)
    seq_len = input_ids.size(1)
    print(f"Total tokens: {seq_len}")

    # Strided sliding window perplexity
    nlls = []
    total_tokens = 0
    prev_end = 0

    num_windows = max(1, (seq_len - max_length) // stride + 1)
    if seq_len <= max_length:
        num_windows = 1

    print(f"Computing perplexity (max_length={max_length}, stride={stride}, windows={num_windows})...")

    for begin_loc in tqdm(range(0, seq_len - 1, stride), desc="PPL"):
        end_loc = min(begin_loc + max_length, seq_len)
        trg_len = end_loc - prev_end  # number of new tokens to score

        window_input_ids = input_ids[:, begin_loc:end_loc].to(device)

        # Build target labels: mask out the context (overlapping) tokens with -100
        target_ids = window_input_ids.clone()
        target_ids[:, :-trg_len] = -100

        with torch.no_grad():
            outputs = model(window_input_ids, labels=target_ids)
            # outputs.loss is the mean cross-entropy over non-masked tokens
            neg_log_likelihood = outputs.loss

        # outputs.loss is averaged over the trg_len tokens; multiply back to get total
        nlls.append(neg_log_likelihood.float() * trg_len)
        total_tokens += trg_len

        prev_end = end_loc
        if end_loc >= seq_len:
            break

    ppl = torch.exp(torch.stack(nlls).sum() / total_tokens).item()

    print(f"\nPerplexity: {ppl:.4f}")
    print(f"Total tokens scored: {total_tokens}")

    # Save results
    results = {
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "perplexity": ppl,
        "total_tokens": total_tokens,
        "stride": stride,
        "max_length": max_length,
    }

    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to {output_file}")

    # Clean up
    del model
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
