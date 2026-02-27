# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Faithful Uncertainty Tuning (FUT) — a research project that teaches language models to express calibrated uncertainty through natural language hedging phrases. The pipeline generates synthetic training data, fine-tunes models via SFT, and evaluates faithfulness using the cMFG metric (Calibrated Mean Faithfulness per Confidence Bin).

## Setup

```bash
pip install -e .
```

Requires GPU (CUDA) for model inference and training. Uses wandb for experiment tracking (project: "unctune", entity: "unctune").

## Key Commands

```bash
# Generate synthetic FUT training data
python3 unctune/scripts/generate.py --model olmo2-13b-sft

# Train a FUT model (CLI entry point)
unctune-train --model olmo2-13b-sft --prompt generate_train --data interweave --name olmo2-13b-fut --stratify --output_dir ./models/

# Evaluate faithfulness (cMFG)
unctune-evaluate --dataset popqa --model olmo2-13b-sft --prompt vanilla --decisiveness_model gpt-5 --nli_model gpt-5

# Evaluate accuracy (Lenient EM)
unctune-evaluate --metric accuracy --dataset popqa --model olmo2-13b-sft --prompt vanilla

# Evaluate perplexity on WikiText (language modeling capability)
unctune-perplexity --model olmo2-13b-sft
unctune-perplexity --model olmo2-local:interweave-1e-5-july
```

Load local fine-tuned models with `--model local:/path/to/model` or `--model olmo2-local:/path/to/model`.

### Multi-GPU Support

For models that don't fit on a single GPU, pass `--device_map auto` to shard across available devices:

```bash
unctune-perplexity --model olmo2-13b-sft --device_map auto
```

The `device_map` parameter is threaded through `load_model()` → model constructors → `from_pretrained()`. Supported by `HuggingFaceModel` and `Olmo2`.

## Architecture

### Three-Stage Pipeline

1. **generate.py** — For each question: greedy decode + 20 samples → NLI-based confidence scoring → verbal uncertainty interweaving. Outputs to `outputs/train_data/`.
2. **train.py** — SFT training using `trl.SFTTrainer` with HuggingFace chat format (`<|assistant|>\n` response template). Optional LoRA/QLoRA. Outputs to `outputs/fine-tuned/`.
3. **evaluate.py** — Generates responses → extracts decisiveness via LLM judge → computes NLI-based confidence → calculates per-bin faithfulness and cMFG. Supports parallel API calls via `--num_workers` (default: 8). Outputs to `outputs/eval_data/`.
4. **perplexity.py** — Computes perplexity on WikiText using strided sliding window. Loads dataset via HuggingFace `datasets`, concatenates text, and scores with configurable `--stride` and `--max_length`. Outputs to `outputs/eval_data/`.

Each stage caches intermediate results as parquet/JSON files and skips recomputation if outputs exist. Use `--override_files` to force regeneration.

### Model Abstraction (`unctune/models/`)

`BaseModel` → `HuggingFaceModel` → model-specific subclasses (Olmo, Olmo2, Mistral, Llama, Qwen). `OpenAIModel` is a separate branch for API-based models. All models expose `greedy_decode(prompts, cfg)` and `sample(prompts, n, cfg)`. Model name strings are mapped in `unctune/scripts/utils.py:load_model()`.

### Dataset Abstraction (`unctune/datasets/`)

`BaseDataset` with `__getitem__` returning `(idx, question, answers)`. Datasets auto-download to `data/`. Available: `popqa`, `nq`, `triviaqa`, `popqa_training`, `popqa-validation`.

### Confidence → Hedge Mapping

Continuous confidence [0,1] maps to 9 verbal phrases (from "impossible" at ~0.07 to "certain" at ~0.92). Defined in `unctune/confidence_verbal.py`. Alternative and stochastic mappings available via `--use_alternative_mapping` / `--use_stochastic_mapping`.

### Key Metrics

- **Confidence**: `1.0 - (contradictions / num_samples)` where contradictions are detected via NLI (DeBERTa or LLM-based)
- **Decisiveness**: LLM judge extracts assertion-level confidence from hedging language
- **Faithfulness**: `1.0 - mean(|confidence - decisiveness|)` per response
- **cMFG**: Mean faithfulness averaged uniformly across 10 confidence bins

## Prompt Variants

Loaded via `load_prompt(name)` in utils.py: `base`, `vanilla`, `generate_train`, `uncertainty`, `metsens+hedge`.

## Training Data Formats

Training uses two uncertainty integration styles selected via `--data`:
- `interweave` — hedging phrases woven into the answer text
- `postfix` — hedging phrases appended after the answer

## Local Model Checkpoints

- `interweave-1e-5-july/` — Pre-trained FUT checkpoint: OLMo-2-1124-13B-SFT fine-tuned with interweave data at lr=1e-5 (July 2025). Full-weight safetensors (~26 GB). Load via `--model olmo2-local:interweave-1e-5-july`.

## Reference Materials (`etc/`)

- `Teaching_LMs_to_faithfully_express_their_uncertainty.pdf` — The FUT paper.
- `Evaluating LLM Coherence and Language Modeling.md` — Notes on reviewer response: measuring semantic consistency (PopQA-TP) and perplexity (WikiText-103) to show FUT doesn't degrade coherence or language modeling.
- `Scherm_afbeelding 2026-02-25 om 17.18.07.png` — Reference screenshot.

## File Naming Conventions

Output files follow the pattern: `{model}-{dataset}-{prompt}[-{options}].{stage}[-{judge_model}].{ext}`

Examples:
- `olmo2-13b-sft-popqa_training-generate_train-0-5654.parquet` (generations)
- `olmo2-13b-sft-popqa_training-generate_train.confidence-deberta-large-0-5654.json` (confidence scores)
- `olmo2-13b-sft-popqa-vanilla.decisiveness-gpt-5.json` (evaluation decisiveness)
- `olmo2-13b-sft-wikitext-103-v1-test.perplexity.json` (perplexity results)
