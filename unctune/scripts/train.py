import argparse
import json
import numpy as np
import pandas as pd
import torch
import wandb

from pathlib import Path
from unctune.scripts.utils import load_model, load_prompt
from unctune.models import GenerationConfig
from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from transformers import EvalPrediction
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="olmo2-13b") 
    parser.add_argument("--data", "-d", type=str, default=None)
    parser.add_argument("--stratify", action="store_true")
    parser.add_argument("--prompt", "-p", type=str, default="generate_train")
    parser.add_argument("--validation_split", type=float, default=0.1)
    parser.add_argument("--name", "-n", type=str, default="")
    parser.add_argument("--num_epochs", "-e", type=int, default=5)
    parser.add_argument("--learning_rate", "-lr", type=float, default=3e-7)
    args = parser.parse_args()
    
    run_name = args.name if args.name else f"{args.model}-{args.prompt}-mle"
    wandb.init(project="unctune", name=run_name)

    # Load the dataset
    prompt = load_prompt(args.prompt)
    dataset = load_training_data(args.model, prompt, stratify=args.stratify, 
                                 validation_split=args.validation_split, filename=args.data)

    # Load the model and tokenizer
    model = load_model(args.model)
    tokenizer = model.tokenizer
    hf_model = model.model

    # Logging and output directories
    project_root = Path(__file__).resolve().parent.parent / ".."
    output_dir= project_root / "outputs" / "fine-tuned" / run_name
    output_dir.mkdir(exist_ok=True)

    sft_config = SFTConfig(
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, 
        gradient_accumulation_steps=8,  # was 4
        per_device_train_batch_size=4, # was 8
        auto_find_batch_size=False,

        max_seq_length=512, # 256
        packing=False,
        num_train_epochs=args.num_epochs, # 20
        
        learning_rate=args.learning_rate, # was 5e-6
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="paged_adamw_8bit",

        save_strategy="no",

        eval_strategy="steps",
        eval_steps=250,
        logging_steps=50,
        logging_dir=None,
        output_dir=output_dir,
        report_to="wandb"
    )

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template="<|assistant|>\n",
        mlm=False
    )

    trainer = SFTTrainer(
        model=hf_model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator
    )

    print(f"Using batch size {trainer.args.per_device_train_batch_size} and gradient accumulation steps {trainer.args.gradient_accumulation_steps}")

    trainer.train()

    print(f"Saving the fine-tuned model to {output_dir}")
    hf_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def load_training_data(model_name, prompt, step_size=-1, stratify=False, validation_split=0.1, filename=None):
    project_root = Path(__file__).resolve().parent.parent / ".."
    train_data_dir = project_root / "outputs" / "train_data"

    if step_size > 0:
        filename_template = f"{str(train_data_dir)}/{model_name}-popqa_training-generate_train.training_data-{model_name}-<lower>-<upper>.json"
        start_idx = 0
        stop_idx = step_size

        dataframe = pd.DataFrame()
        while True:
            filename = Path(filename_template.replace("<lower>", str(start_idx)).replace("<upper>", str(stop_idx)))
            if not filename.exists():
                break

            # Load the training data chunk
            with open(filename, 'r') as f:
                data = json.load(f)["data"]
            print(f"Loaded {len(data)} training examples from {filename}")

            dataframe = pd.concat([dataframe, pd.DataFrame(data)], ignore_index=True)

            start_idx = stop_idx
            stop_idx += step_size
    else:
        if filename is None:
            filename = Path(train_data_dir / f"{model_name}-popqa_training_all_generated_data_NEW.json")
        with open(filename, 'r') as f:
            data = json.load(f)["data"]
        print(f"Loaded {len(data)} training examples from {filename}")
        dataframe = pd.DataFrame(data)

    if len(dataframe) == 0:
        raise ValueError(f"No training data found at {filename}.")
    
    # Add a confidence bin column for statistics, confidence values in 10 bins: [0, 0.1], (0.1, 0.2], ..., (0.9, 1.0]
    bins = [0.1 * i for i in range(11)]
    dataframe["confidence_bin"] = pd.cut(dataframe["confidence"], bins=bins, labels=False, include_lowest=True)
    print("Confidence bin distribution:")
    print(dataframe["confidence_bin"].value_counts(normalize=True).sort_index())

    # If stratify==True, stratify according to the confidence bins
    if stratify:
        print("Stratiyfing the data...")

        # Stratify the data
        n_per_bin = len(dataframe) // 10
        dataframe = dataframe.groupby("confidence_bin", group_keys=False).apply(lambda x: x.sample(n=n_per_bin, replace=True, random_state=42))

        # Shuffle the stratified data
        dataframe = dataframe.sample(frac=1, random_state=42).reset_index(drop=True)

        print("After stratification:")
        print(dataframe["confidence_bin"].value_counts(normalize=True).sort_index())

    # Apply the prompt, strip some whitespace away
    dataframe["question"] = dataframe["question"].apply(lambda x: prompt(x))
    dataframe["answer_unc_hedged"] = dataframe["answer_unc_hedged"].apply(lambda x: x.strip())

    # Drop irrelevant columns
    dataframe = dataframe.drop(columns=["confidence", "uncertainty_hedger", "answer", "confidence_bin"])

    # Convert into a HuggingFace Dataset
    dataset = Dataset.from_pandas(dataframe)

    # Reformat the dataset to chat template, one column named messages mapping to [{'role': 'user', 'content': 'question'}, {'role': 'system', 'content': 'answer_unc_hedged'}]
    def format_messages(example):
        return {
            "messages": [
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["answer_unc_hedged"]}
            ]
        }
    dataset = dataset.map(format_messages, remove_columns=dataset.column_names)

    # Split the dataset into training and validation sets
    validation_size = int(len(dataset) * validation_split) if isinstance(validation_split, float) else validation_split
    dataset = dataset.train_test_split(test_size=validation_size, seed=42)
    return dataset
