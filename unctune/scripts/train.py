import argparse
import json
import numpy as np
import pandas as pd
import torch
import wandb
import time

from pathlib import Path
from unctune.scripts.utils import load_model, load_prompt
from unctune.scripts.compute_semantic_TVD_biNLI import compute_semantic_TVD, nli_bidirectional_entailment
from unctune.decisiveness import DecisivenessExtractorSimple
from unctune.models import GenerationConfig, Olmo2
from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from transformers import EvalPrediction
from transformers import TrainerCallback
from sklearn.model_selection import train_test_split
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import DatasetDict
from peft import LoraConfig, get_peft_model
from transformers import BitsAndBytesConfig

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="olmo2-13b-sft")
    parser.add_argument("--lora", "-l", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--quant", "-q", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--data", "-d", type=str, default=None)
    parser.add_argument("--stratify", action="store_true")
    parser.add_argument("--prompt", "-p", type=str, default="generate_train")
    parser.add_argument("--validation_split", type=float, default=100)
    parser.add_argument("--name", "-n", type=str, default="")
    parser.add_argument("--num_epochs", "-e", type=int, default=3)
    parser.add_argument("--eval_every", type=int, default=250, help="Evaluate every N steps")
    parser.add_argument("--max_sequence_length", type=int, default=256, help="Maximum sequence length for training")
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--optim", "-o", type=str, default="paged_adamw_8bit")
    parser.add_argument("--use_altmap", action="store_true")
    parser.add_argument("--stochastic", action="store_true")
    args = parser.parse_args()
    print(args)

    run_name = args.name if args.name else f"{args.model}-{args.prompt}-mle"
    wandb.init(project="unctune", entity="unctune", name=run_name)

    # Load the dataset
    prompt = load_prompt(args.prompt)
    dataset, raw_validation = load_training_data(args.model, prompt, stratify=args.stratify, 
                                                 validation_split=args.validation_split,
                                                 postfix=(args.data == "postfix"), use_altmap=args.use_altmap, stochastic=args.stochastic)        

    # Load the model and tokenizer
    quantization_config = None
    if args.quant:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch.bfloat16
        )

    model = load_model(args.model, quantization_config)
    tokenizer = model.tokenizer
    hf_model = model.model

    # Logging and output directories
    if not args.output_dir:
        project_root = Path(__file__).resolve().parent.parent / ".."
        output_dir= project_root / "outputs" / "fine-tuned" / run_name
    else:
        output_dir = Path(args.output_dir) / run_name
    output_dir.mkdir(exist_ok=True)

    sft_config = SFTConfig(
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False}, 
        gradient_accumulation_steps=4,
        per_device_train_batch_size=8,
        auto_find_batch_size=False,

        max_seq_length=args.max_sequence_length,
        packing=False,
        num_train_epochs=args.num_epochs,
        
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="paged_adamw_8bit",

        save_strategy="no",

        eval_strategy="steps",
        eval_steps=args.eval_every, 
        logging_steps=50,
        logging_dir=None,
        output_dir=output_dir,
        report_to="wandb", 
        run_name=run_name,
        label_names=["labels"]
    )

    response_template = "[/INST]" if args.model == "mistral-12b" else "<|assistant|>\n"
    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template=response_template,
        mlm=False
    )

    # LoRA training
    if args.lora:
        peft_config = LoraConfig(
            r=16, # 8 is common, 16 gives more expressivity
            lora_alpha=32, #  2 * r
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            task_type="CAUSAL_LM",
        )
        hf_model = get_peft_model(hf_model, peft_config)
        hf_model.print_trainable_parameters()

    trainer = SFTTrainer(
        model=hf_model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["validation"],
        data_collator=collator
    )

    results = trainer.evaluate()

    print(f"Using batch size {trainer.args.per_device_train_batch_size} and gradient accumulation steps {trainer.args.gradient_accumulation_steps}")

    trainer.train()

    print(f"Saving the fine-tuned model to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

def load_training_data(model_name, prompt, filename=None, stratify=False, 
                       validation_split=100, postfix=False, use_altmap=False, stochastic=False):
    project_root = Path(__file__).resolve().parent.parent / ".."
    train_data_dir = project_root / "outputs" / "train_data"
    target_column = "answer_unc_hedged_postfix" if postfix else "answer_unc_hedged_interweave"

    if filename is None:
        mapping_str = "-altmap" if use_altmap else ""
        mapping_str = "-stochmap" if stochastic else mapping_str
        filename = train_data_dir / f"{model_name}{mapping_str}-training.json"
    with open(filename, 'r') as f:
        data = json.load(f)

    # TODO: hotfix
    if use_altmap:
        data = data["data"]
    print(f"Loaded {len(data)} training examples from {filename}")
    dataframe = pd.DataFrame(data)

    if len(dataframe) == 0:
        raise ValueError(f"No training data found at {filename}.")
    
    # Add a confidence bin column for statistics, confidence values in 10 bins: [0, 0.1], (0.1, 0.2], ..., (0.9, 1.0]
    bins = [0.1 * i for i in range(11)]
    dataframe["confidence_bin"] = pd.cut(dataframe["confidence"], bins=bins, labels=False, include_lowest=True)
    print("Confidence bin distribution:")
    print(dataframe["confidence_bin"].value_counts(normalize=True).sort_index())
 
    # Apply the prompt, strip some whitespace away
    dataframe["question"] = dataframe["question"].apply(lambda x: prompt(x))
    dataframe[target_column] = dataframe[target_column].apply(lambda x: x.strip())

    # validation_size = int(len(dataframe) * validation_split) if isinstance(validation_split, float) else validation_split
    train_df, val_df_raw = grouped_train_test_split(dataframe, validation_split)

    # If stratify==True, stratify according to the confidence bins
    if stratify:
        print("Stratiyfing the data...")

        # Stratify the data
        n_per_bin = len(train_df) // 10
        train_df = train_df.groupby("confidence_bin", group_keys=False).apply(lambda x: x.sample(n=n_per_bin, replace=True, random_state=42))

        # Shuffle the stratified data
        train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

        print("After stratification:")
        print(train_df["confidence_bin"].value_counts(normalize=True).sort_index())

    # Drop irrelevant columns
    train_df = train_df.drop(columns=["confidence", "uncertainty_hedger", "answer", "confidence_bin"])
    val_df = val_df_raw.drop(columns=["confidence", "uncertainty_hedger", "answer", "confidence_bin"])

    # Convert into a HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)

    def format_messages(example):
        return {
            "messages": [
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example[target_column]}
            ]
        }
    train_dataset = train_dataset.map(format_messages, remove_columns=train_dataset.column_names)
    val_dataset = val_dataset.map(format_messages, remove_columns=val_dataset.column_names)
    dataset = DatasetDict({"train": train_dataset, "validation": val_dataset})

    print(f"Training set size: {len(dataset['train'])}")
    print(f"Validation set size: {len(dataset['validation'])}")
    print(f"Example training datapoint: {dataset['train'][0]}")

    return dataset, val_df_raw

def grouped_train_test_split(dataframe, n_val_questions):

    # Get unique questions
    unique_questions = dataframe["question"].unique()
    
    # Perform train-test split on the unique questions
    train_questions, test_questions = train_test_split(
        unique_questions, test_size=n_val_questions, random_state=42
    )

    # Filter the dataframe based on the split
    train_df = dataframe[dataframe["question"].isin(train_questions)]
    test_df = dataframe[dataframe["question"].isin(test_questions)]
    
    return train_df, test_df
