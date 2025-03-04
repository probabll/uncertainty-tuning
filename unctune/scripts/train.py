import argparse
import json
import numpy as np
import pandas as pd
import torch

from pathlib import Path
from unctune.scripts.utils import load_model
from unctune.prompts import UncertaintyPrompt
from unctune.models import GenerationConfig
from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from transformers import DataCollatorForSeq2Seq

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="olmo2-13b") 
    parser.add_argument("--stratify", action="store_true")
    parser.add_argument("--num_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--validation_split", type=float, default=0.1)
    args = parser.parse_args()
    
    # Load the dataset
    prompt = UncertaintyPrompt()
    train, valid = load_training_data(args.model, prompt, step_size=500, stratify=args.stratify, validation_split=args.validation_split)

    # Load the model and tokenizer
    model = load_model(args.model)
    tokenizer = model.tokenizer
    hf_model = model.model

    def format_example(example):
        chat_template_fn = model._apply_chat_template if hasattr(model, "_apply_chat_template") else lambda x: x
        input_text = chat_template_fn(example["question"])
        target_text = example["answer_unc_hedged"] 
        return {"input_text": input_text, "target_text": target_text}
    train = train.map(format_example, batched=True)
    valid = valid.map(format_example, batched=True)

    # Tokenize the dataset
    def tokenize_fn(examples):
        model_inputs = tokenizer(
            examples["input_text"], 
            padding="longest", 
            truncation=True, 
            max_length=2048
        )

        with tokenizer.as_target_tokenizer():
            labels = tokenizer(
                examples["target_text"], 
                padding="longest", 
                truncation=True, 
                max_length=2048
            )

        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    train = train.map(tokenize_fn, batched=True)
    valid = valid.map(tokenize_fn, batched=True)

    # Use a GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    hf_model.to(device).eval()    

    project_root = Path(__file__).resolve().parent.parent / ".."
    checkpoint_dir = project_root / "outputs" / "model_checkpoints"
    checkpoint_dir.mkdir(exist_ok=True)
    log_dir = project_root / "outputs" / "training_logs"
    log_dir.mkdir(exist_ok=True)
    training_args = TrainingArguments(  
        output_dir=checkpoint_dir,
        evaluation_strategy=None, # TODO
        save_strategy="epoch",                                 
        logging_dir=log_dir,                                   
        per_device_train_batch_size=args.batch_size,           
        per_device_eval_batch_size=args.batch_size,            
        num_train_epochs=args.num_epochs,                      
        weight_decay=0.,              
        logging_steps=10,             
    )

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=hf_model)
    trainer = Trainer(
        model=hf_model,
        args=training_args,
        train_dataset=train,                  
        eval_dataset=None, # TODO     
        data_collator=data_collator,  
    )

    trainer.train()

def load_training_data(model_name, prompt, step_size=500, stratify=True, validation_split=0.1):
    project_root = Path(__file__).resolve().parent.parent / ".."
    output_dir = project_root / "outputs" / "train_data"
    filename_template = f"{str(output_dir)}/{model_name}-popqa_training-generate_train.training_data-{model_name}-<lower>-<upper>.json"
    start_idx = 0
    stop_idx = step_size

    dataframe = pd.DataFrame()
    while True:
        filename = Path(filename_template.replace("<lower>", str(start_idx)).replace("<upper>", str(stop_idx)))
        if not filename.exists():
            break

        # Load the training data chunk
        with open(filename, 'r') as f:
            data = json.load(f)["data"] # question, answer, confidence, uncertainty_hedger, answer_unc_hedged
        print(f"{filename}:")

        dataframe = pd.concat([dataframe, pd.DataFrame(data)], ignore_index=True)

        start_idx = stop_idx
        stop_idx += step_size

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
        dataframe = dataframe.groupby("confidence_bin", group_keys=False).apply(lambda x: x.sample(n=n_per_bin, replace=True))

        # Shuffle the stratified data
        dataframe = dataframe.sample(frac=1).reset_index(drop=True)

        print("After stratification:")
        print(dataframe["confidence_bin"].value_counts(normalize=True).sort_index())

    # Apply the prompt, strip some whitespace away
    dataframe["question"] = dataframe["question"].apply(lambda x: prompt(x))
    dataframe["answer_unc_hedged"] = dataframe["answer_unc_hedged"].apply(lambda x: x[0].strip())

    # Drop irrelevant columns
    dataframe = dataframe.drop(columns=["confidence", "uncertainty_hedger", "answer", "confidence_bin"])

    # Convert into a HuggingFace Dataset
    dataset = Dataset.from_pandas(dataframe)

    # Split the dataset into training and validation sets
    validation_size = int(len(dataset) * validation_split) if isinstance(validation_split, float) else validation_split
    dataset = dataset.train_test_split(test_size=validation_size)
    return dataset["train"], dataset["test"]