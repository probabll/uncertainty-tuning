import argparse
import json
import numpy as np
import pandas as pd
import torch
import wandb
import time

from pathlib import Path
from unctune.scripts.utils import load_model, load_prompt
from unctune.scripts.compute_semantic_TVD_biNLI import compute_semantic_TVD
from unctune.models import GenerationConfig, Olmo2
from datasets import Dataset
from transformers import Trainer, TrainingArguments, AutoModelForSequenceClassification, AutoTokenizer
from transformers import EvalPrediction
from transformers import TrainerCallback
from sklearn.model_selection import train_test_split
from trl import SFTConfig, SFTTrainer, DataCollatorForCompletionOnlyLM
from datasets import DatasetDict

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", "-m", type=str, default="olmo2-13b-sft")
    parser.add_argument("--data", "-d", type=str, default=None)
    parser.add_argument("--stratify", action="store_true")
    parser.add_argument("--prompt", "-p", type=str, default="generate_train")
    parser.add_argument("--validation_split", type=float, default=100)
    parser.add_argument("--name", "-n", type=str, default="")
    parser.add_argument("--num_epochs", "-e", type=int, default=3)
    parser.add_argument("--learning_rate", "-lr", type=float, default=1e-5)
    args = parser.parse_args()
    
    run_name = args.name if args.name else f"{args.model}-{args.prompt}-mle"
    wandb.init(project="unctune", entity="unctune", name=run_name)

    # Load the dataset
    prompt = load_prompt(args.prompt)
    if args.data == "unprocessed":
        dataset = load_unprocessed_samples(prompt)
        eval_data = load_unprocessed_samples(prompt, retain_structure=True)["test"]
    else:
        dataset = load_training_data(args.model, prompt, stratify=args.stratify, 
                                    validation_split=args.validation_split,
                                    postfix=(args.data == "postfix"))        
        eval_data = load_training_data(args.model, prompt, stratify=args.stratify, 
                                    validation_split=args.validation_split,
                                    postfix=(args.data == "postfix"), retain_structure=True)["test"] 
        

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
        gradient_accumulation_steps=4,
        per_device_train_batch_size=8,
        auto_find_batch_size=False,

        max_seq_length=256, # TODO changed back again from 512, is this enough?
        packing=False,
        num_train_epochs=args.num_epochs, # 20
        
        learning_rate=args.learning_rate, # was 5e-6
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        optim="paged_adamw_8bit",

        save_strategy="no",

        eval_strategy="steps",
        eval_steps=250, # TODO
        logging_steps=50,
        logging_dir=None,
        output_dir=output_dir,
        report_to="wandb", 
        run_name=run_name, 
    )

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template="<|assistant|>\n",
        mlm=False
    )

    validator = ValidationCallback(tokenizer, eval_data, eval_every=1000) # TODO
    trainer = SFTTrainer(
        model=hf_model,
        processing_class=tokenizer,
        args=sft_config,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=collator,
        callbacks=[validator],
    )

    results = trainer.evaluate()
    step = max(1, trainer.state.global_step)
    # for key, value in results.items():
    #     wandb.log({f"eval/{key}": value}, step=step)

    print(f"Using batch size {trainer.args.per_device_train_batch_size} and gradient accumulation steps {trainer.args.gradient_accumulation_steps}")

    trainer.train()

    print(f"Saving the fine-tuned model to {output_dir}")
    hf_model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

class ValidationCallback(TrainerCallback):

    def __init__(self, tokenizer, eval_data, eval_every):
        super().__init__()
        self.tokenizer = tokenizer
        self.eval_data = eval_data
        self.initial_samples = None
        self.eval_every = eval_every
        self.initial_tvd = None

        self.nli_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
        self.nli_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli")

    def _compute_perplexity(self, hf_model, dataloader):    
        with torch.no_grad():
            total_loss = 0.0
            total_tokens = 0    
            for batch in dataloader:
                
                input_ids = batch["input_ids"].to(hf_model.device)
                attention_mask = batch["attention_mask"].to(hf_model.device)
                labels = batch["labels"].to(hf_model.device)

                # Forward pass
                outputs = hf_model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss

                # Accumulate loss and token count
                npredictions = labels[labels != -100].shape[0] # Count non-input, non-padding tokens
                total_loss += loss.item() * npredictions
                total_tokens += npredictions 
            avg_loss = total_loss / total_tokens
            perplexity = torch.exp(torch.tensor(avg_loss))
        return perplexity

    def _compute_faithfulness(self, hf_model):
        
        # For generation, we need left padding
        tokenizer = self.tokenizer
        padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        model = Olmo2(model=hf_model, tokenizer=self.tokenizer) # TODO hardcoded OLMO-2 now
        prompts = [self.eval_data[i*10]["question"] for i in range(98)] # TODO was 20
        references = [self.eval_data[i*10]["references"] for i in range(98)] # TODO was 20

        np.random.seed(42)
        prompts_subset = np.random.choice(prompts, size=20, replace=False)

        gen_cfg = GenerationConfig(max_length=250, remove_prompt=True, device=hf_model.device)
        greedy = model.greedy_decode(prompts, cfg=gen_cfg)

        samples = model.sample(prompts_subset, n=10, cfg=gen_cfg)
        samples = [samples[i*10:i*10+10] for i in range(0, len(prompts_subset))]

        tokenizer.padding_side = padding_side

        correct = []
        for greedy_gen, reference_list in zip(greedy, references):
            for ref in reference_list:
                if ref.lower() in greedy_gen.lower():
                    correct.append(1.)
                    break
            correct.append(0.)
        accuracy = np.mean(correct)

        return 0., prompts, greedy, prompts_subset, samples, accuracy

    def on_evaluate(self, args, state, control, **kwargs):
        step = max(1, state.global_step)
        # print(f"Evaluating at step {step}...")

        start_time = time.time()
        hf_model = kwargs.get("model")
        eval_dl = kwargs.get("eval_dataloader")
        perplexity = self._compute_perplexity(hf_model, eval_dl)

        wandb.log({"eval/perplexity": perplexity}, step=step)

        if step % self.eval_every == 0 or step == 1:
            # print("Running expensive evaluation...")
            faithfulness, prompts, greedy, prompts_subset, samples, acc = self._compute_faithfulness(hf_model)
            # wandb.log({"eval/faithfulness": faithfulness}, step=step)
            greedy_table = wandb.Table(columns=["prompt", "greedy"])
            sample_table = wandb.Table(columns=["prompt", "sample"])
            for prompt, greedy_i in zip(prompts, greedy):
                greedy_table.add_data(prompt, greedy_i)

            for prompt, samples_i in zip(prompts_subset, samples):
                for sample in samples_i:
                    sample_table.add_data(prompt, sample)
            wandb.log({f"eval-greedy/{step}": greedy_table}, step=step)
            wandb.log({f"eval-samples/{step}": sample_table}, step=step)
            wandb.log({"eval/QA-accuracy": acc}, step=step)

            if self.initial_samples is None:
                self.initial_samples = samples
                faithfulness, prompts, greedy, prompts_subset, samples, acc = self._compute_faithfulness(hf_model)
            hf_model.cpu()
            self.nli_model.cuda()
            tvds = []
            for prompt, samples_i, samples_j in zip(prompts_subset, self.initial_samples, samples):
                sem_tvd, _, _ = compute_semantic_TVD(prompt, samples_i, samples_j, self.nli_model, self.nli_tokenizer)
                tvds.append(sem_tvd)

            if self.initial_tvd is None:
                self.initial_tvd = np.array(tvds)

            wandb.log({"eval/semantic_TVD": np.mean(tvds)}, step=step)
            wandb.log({"eval/semantic_TVD_diff": np.mean(np.array(tvds) - self.initial_tvd)}, step=step)
            self.nli_model.cpu()
            hf_model.cuda()
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        # print(f"Evaluation took {elapsed_time:.2f} seconds")

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

def load_unprocessed_samples(prompt, step_size=500, validation_split=0.02, retain_structure=False):
    project_root = Path(__file__).resolve().parent.parent / ".."
    train_data_dir = project_root / "outputs" / "train_data" / "samples-sft"
    # filename_template = f"{str(train_data_dir)}/olmo2-13b-popqa_training-generate_train-<lower>-<upper>.parquet"
    start_idx = 0
    stop_idx = step_size

    dataframe = pd.DataFrame()
    for filename in train_data_dir.glob("*.parquet"):

        # filename = Path(filename_template.replace("<lower>", str(start_idx)).replace("<upper>", str(stop_idx)))
        # if not filename.exists():
        #     break

        chunk = pd.read_parquet(filename)
        print(f"Loaded {len(chunk)} training examples from {filename}")
        dataframe = pd.concat([dataframe, chunk], ignore_index=True)
        start_idx = stop_idx
        stop_idx += step_size

    # Flatten the samples
    dataframe = dataframe.explode("samples", ignore_index=True)
    dataframe = dataframe.rename(columns={"samples": "sample"})
    
    # Apply the prompt, strip some whitespace away
    dataframe["question"] = dataframe["question"].apply(lambda x: prompt(x))
    dataframe["sample"] = dataframe["sample"].apply(lambda x: x.strip())

    # Drop irrelevant columns
    if not retain_structure:
        dataframe = dataframe.drop(columns=["references", "ref", "greedy", "sample_refs"])

    validation_size = validation_split # needs to be a an integer
    train_df, test_df = grouped_train_test_split(dataframe, validation_size)

    # Convert into a HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    if retain_structure:
        return DatasetDict({"train": train_dataset, "test": test_dataset})

    def format_messages(example):
        return {
            "messages": [
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example["sample"]}
            ]
        }
    train_dataset = train_dataset.map(format_messages, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(format_messages, remove_columns=test_dataset.column_names)

    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    return dataset

def load_training_data(model_name, prompt, step_size=-1, stratify=False, 
                       validation_split=100, filename=None, postfix=False,
                       retain_structure=False):
    project_root = Path(__file__).resolve().parent.parent / ".."
    train_data_dir = project_root / "outputs" / "train_data"
    target_column = "answer_unc_hedged_postfix" if postfix else "answer_unc_hedged_interweave"

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
            if filename.suffix == ".json":
                with open(filename, 'r') as f:
                    chunk = json.load(f)["data"]
                chunk = pd.DataFrame(data)
            elif filename.suffix == ".parquet":
                chunk = pd.read_parquet(filename)
            else:
                raise ValueError(f"Unsupported file format: {filename.suffix}")
        
            print(f"Loaded {len(chunk)} training examples from {filename}")

            dataframe = pd.concat([dataframe, chunk], ignore_index=True)

            start_idx = stop_idx
            stop_idx += step_size
    else:
        if filename is None:
            # filename = Path(train_data_dir / f"{model_name}-popqa_training_all_generated_data_NEW.json")
            filename = Path(train_data_dir / "training-data-01-05.json")
        with open(filename, 'r') as f:
            data = json.load(f) #["data"]
        print(f"Loaded {len(data)} training examples from {filename}")
        dataframe = pd.DataFrame(data)

    if len(dataframe) == 0:
        raise ValueError(f"No training data found at {filename}.")
    
    # Add a confidence bin column for statistics, confidence values in 10 bins: [0, 0.1], (0.1, 0.2], ..., (0.9, 1.0]
    bins = [0.1 * i for i in range(11)]
    dataframe["confidence_bin"] = pd.cut(dataframe["confidence"], bins=bins, labels=False, include_lowest=True)
    if not retain_structure:
        print("Confidence bin distribution:")
        print(dataframe["confidence_bin"].value_counts(normalize=True).sort_index())
 
    # Apply the prompt, strip some whitespace away
    dataframe["question"] = dataframe["question"].apply(lambda x: prompt(x))
    dataframe[target_column] = dataframe[target_column].apply(lambda x: x.strip())

    # validation_size = int(len(dataframe) * validation_split) if isinstance(validation_split, float) else validation_split
    train_df, test_df = grouped_train_test_split(dataframe, validation_split)

    if retain_structure:
        train_dataset = Dataset.from_pandas(train_df)
        test_dataset = Dataset.from_pandas(test_df)
        return DatasetDict({"train": train_dataset, "test": test_dataset})

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
    test_df = test_df.drop(columns=["confidence", "uncertainty_hedger", "answer", "confidence_bin"])

    # Convert into a HuggingFace Dataset
    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)

    def format_messages(example):
        return {
            "messages": [
                {"role": "user", "content": example["question"]},
                {"role": "assistant", "content": example[target_column]}
            ]
        }
    train_dataset = train_dataset.map(format_messages, remove_columns=train_dataset.column_names)
    test_dataset = test_dataset.map(format_messages, remove_columns=test_dataset.column_names)
    dataset = DatasetDict({"train": train_dataset, "test": test_dataset})

    print(f"Training set size: {len(dataset['train'])}")
    print(f"Validation set size: {len(dataset['test'])}")
    print(f"Example training datapoint: {dataset['train'][0]}")

    return dataset

    # # Convert into a HuggingFace Dataset
    # dataset = Dataset.from_pandas(dataframe)

    # def format_messages(example):
    #     return {
    #         "messages": [
    #             {"role": "user", "content": example["question"]},
    #             {"role": "assistant", "content": example[target_column]}
    #         ]
    #     }
    # dataset = dataset.map(format_messages, remove_columns=dataset.column_names)

    # # Split the dataset into training and validation sets
    # validation_size = int(len(dataset) * validation_split) if isinstance(validation_split, float) else validation_split
    # dataset = dataset.train_test_split(test_size=validation_size, seed=42)

    # print(f"Training set size: {len(dataset['train'])}")
    # print(f"Validation set size: {len(dataset['test'])}")


    # return dataset
