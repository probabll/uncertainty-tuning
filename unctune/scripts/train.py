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
        label_names=["labels"]  # To avoid lora warning "No label_names provided for model class ".
    )

    collator = DataCollatorForCompletionOnlyLM(
        tokenizer=tokenizer,
        response_template="<|assistant|>\n",
        mlm=False
    )

    # We run the expensive validation every other evaluation step
    # validator = ValidationCallback(tokenizer, 
    #                                raw_validation, 
    #                                eval_every=args.eval_every*2,
    #                                nsamples=10,
    #                                subset_size=40,
    #                                max_sequence_length=args.max_sequence_length,
    #                                output_dir=output_dir) 

    # LoRA training
    if args.lora:
        peft_config = LoraConfig(
            r=16, # 8 is common, 16 gives more expressivity
            lora_alpha=32, #  2 * r
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"], # can add more modules here
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
        # callbacks=[validator],
    )

    results = trainer.evaluate()

    print(f"Using batch size {trainer.args.per_device_train_batch_size} and gradient accumulation steps {trainer.args.gradient_accumulation_steps}")

    trainer.train()

    print(f"Saving the fine-tuned model to {output_dir}")
    trainer.model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)

class ValidationCallback(TrainerCallback):

    def __init__(self, tokenizer, validation_dataset, eval_every, nsamples, max_sequence_length=256, subset_size=20, output_dir=None):
        super().__init__()
        self.tokenizer = tokenizer
        self.eval_every = eval_every
        self.max_sequence_length = max_sequence_length
        self.initial_samples = None
        self.nsamples = nsamples
        self.subset_size = subset_size
        self.output_dir = Path(output_dir) if output_dir else None
        self.best_cMFG = -float("inf")

        # We use 1 generation per unique prompt, and then subsample subset_size prompts from that
        self.val_data = validation_dataset.groupby("question", group_keys=False).apply(lambda x: x.sample(1, random_state=42))

        # Now balance the validation data across confidence bins
        # Not ideal, but we don't know greedy confidence bins
        bins = self.val_data["confidence_bin"].unique()
        n_bins = len(bins)
        n_per_bin = max(1, self.subset_size // n_bins)
        self.val_data = (
            self.val_data.groupby("confidence_bin", group_keys=False)
            .apply(lambda x: x.sample(min(len(x), n_per_bin), random_state=42))
            .reset_index(drop=True)
        )

        # self.nli_tokenizer = AutoTokenizer.from_pretrained("microsoft/deberta-large-mnli")
        # self.nli_model = AutoModelForSequenceClassification.from_pretrained("microsoft/deberta-large-mnli").cpu()

        # self.triviaqa_val_data = TriviaQAEval()

    def on_evaluate(self, args, state, control, **kwargs):
        step = max(1, state.global_step)
        print(f"Evaluating at step {step}...")

        start_time = time.time()
        hf_model = kwargs.get("model")
        eval_dl = kwargs.get("eval_dataloader")

        perplexity = self._compute_perplexity(hf_model, eval_dl)
        wandb.log({"eval/perplexity": perplexity}, step=step)

        # We don't run the expensive evaluations on every evaluation call
        # if step % self.eval_every == 0 or step == 1:
        #     greedy, samples = self._generate(hf_model)

        #     # We store the first batch of samples for TVD computation
        #     if self.initial_samples is None:
        #         self.initial_samples = samples

        #     # Log the samples and greedy generations
        #     prompts = self.val_data["question"].tolist()
        #     greedy_table = wandb.Table(columns=["prompt", "greedy"])
        #     sample_table = wandb.Table(columns=["prompt", "sample"])
        #     for prompt, greedy_i in zip(prompts, greedy):
        #         greedy_table.add_data(prompt, greedy_i)
        #     for prompt, samples_i in zip(prompts, samples):
        #         for sample in samples_i:
        #             sample_table.add_data(prompt, sample)
        #     wandb.log({f"eval-greedy/{step}": greedy_table}, step=step)
        #     wandb.log({f"eval-samples/{step}": sample_table}, step=step)

        #     # Compute accuracy
        #     references = self.val_data["references"]
        #     correct = []
        #     for greedy_gen, reference_list in zip(greedy, references):
        #         for ref in reference_list:
        #             if ref.lower() in greedy_gen.lower():
        #                 correct.append(1.)
        #                 break
        #         correct.append(0.)
        #     accuracy = np.mean(correct)
        #     wandb.log({"eval/QA-accuracy": accuracy}, step=step)

        #     # GPU/CPU management
        #     hf_model.cpu()
        #     self.nli_model.cuda()

        #     # Compute semantic TVD. Balanced across confidence bins somewhat based on the data selection procedure
        #     TVDs = []
        #     for prompt, samples_i, samples_j in zip(prompts, self.initial_samples, samples):
        #         sem_tvd, _, _ = compute_semantic_TVD(prompt, samples_i, samples_j, self.nli_model, self.nli_tokenizer)
        #         TVDs.append(sem_tvd)
        #     wandb.log({"eval/semantic_TVD": np.mean(TVDs)}, step=step)

        #     # Faithfulness evaluation
        #     # prompts, greedy
        #     cMFG = self._faithfulness(prompts, greedy, samples)
        #     wandb.log({"eval/cMFG": cMFG}, step=step)

        #     # Store best model based on faithfulness score
        #     if cMFG > self.best_cMFG and self.output_dir:
        #         self.best_cMFG = cMFG
        #         best_dir = self.output_dir / "best-faithfulness"
        #         best_dir.mkdir(parents=True, exist_ok=True)
        #         hf_model.save_pretrained(best_dir)
        #         self.tokenizer.save_pretrained(best_dir)
        #         print(f"[BEST] saved to {best_dir}  (cMFG={cMFG:.4f})")

        #     self.nli_model.cpu()
        #     hf_model.cuda()
            
        end_time = time.time()
        elapsed_time = end_time - start_time
        print(f"Evaluation took {elapsed_time:.2f} seconds")        

    def _faithfulness(self, prompts, greedy, samples):
        decisiveness_extractor = DecisivenessExtractorSimple()

        confidence_list = []
        decisiveness_list = []
        faithfulness_list = []

        for prompt, generation_to_evaluate, sample_list in zip(prompts, greedy, samples):
        
            # Confidence
            entailment = []
            for sample in sample_list:
                entails = nli_bidirectional_entailment(self.nli_model, self.nli_tokenizer,
                                    prompt, generation_to_evaluate, sample, lenient=True)
                entailment.append(entails)
                
            confidence = np.mean(entailment)
            confidence_list.append(confidence)

            # Decisiveness
            decisiveness = decisiveness_extractor(prompt, generation_to_evaluate)[0][1]
            decisiveness_list.append(decisiveness)

            # Faithfulness
            faithfulness_list.append(1.0 - np.abs(confidence - decisiveness))
            
        print("[FAITHFULNESS EVALUATION]")
        print(f"Confidence: {confidence_list}")
        print(f"Decisiveness: {decisiveness_list}")
        print(f"Faithfulness: {faithfulness_list}")

        # cMFG computation
        confidence_bins = [i/10 for i in range(1, 11)]
        bin_ids = []
        
        # Bin the data
        for confidence in confidence_list:
            bin_idx = 0
            for i, bin_val in enumerate(confidence_bins):
                if confidence <= bin_val:
                    bin_idx = i
                    break
            bin_ids.append(bin_idx)

        # MFG per bin
        MFG_per_bin = [0.0 for _ in range(len(confidence_bins))]
        for i in range(len(confidence_bins)):
            bin_faithfulnesses = [faithfulness_list[j] for j in range(len(faithfulness_list)) if bin_ids[j] == i]
            MFG_per_bin[i] = sum(bin_faithfulnesses)/len(bin_faithfulnesses) if len(bin_faithfulnesses) > 0 else 0.5

        # Assumes uniform distribution of confidence scores
        cMFG = sum(MFG_per_bin)/len(MFG_per_bin)

        return cMFG

    def _generate(self, hf_model):
        
        # For generation, we need left padding
        tokenizer = self.tokenizer
        padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"

        model = Olmo2(model=hf_model, tokenizer=self.tokenizer) # TODO we assume OLMO-2 now, have to see how to generalize this

        gen_cfg = GenerationConfig(max_length=self.max_sequence_length, 
                                   remove_prompt=True, device=hf_model.device)
        
        prompts = self.val_data["question"].tolist()
        greedy = model.greedy_decode(prompts, cfg=gen_cfg)
        samples = []
        for prompt in prompts:
            samples.append(model.sample([prompt], n=self.nsamples, cfg=gen_cfg))

        # Restore the model tokenizer
        tokenizer.padding_side = padding_side

        return greedy, samples
    
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

def load_training_data(model_name, prompt, filename=None, stratify=False, 
                       validation_split=100, postfix=False, use_altmap=False, stochastic=False):
    project_root = Path(__file__).resolve().parent.parent / ".."
    train_data_dir = project_root / "outputs" / "train_data"
    target_column = "answer_unc_hedged_postfix" if postfix else "answer_unc_hedged_interweave"

    if filename is None:
        if model_name == "olmo2-13b-sft":
            if not use_altmap:
                filename = train_data_dir / "training-data-10-07.json"
            else:
                if stochastic:
                    filename = train_data_dir / "olmo2-13b-sft-popqa_training-generate_train-stochmap.training_data-mistral-12b-0-5654.json"
                else:
                    filename = train_data_dir / "olmo2-13b-sft-popqa_training-generate_train-altmap.training_data-mistral-12b-0-5654.json"
        elif model_name == "olmo2-7b-sft":
            filename = train_data_dir / "olmo2-7b-training.json"
        elif model_name == "llama-8b-sft":
            filename = train_data_dir / "llama-8b-training.json"
        else:
            raise ValueError(f"No training data for {model_name}.")
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
