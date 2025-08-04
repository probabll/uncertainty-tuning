import json
import pandas as pd

from pathlib import Path
from sklearn.model_selection import train_test_split
from datasets import Dataset, DatasetDict

from unctune.prompts import GenerateTrainingDataPrompt
from unctune.datasets.base_dataset import BaseDataset

class PopQAValidation(BaseDataset):

    def __init__(self):
        super().__init__()

    def _load_data(self):
        _, validation_data = load_training_data(GenerateTrainingDataPrompt(), stratify=True)        
        return validation_data
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """ Returns the questions and answers as a tuple at the given index or slice. """
        if isinstance(idx, slice):
            return self.data.iloc[idx].index.values, self.data.iloc[idx]["question"].values, self.data.iloc[idx]["references"].values
        else:
            return int(self.data.iloc[idx].name), self.data.iloc[idx]["question"], self.data.iloc[idx]["references"]
        
def load_training_data(prompt, filename=None, stratify=False, 
                       validation_split=100, postfix=False):
    project_root = Path(__file__).resolve().parent.parent / ".."
    train_data_dir = project_root / "outputs" / "train_data"
    target_column = "answer_unc_hedged_postfix" if postfix else "answer_unc_hedged_interweave"

    if filename is None:
        filename = train_data_dir / "training-data-10-07.json"
    with open(filename, 'r') as f:
        data = json.load(f)
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