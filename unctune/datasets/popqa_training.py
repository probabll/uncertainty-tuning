from unctune.datasets.base_dataset import BaseDataset
import os
import urllib.request
import pandas as pd
from typing import Generic, TypeVar

class PopQATrain(BaseDataset):

    FILE_LOC = "https://raw.githubusercontent.com/AlexTMallen/adaptive-retrieval/main/data/popQA.tsv"
    FILENAME = "popQA.tsv"

    def __init__(self, min_gen_index: int, max_gen_index: int):
        """
        PopQA dataset preprocessed as in https://arxiv.org/abs/2405.16908.
        """
        self.min_gen_index = min_gen_index
        self.max_gen_index = max_gen_index
        super().__init__()
        
    def _load_data(self):
        data = self._load_or_download_data_file()
        data = self._preprocess(data, self.min_gen_index, self.max_gen_index)
        return data

    def _load_or_download_data_file(self):
        data_file = self.data_dir / self.FILENAME
        if not os.path.exists(data_file):
            print(f"Downloading PopQA dataset from {self.FILE_LOC}...")
            try:
                urllib.request.urlretrieve(self.FILE_LOC, data_file)
            except Exception as e:
                print(f"Failed to download PopQA dataset: {e}")
                raise e
        return pd.read_csv(data_file, skiprows=0, sep="\t")

    def _preprocess(self, data, min_gen_index, max_gen_index):
        data = data[["prop", "subj", "obj", "question", "possible_answers", "s_pop", "o_pop"]].copy()  # Create a copy upfront

        # Only keep relations defined in self.used_relations
        data.loc[:, "keep_row"] = data["prop"].isin(["director", "screenwriter", "producer", "author", "place of birth", "occupation"])
        data = data[data["keep_row"]]
        del data["keep_row"]

        # Remove duplicate questions
        data = data.drop_duplicates(subset=["question"], keep=False)

        # Remove entities with less than or equal to 2 characters
        data.loc[:, "keep_row"] = data["subj"].map(lambda s: isinstance(s, str) and len(s) > 2)
        data = data[data["keep_row"]]
        del data["keep_row"]
        data.loc[:, "keep_row"] = data["obj"].map(lambda s: isinstance(s, str) and len(s) > 2)
        data = data[data["keep_row"]]
        del data["keep_row"]

        # Rename the answer column and only retain a subset
        data.loc[:, "answers"] = data["possible_answers"]
        data_all = data[["prop", "s_pop", "o_pop", "question", "answers"]]

        # Only keep a subset of 932 rows
        data_eval = data_all.sample(932, random_state=0)
        data_train = pd.concat([data_all, data_eval]).drop_duplicates(keep=False) #As training data, we use the subset of PopQA we did not use for evaluation
        return data_train[min_gen_index:max_gen_index]

if __name__ == "__main__": 
    popqa = PopQATrain()
    print(len(popqa.data))
    print(popqa.data)