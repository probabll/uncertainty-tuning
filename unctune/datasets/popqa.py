from unctune.datasets.base_dataset import BaseDataset
import os
import urllib.request
import ast
import pandas as pd

class PopQA(BaseDataset):

    FILE_LOC = "https://raw.githubusercontent.com/AlexTMallen/adaptive-retrieval/main/data/popQA.tsv"
    FILENAME = "popQA.tsv"

    def __init__(self):
        """
        PopQA dataset preprocessed as in https://arxiv.org/abs/2405.16908.
        """
        super().__init__()

    def _load_data(self):
        data = self._load_or_download_data_file()
        data = self._preprocess(data)
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

    def _preprocess(self, data):
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
        data["answers"] = data["answers"].apply(ast.literal_eval)
        data = data[["prop", "s_pop", "o_pop", "question", "answers"]]

        # Only keep a subset of 932 rows
        data = data.sample(932, random_state=0)
        return data
    
    def __getitem__(self, idx):
        """ Returns the questions and answers as a tuple at the given index or slice. """
        if isinstance(idx, slice):
            return self.data.iloc[idx].index.values, self.data.iloc[idx]["question"].values, self.data.iloc[idx]["answers"].values
        else:
            return int(self.data.iloc[idx].name), self.data.iloc[idx]["question"], self.data.iloc[idx]["answers"]

if __name__ == "__main__": 
    popqa = PopQA()
    print(len(popqa))
    print(popqa[0])