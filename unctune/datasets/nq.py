from datasets import load_dataset
from unctune.datasets.base_dataset import BaseDataset
import os
import urllib.request
import pandas as pd

class NQ(BaseDataset):

    def __init__(self):
        """
        NQ dataset preprocessed as in https://arxiv.org/abs/2405.16908.
        """
        super().__init__()

    def _load_data(self):
        data = load_dataset("sewon/ambig_qa", "light")
        data = self._preprocess(data)
        return data

    def _preprocess(self, data):
        data = data["validation"].to_pandas()
        data["type"] = data["annotations"].map(lambda s: s["type"][0])
        data = data[data["type"] == "singleAnswer"]
        data["answer"] = data["annotations"].map(lambda s: s["answer"][0])
        data = data[["id", "question", "answer"]]
        return data

if __name__ == "__main__":
    nq = NQ()
    print(len(nq.data))
    print(nq.data)