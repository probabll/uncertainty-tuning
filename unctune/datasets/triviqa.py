from pathlib import Path
from unctune.datasets.base_dataset import BaseDataset
from datasets import load_dataset

class TriviaQAEval(BaseDataset):

    def __init__(self):
        self.data = self._load_data("validation")

    def _load_data(self, split):
        return load_dataset("trivia_qa", "unfiltered", split=split)

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """ Returns the questions and answers as a tuple at the given index or slice. """
        if isinstance(idx, slice):
            question_ids = self.data["question_id"][idx]
            questions = self.data["question"][idx]
            answers = self.data["answer"][idx]
            
            # Build list of tuples for the slice
            result = []
            for i in range(len(question_ids)):
                answer_list = [answers[i]["value"]] + answers[i]["aliases"]
                result.append((question_ids[i], questions[i], answer_list))
            return result
        else:
            return (self.data["question_id"][idx], \
                     self.data["question"][idx], \
                    [self.data["answer"][idx]["value"]] + self.data["answer"][idx]["aliases"])