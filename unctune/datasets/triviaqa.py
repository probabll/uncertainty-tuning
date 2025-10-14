from pathlib import Path
from unctune.datasets.base_dataset import BaseDataset
from datasets import load_dataset

class TriviaQAEval(BaseDataset):

    def __init__(self):
        self.data = self._load_data("validation")

    def _load_data(self, split):
        dataset = load_dataset("trivia_qa", "unfiltered", split=split)
        dataset = dataset.shuffle(seed=42).select(range(min(932, dataset.num_rows)))
        return dataset

    def __len__(self):
        return len(self.data)
    
    def _has_support(self, ex):
        def any_has(d):
            if isinstance(d, dict):
                for v in d.values():
                    if isinstance(v, list) and any(isinstance(i, dict) and i.get("has_answer") for i in v):
                        return True
            return False
        return any_has(ex.get("search_results")) or any_has(ex.get("entity_pages"))

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
        
if __name__ == "__main__":
    triviaqa = TriviaQAEval()
    print(len(triviaqa))
    print(triviaqa[0])
    print(triviaqa[1])
    print(triviaqa[-1])