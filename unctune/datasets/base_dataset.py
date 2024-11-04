from abc import ABC, abstractmethod
from pathlib import Path

class BaseDataset(ABC):

    def __init__(self):

        # Create data folder in project root
        self.data_dir = Path(__file__).resolve().parent.parent/ ".." / "data"
        self.data_dir.mkdir(exist_ok=True)
        self.data = self._load_data()

    @abstractmethod
    def _load_data(self):
        pass
