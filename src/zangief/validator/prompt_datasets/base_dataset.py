from abc import abstractmethod


class BaseDataset:
    def __init__(self):
        pass

    @abstractmethod
    def get_random_record(self) -> str:
        pass
