from abc import ABC, abstractmethod

from detection_dataset.utils import Dataset


class BaseReader(ABC):
    def __init__(self, path: str) -> None:
        """Base class for loading datasets in memory.

        Args:
            path: Path to the dataset
        """

        self.path = path

    @abstractmethod
    def load(self) -> Dataset:
        """Load a dataset.

        Returns:
            A Dataset instance containing the data loaded.
        """
