from abc import ABC, abstractmethod

import pandas as pd


class BaseReader(ABC):
    def __init__(self, path: str) -> None:
        """Base class for loading datasets in memory.

        Args:
            path: Path to the dataset
        """

        self.path = path
        self._data = self._load()

    @abstractmethod
    def _load(self) -> pd.DataFrame:
        """Load a dataset.

        Returns:
            A pandas DataFrame containing the dataset.
        """

    @property
    def data(self) -> pd.DataFrame:
        """Access the class data.

        Returns:
            A pandas DataFrame containing the dataset.
        """

        return self._data
