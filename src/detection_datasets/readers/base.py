from abc import ABC, abstractmethod

import pandas as pd


class BaseReader(ABC):
    """This is the base class that other readers inherit from.

    It enforces the use of the `read()` method in all writers.
    """

    def __init__(self, path: str) -> None:
        """Base class for loading datasets in memory.

        Args:
            path: Path to the dataset
        """

        self.path = path

    @abstractmethod
    def read(self) -> pd.DataFrame:
        """Read a dataset from disk.

        Returns:
            A pandas DataFrame containing the data loaded.
        """
