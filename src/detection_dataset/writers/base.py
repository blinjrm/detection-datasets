import os
from abc import ABC, abstractmethod

import pandas as pd

from detection_dataset.utils import Dataset


class BaseWriter(ABC):
    def __init__(
        self,
        dataset: Dataset,
        name: str,
        path: str,
    ) -> None:
        """Base class for writing datasets to disk.

        Args:
            dataset: Dataframe containing the dataset to write to disk.
            name: Name of the dataset to be created in the "path" directory.
            path: Path to the directory where the dataset will be created.
        """

        self.data = dataset.data_by_image
        self.name = name
        self.path = path
        self.dataset_dir = os.path.join(self.path, self.name)
        self.class_names = dataset.category_names
        self.n_classes = dataset.n_categories

    @abstractmethod
    def write(self) -> pd.DataFrame:
        """Writes the dataset to disk."""
