from abc import ABC, abstractmethod

import pandas as pd


class BaseWriter(ABC):
    def __init__(self, data: pd.DataFrame, path: str, name: str, labels_mapping: dict) -> None:
        """Base class for writing datasets to disk.

        Args:
            data: Dataframe containing the dataset to write to disk.
            path: Path to the directory where the dataset will be stored.
            name: Name of the dataset.
            labels_mapping: A dictionary mapping original labels to new labels.
        """

        self.data = data
        self.path = path
        self.name = name
        self.labels_mapping = labels_mapping

    @abstractmethod
    def _write(self) -> pd.DataFrame:
        """Writes the dataset to disk."""

    def _data_by_image(self) -> pd.DataFrame:
        """Returns the dataframe grouped by image.

        Returns:
            A dataframe grouped by image
        """

        data = self.data.groupby(["filename"])

        return pd.DataFrame(
            {
                "width": data["width"].first(),
                "height": data["height"].first(),
                "category": data["category"].apply(list),
                "attributes": data["attributes"].apply(list),
                "area": data["area"].apply(list),
                "bbox": data["bbox"].apply(list),
            }
        ).reset_index()
