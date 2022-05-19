from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from detection_dataset.utils import Split


class BaseWriter(ABC):
    def __init__(
        self,
        data: pd.DataFrame,
        path: str,
        name: str,
        labels_mapping: dict,
        n_images: Optional[int] = None,
        splits: Optional[Tuple[Union[int, float]]] = (0.8, 0.1, 0.1),
    ) -> None:
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
        self.n_images = n_images
        self.splits = splits
        self.final_data = self._make_final_data()

        if self.labels_mapping:
            self._map_labels()

    def _make_final_data(self) -> None:
        """Creates the final dataset.

        The final dataset takes into account the number of images to include,
        and the splits between train, val and test.

        Returns:
            A dataframe containing the final dataset.

        Raises:
            ValueError: If the values in the splits tuple are not of type float or int.
                All values inside the tuple must be of the same type, either float or int.
        """

        data = self.data.copy()

        if all([isinstance(x, float) for x in self.splits]):
            if self.n_images:
                data = data.sample(frac=self.n_images, random_state=42)

            # Taken from
            # https://stackoverflow.com/questions/38250710/how-to-split-data-into-3-sets-train-validation-and-test
            data_train, data_val, data_test = np.split([int(self.split[0] * len(data)), int(self.split[1] * len(data))])

        elif all([isinstance(x, int) for x in self.splits]):
            if self.n_images:
                print("WARNING: n_images is ignored when splits are specified as integers.")

            data_train = data.loc[data.split == Split.train, :].sample(self.splits[0])
            data_val = data.loc[data.split == Split.val, :].sample(self.splits[1])
            data_test = data.loc[data.split == Split.test, :].sample(self.splits[2])

        else:
            raise ValueError("Splits must be either int or float")

        return pd.concat([data_train, data_val, data_test])

    def _map_labels(self) -> None:
        """Maps the labels to the new labels."""

        self.final_data["category"] = self.final_data["category"].apply(lambda x: [self.labels_mapping[y] for y in x])

    def _data_by_image(self) -> pd.DataFrame:
        """Returns the dataframe grouped by image.

        Returns:
            A dataframe grouped by image
        """

        data = self.final_data.groupby(["filename"])

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

    @abstractmethod
    def write(self) -> pd.DataFrame:
        """Writes the dataset to disk."""
