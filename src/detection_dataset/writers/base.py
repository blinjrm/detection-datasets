import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from detection_dataset.utils import Dataset, Split


class BaseWriter(ABC):
    def __init__(
        self,
        dataset: Dataset,
        path: str,
        name: str,
        labels_mapping: Optional[dict] = None,
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

        self.data = dataset.data
        self.class_names = dataset.categories
        self.n_classes = len(dataset.categories)
        self.path = path
        self.name = name
        self.dataset_dir = os.path.join(self.path, self.name)
        self.labels_mapping = labels_mapping
        self.n_images = n_images
        self.splits = splits
        if self.labels_mapping:
            self._map_labels()
        self.data_by_image = self._data_by_image()
        self.final_data = self._make_final_data()

    def _map_labels(self) -> None:
        """Maps the labels to the new labels."""

        self.data["category"] = self.data["category"].apply(lambda x: [self.labels_mapping[y] for y in x])

    def _data_by_image(self) -> pd.DataFrame:
        """Returns the dataframe grouped by image.

        Returns:
            A dataframe grouped by image
        """

        data = self.data.groupby(["image_id"])

        return pd.DataFrame(
            {
                "bbox_id": data["bbox_id"].apply(list),
                "category_id": data["category_id"].apply(list),
                "bbox": data["bbox"].apply(list),
                "width": data["width"].first(),
                "height": data["height"].first(),
                "area": data["area"].apply(list),
                "image_name": data["image_name"].first(),
                "image_path": data["image_path"].first(),
                "split": data["split"].first(),
            }
        ).reset_index()

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

        data = self.data_by_image.copy()

        if all([isinstance(x, float) for x in self.splits]):
            if self.n_images:
                data = data.sample(n=self.n_images, random_state=42)

            n_train = int(self.splits[0] * len(data))
            n_val = int((self.splits[0] + self.splits[1]) * len(data))
            data_train, data_val, data_test = np.split(data, [n_train, n_val])
            data_train["split"] = Split.train.value
            data_val["split"] = Split.val.value
            data_test["split"] = Split.test.value

        elif all([isinstance(x, int) for x in self.splits]):
            if self.n_images:
                print("WARNING: n_images is ignored when splits are specified as integers.")

            data_train = data.loc[data.split == Split.train.value, :].sample(self.splits[0])
            data_val = data.loc[data.split == Split.val.value, :].sample(self.splits[1])
            data_test = data.loc[data.split == Split.test.value, :].sample(self.splits[2])

        else:
            raise ValueError("Splits must be either int or float")

        return pd.concat([data_train, data_val, data_test])

    @abstractmethod
    def write(self) -> pd.DataFrame:
        """Writes the dataset to disk."""
