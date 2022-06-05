import os
from abc import ABC, abstractmethod
from typing import Optional, Tuple, Union

import pandas as pd

from detection_dataset.utils import Dataset


class BaseWriter(ABC):
    def __init__(
        self,
        dataset: Dataset,
        path: str,
        name: str,
        n_images: Optional[int] = None,
        splits: Optional[Tuple[Union[int, float]]] = (0.8, 0.1, 0.1),
    ) -> None:
        """Base class for writing datasets to disk.

        Args:
            dataset: Dataframe containing the dataset to write to disk.
            path: Path to the directory where the dataset will be stored.
            name: Name of the dataset to be created in the "path" directory.
            category_mapping: A dictionary mapping original categories to new categories.
            n_images: Number of images to include in the dataset.
            splits: Tuple containing the proportion of images to include in the train, val and test splits,
                if specified as floats,
                or the number of images to include in the train, val and test splits, if specified as integers.
                Specifying splits as integers is not compatible with specifying n_images, and n_images will be ignored.
                If not specified, the dataset will be split in 80% train, 10% val and 10% test.
        """

        self.data = dataset.data
        self.class_names = dataset.class_names
        self.n_classes = len(dataset.class_names)
        self.path = path
        self.name = name
        self.dataset_dir = os.path.join(self.path, self.name)
        self.n_images = n_images
        self.splits = splits

        # self.data_by_image = self._data_by_image()
        # self.final_data = self._make_final_data()

    # def _data_by_image(self) -> pd.DataFrame:
    #     """Returns the dataframe grouped by image.

    #     Returns:
    #         A dataframe grouped by image
    #     """

    #     data = self.data.groupby(["image_id"])

    #     return pd.DataFrame(
    #         {
    #             "bbox_id": data["bbox_id"].apply(list),
    #             "category_id": data["category_id"].apply(list),
    #             "bbox": data["bbox"].apply(list),
    #             "width": data["width"].first(),
    #             "height": data["height"].first(),
    #             "area": data["area"].apply(list),
    #             "image_name": data["image_name"].first(),
    #             "image_path": data["image_path"].first(),
    #             "split": data["split"].first(),
    #         }
    #     ).reset_index()

    @abstractmethod
    def write(self) -> pd.DataFrame:
        """Writes the dataset to disk."""
