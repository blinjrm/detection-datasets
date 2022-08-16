from typing import Dict, List, Optional, Tuple, Union

import pandas as pd

from detection_datasets.utils import Dataset, reader_factory, writer_factory
from detection_datasets.utils.constants import DEFAULT_DATASET_DIR
from detection_datasets.utils.enums import Destinations


class Converter:
    """Convert a dataset from one format to another.

    Their are 3 steps to convert a dataset:
    - Read the existing dataset, specifying the format
    - Optionally transform the dataset
    - Write the dataset to a new format, specifying the destination
    """

    def __init__(self) -> None:
        """Initialize the converter."""

        self._dataset = Dataset()
        self.category_mapping = None

    def read(
        self,
        dataset_format: str,
        path: str,
        **kwargs: Dict[str, str],
    ) -> None:
        """Reads the dataset.

        This is a factory method that can read the dataset from different format.

        Args:
            dataset_format: Format of the dataset.
                Currently supported formats:
                - "coco": COCO format
            path: Path to the dataset on the local filesystem.
            **kwargs: Keyword arguments specific to the dataset_format.
        """

        config = {}
        config["path"] = path

        reader = reader_factory.get(dataset_format, **config)
        data = reader.load(**kwargs)
        self._dataset.concat(data)

    def transform(
        self,
        category_mapping: Optional[pd.DataFrame] = None,
        n_images: Optional[int] = None,
        splits: Optional[Tuple[Union[int, float]]] = None,
    ) -> None:
        """Transforms the dataset.

        3 types of transformations can be applied to the dataset:
        - Map existing categories to new categories
        - Reduce the number of images
        - create new (train, val, test) splits

        Args:
            category_mapping (optional): A DataFrame mapping original categories to new categories. Defaults to None.
            n_images (optional): Number of images to include in the dataset. Respects the proportion of images in each
                split. Defaults to None.
            splits: Iterable containing the proportion of images to include in the train, val and test splits.
                The sum of the values in the iterable must be equal to 1. The original splits will be overwritten.
                Defaults to None.
        """

        if category_mapping is not None:
            self.category_mapping = category_mapping
            self._dataset.map_categories(category_mapping)
        if splits:
            self._dataset.split(splits)
        if n_images:
            self._dataset.select(n_images)

    def write(
        self,
        dataset_format: str,
        name: str,
        destinations: Union[str, List[str]],
        **kwargs: Dict[str, str],
    ) -> None:
        """Writes the dataset.

        This is a factory method that can write the dataset:
        - In a given format (e.g. COCO, MMDET, YOLO)
        - To a given destination (e.g. local directory, W&B artifacts)

        Args:
            dataset_format: Format of the dataset.
                Currently supported formats:
                - "yolo": YOLO format
                - "mmdet": MMDET internal format, see:
                    https://mmdetection.readthedocs.io/en/latest/tutorials/customize_dataset.html#reorganize-new-data-format-to-middle-format
            name: Name of the dataset.
            destinations: Where to write the dataset.
                Currently supported destinations:
                - "local_disk": Local disk
                - "wandb": W&B artifacts
            **kwargs: Keyword arguments specific to the dataset_format.
        """

        if not isinstance(destinations, list):
            destinations = [destinations]

        config = {}
        config["dataset"] = self._dataset
        config["name"] = name
        config["destinations"] = destinations

        if Destinations.LOCAL_DISK in destinations:
            if not kwargs.get("path", None):
                raise ValueError("Path must be specified when writing to local filesystem.")
            config["path"] = kwargs["path"]
        else:
            config["path"] = DEFAULT_DATASET_DIR

        writer = writer_factory.get(dataset_format, **config)
        writer.write()

    @property
    def dataset(self) -> Dataset:
        """Access the class dataset.

        Returns:
            A dataset object containing the data.
        """

        return self._dataset

    @property
    def data(self) -> Dataset:
        """Access the class data.

        Returns:
            A pandas DataFrame containing the dataset.
        """

        return self._dataset.data
