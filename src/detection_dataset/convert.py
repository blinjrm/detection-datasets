from typing import Dict, Optional, Tuple, Union

import pandas as pd

from detection_dataset.utils import Dataset, reader_factory, writer_factory


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
        self.n_images = len(self._dataset.data)
        # self.splits = self._dataset.read_splits()

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

        self.reader = reader_factory.get(dataset_format, **config)
        dataset = self.reader.load(**kwargs)
        self._dataset.concat(dataset)
        # self.splits = self._dataset.read_splits()

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
            category_mapping (optional): A dictionary mapping original categories to new categories. Defaults to None.
            n_images (optional): Number of images to include in the dataset. Defaults to None.
            splits (optional): Tuple containing the proportion of images to include in the train, val and test splits,
                if specified as floats, or the number of images to include in the  splits, if specified as integers.
                Specifying splits as integers is not compatible with specifying n_images, and n_images will be ignored.
                If not specified, the original splits from the dataset will be used. Defaults to None.
        """

        if category_mapping is not None:
            self.category_mapping = category_mapping
            self._dataset.map_categories(category_mapping)
        if n_images:
            self.n_images = n_images
            self._dataset.limit_images(n_images)
        if splits:
            self.splits = splits
            self._dataset.split(splits)

    def write(
        self,
        dataset_format: str,
        name: str,
        destination: str,
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
            destination: Where to write the dataset.
                Currently supported destinations:
                - "local": Local directory
                - "wandb": W&B artifacts
            **kwargs: Keyword arguments specific to the dataset_format.
        """

        # if destination == "local" and path is None:
        #     raise ValueError("Path must be specified when writing to local filesystem.")

        config = {}
        config["dataset"] = self._dataset
        config["dataset_format"] = dataset_format
        config["name"] = name

        self.writer = writer_factory.get(dataset_format, **config)
        self.writer.write(destination, **kwargs)

    @property
    def dataset(self) -> pd.DataFrame:
        """Access the class data.

        Returns:
            A pandas DataFrame containing the dataset.
        """

        return self._dataset
