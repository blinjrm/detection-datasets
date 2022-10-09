from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Sequence, Value, load_dataset
from PIL import Image as PILImage

from detection_datasets.bbox import Bbox
from detection_datasets.utils.enums import Split
from detection_datasets.utils.factories import reader_factory, writer_factory
from detection_datasets.utils.hub import ORGANISATION, available_in_hub
from detection_datasets.utils.visualization import show_image_bbox

CACHE_DIR = ".detection-datasets"
SPLITS = ["train", "val", "test"]


class DetectionDataset:

    COLUMNS = [
        "image_id",
        "image_path",
        "width",
        "height",
        "split",
        "bbox_id",
        "category_id",
        "category",
        "bbox",
        "area",
    ]

    _data = pd.DataFrame(columns=COLUMNS).set_index(["image_id", "bbox_id"])

    def __init__(self, data: pd.DataFrame = None) -> None:
        """Initialize the dataset.

        Don't call the constructr directly, use `from_hub()` or `from_disk()` methods instead.

        Args:
            data: The data used to initialize the dataset.
                Defaults to None.
        """

        self._format = "init"

        if data is not None:
            self._concat(data)

    @property
    def data(self) -> pd.DataFrame:
        """Getter for the data, with annotations grouped by images.

        Returns:
            The data contained in the dataset as a Pandas DataFrame.
        """

        return self.get_data()

    def get_data(self, index: str = "image") -> pd.DataFrame:
        """Getter for the data, with the possibility to specify the format.

        Args:
            index: The desired format of the data.
                Can be either "image" or "bbox".
                Defaults to "image".

        Returns:
            The data contained in the dataset as a Pandas DataFrame in the specified format.
        """

        data = self.set_format(index=index)

        return data

    @property
    def format(self) -> str:
        """Getter for the current format of the data, which can either be "image" or "bbox".

        Returns:
            The current format of the data.
        """

        return self._format

    def _concat(self, other_data: pd.DataFrame, other_data_format: str = "bbox") -> None:
        """Concatenate the existing data with new data.

        This allows to load multiple datasets, potentially from different sources (disk & hub) into one larger dataset.

        Args:
            other_data: The data being added to the dataset.
            other_data_format: The format of the new data.
                Defaults to "bbox".
        """

        self.set_format(index=other_data_format)
        self._data = pd.concat([self._data.reset_index()[self.COLUMNS], other_data[self.COLUMNS]])
        self.set_format(index="image")

    def from_hub(self, dataset_name: str, repo_name: str = ORGANISATION, in_memory: bool = False) -> DetectionDataset:
        """Load a dataset from the Hugging Face Hub.

        Args:
            dataset_name: name of the dataset, without the organisation's prefix.
            repo_name: name of the Hugging Face profile or organisation where the dataset is stored.
                Defaults to "detection-datasets".
            in_memory: whether to keep the images in memory.
                Set to "True" to keep the image in memory in the Pandas DataFrame, if the dataset is small.
                Set to "False" if the system runs out of memory, then the images will be downloaded
                and only the path to these images will be saved in the data.
                Defaults to False.

        Returns:
            The DetectionDataset instance. This allows for method cascading.
        """

        if dataset_name not in available_in_hub(repo_name=repo_name):
            raise ValueError(
                f"""{dataset_name} is not available on the Hub.
            Use `DetectionDataset.available_in_hub() to get the list of available datasets."""
            )

        path = "/".join([repo_name, dataset_name])
        ds = load_dataset(path=path)
        categories = ds[list(ds.keys())[0]].features["objects"].feature["category"]

        if not in_memory:
            DOWNLOAD_PATH = self._get_temp_dir()

            def download_images(row):
                file_path = "".join([DOWNLOAD_PATH.as_posix(), "/", str(row["image_id"]), ".jpg"])
                row["image"].save(file_path)
                row["image_path"] = file_path
                return row

            ds = ds.map(
                download_images,
                remove_columns="image",
                load_from_cache_file=False,
                desc="Extracting images from parquet",
            )

        df_splits = []
        for key in ds.keys():
            df_split = ds[key].to_pandas()
            df_split["split"] = key

            df_splits.append(df_split)

        df = pd.concat(df_splits)
        df = df.reset_index(drop=True)
        objects = pd.json_normalize(df["objects"])
        data = df.join(objects)

        if "image_path" not in data.columns:
            data["image_path"] = [x["bytes"] for x in data.loc[:, "image"]]

        data = data.drop(columns=["objects", "image"], errors="ignore")
        data["category_id"] = data.loc[:, "category"]
        data["category"] = [[categories.int2str(int(x)) for x in row["category"]] for _, row in data.iterrows()]

        data = data.explode(["bbox_id", "category_id", "category", "bbox", "area"])
        data["bbox"] = [Bbox.from_voc(row.bbox, row.width, row.height, row.bbox_id) for _, row in data.iterrows()]

        self._concat(other_data=data)

        return self

    def from_disk(self, dataset_format: str, path: str, **kwargs) -> DetectionDataset:
        """Load a dataset from disk.

        This is a factory method that can read the dataset from different formats,
        when the dataset is already in a local directory.

        Args:
            dataset_format: Format of the dataset.
                Currently supported values and formats:
                - "coco": COCO format
            path: Path to the dataset on the local filesystem.
            **kwargs: Keyword arguments specific to the dataset_format.

        Returns:
            The DetectionDataset instance. This allows for method cascading.

        Example:
            ```Python
            config = {
                "dataset_format": "coco",
                "path": "PATH/TO/DATASET",
                "splits": {
                    "train": (train_annotations.json, 'train'),
                    "val": (test_annotations.json, 'test'),
                },
            }
            dd = DetectionDataset().from_disk(**config)
            ```
        """

        reader = reader_factory.get(dataset_format=dataset_format.lower(), path=path, **kwargs)
        data = reader.read()

        self._concat(other_data=data)

        return self

    def to_hub(self, dataset_name: str, repo_name: str, **kwargs) -> DetectionDataset:
        """Push the dataset to the hub as a Parquet dataset.

        This method wraps Hugging Face's DatasetDict.push_to_hub() method.

        The dataset is pushed as a DatasetDict, meaning the each split (train, val, test), if present,
        will be a separate Dataset instance inside this DatasetDict.

        Args:
            dataset_name: name of the dataset inside the user/organisation's repository.
            repo_name: user of organisation to push the dataset to.

        Returns:
            The DetectionDataset instance. This allows for method cascading.
        """

        repo_id = "/".join([repo_name, dataset_name])

        hf_dataset_dict = self._get_hf_dataset()
        hf_dataset_dict.push_to_hub(repo_id=repo_id, **kwargs)
        print(f"The dataset was uploaded to https://huggingface.co/datasets/{repo_id}")

        return self

    def _get_hf_dataset(self) -> DatasetDict:
        """Get the data formatted as an Hugging Face DatasetDict instance.

        The DatasetDict contains a Dataset for each split present in the data.
        All methods and properties of the DatasetDict can then be used.

        Returns:
            Data formatted as an Hugging Face DatasetDict instance
        """

        data = self.set_format(index="image").copy().reset_index()
        data["image_id"] = data.loc[:, "image_id"].astype(int)
        data["bbox_id"] = [[int(bbox_id) for bbox_id in bbox_ids] for bbox_ids in data.bbox_id]
        data["bbox"] = [[bbox.to_voc() for bbox in bboxes] for bboxes in data.bbox]

        hf_dataset_dict = DatasetDict()

        for split in self.splits:
            split_data = data[data.split == split]
            images_data = []

            for _, row in split_data.iterrows():
                objects = {}
                objects["bbox_id"] = row["bbox_id"]
                objects["category"] = row["category"]
                objects["bbox"] = row["bbox"]
                objects["area"] = row["area"]

                image = {}
                image["image_id"] = row["image_id"]
                image["image"] = row["image_path"]
                image["width"] = row["width"]
                image["height"] = row["height"]
                image["objects"] = objects

                images_data.append(image)

            df = pd.DataFrame.from_dict(images_data)

            features = self._get_hf_features()

            ds = Dataset.from_pandas(df=df, features=features, split=split)
            hf_dataset_dict[split] = ds

        return hf_dataset_dict

    def to_disk(self, dataset_format: str, name: str, absolute_path: str) -> DetectionDataset:
        """Write the dataset to disk.

        This is a factory method that can write the dataset to disk in the selected format (e.g. COCO, MMDET, YOLO)

        Args:
            dataset_format: Format of the dataset.
                Currently supported formats:
                - "yolo": YOLO format
                - "mmdet": MMDET internal format, see:
                    https://mmdetection.readthedocs.io/en/latest/tutorials/customize_dataset.html#reorganize-new-data-format-to-middle-format
                - "coco": COCO format
            name: Name of the dataset to be created in the "path" directory.
            absolute_path: Absolute path to the directory where the dataset will be created.
            **kwargs: Keyword arguments specific to the dataset_format.

        Returns:
            The DetectionDataset instance. This allows for method cascading.
        """

        writer = writer_factory.get(dataset_format=dataset_format.lower(), dataset=self, name=name, path=absolute_path)
        writer.write()

        return self

    @staticmethod
    def _get_temp_dir() -> str:
        """Get the path for the temp directory, create it if needed.

        Returns:
            The path to the library's temp directory.
        """

        DOWNLOAD_PATH = Path.home() / CACHE_DIR
        DOWNLOAD_PATH.mkdir(parents=True, exist_ok=True)

        return DOWNLOAD_PATH

    def _get_hf_features(self) -> Features:
        """Get the feature types for the Hugging Face dataset.

        Returns:
            Features for the Hugging Face dataset.
        """

        return Features(
            {
                "image_id": Value(dtype="int64"),
                "image": Image(decode=True),
                "width": Value(dtype="int64"),
                "height": Value(dtype="int64"),
                "objects": Sequence(
                    {
                        "bbox_id": Value(dtype="int64"),
                        "category": ClassLabel(names=self.category_names),
                        "bbox": Sequence(feature=Value(dtype="float64"), length=4),
                        "area": Value(dtype="float64"),
                    }
                ),
            }
        )

    def set_format(self, index: str) -> pd.DataFrame:
        """Set the format of the data.

        The data contained in the dataset can either have:
        - One row per image, with the annotations grouped as a list
        - One row per annotation, with each image appearing on multiple rows

        Args:
            index: How to organise the data, can be "image" or "bbox".

        Raises:
            ValueError: If the specified format is unknown.

        Returns:
            Data contained in the dataset.
        """

        if index == self._format:
            pass
        elif index == "image":
            self._data_by_image()
        elif index == "bbox":
            self._data_by_bbox()
        else:
            raise ValueError(f"The index must be either 'image' or 'bbox', not '{index}'.")

        return self._data.copy()

    def _data_by_image(self) -> pd.DataFrame:
        """Returns the data grouped by image.

        Returns:
            A DataFrame grouped by image, meaning that each may contain data related to multiple bboxes.
        """

        data = self._data.reset_index().groupby("image_id")
        self._data = pd.DataFrame(
            {
                "image_path": data["image_path"].first(),
                "width": data["width"].first(),
                "height": data["height"].first(),
                "split": data["split"].first(),
                "bbox_id": data["bbox_id"].apply(list),
                "bbox": data["bbox"].apply(list),
                "category_id": data["category_id"].apply(list),
                "category": data["category"].apply(list),
                "area": data["area"].apply(list),
            }
        )

        self._format = "image"

    def _data_by_bbox(self) -> pd.DataFrame:
        """Converts a DataFrame arranged by image to a DataFrame arranged by bbox.

        This method reverses the effect of calling self._data_by_image().

        Args:
            data: Dataframe to explode.

        Returns:
            A DataFrame arranged by bbox instead of images.
        """

        self._data = (
            self._data.reset_index()
            .explode(["bbox_id", "category_id", "category", "bbox", "area"])
            .set_index(["image_id", "bbox_id"])
        )

        self._format = "bbox"

    def select(self, n_images: int, seed: int = 42) -> DetectionDataset:
        """Limits the number of images to n_images.

        Args:
            n_images: Number of images to include in the dataset.
                The original proportion of images between splits will be respected.
            seed: Random seed.

        Returns:
            The DetectionDataset instance. This allows for method cascading.
        """

        data_by_image = self.set_format(index="image")

        if self.n_images > len(data_by_image):
            raise ValueError(
                "The number of images to include in the dataset is greater than the number of existing images."
            )

        split_data = []

        for split in Split:
            sample_size = int(n_images * self.split_proportions[split.value])
            split_data.append(
                data_by_image.loc[data_by_image.split == split.value, :].sample(n=sample_size, random_state=seed)
            )

        self._data = pd.concat(split_data)

        return self

    def shuffle(self, seed: int = 42) -> DetectionDataset:
        """Shuffles the dataset.

        Args:
            seed: Random seed.

        Returns:
            The DetectionDataset instance. This allows for method cascading.
        """

        data_by_image = self.set_format(index="image")

        split_data = []

        for split in Split:
            split_data.append(
                data_by_image.loc[data_by_image.split == split.value, :].sample(frac=1, random_state=seed)
            )

        self._data = pd.concat(split_data)

        return self

    def split(self, splits: Iterable[float]) -> DetectionDataset:
        """Splits the dataset into train, val and test.

        Args:
            splits: Iterable containing the proportion of images to include in the train, val and test splits.
                The sum of the values in the iterable must be equal to 1.
                The original splits will be overwritten.

        Returns:
            The DetectionDataset instance. This allows for method cascading.
        """

        if len(splits) != 3:
            raise ValueError("The splits must contain 3 elements.")

        if sum(splits) != 1:
            raise ValueError(f"The sum of the proportion for each split must be equal to 1, here it is: {sum(splits)}.")

        data_by_image = self.set_format(index="image")

        n_train = int(splits[0] * len(data_by_image))
        n_val = int(n_train + splits[1] * len(data_by_image))
        n_test = int(n_val + splits[2] * len(data_by_image))

        data_by_image = data_by_image.sample(frac=1, random_state=42)
        data_train, data_val, data_test, _ = np.split(data_by_image, [n_train, n_val, n_test])
        data_train["split"] = Split.TRAIN.value
        data_val["split"] = Split.VAL.value
        data_test["split"] = Split.TEST.value

        self._data = pd.concat([data_train, data_val, data_test])

        return self

    def map_categories(self, mapping: dict[str, str]) -> DetectionDataset:
        """Maps the categories to the new categories.

        The new categoy names replace the existing ones.
        Annotations with categories not present in the mapping are dropped.
        The new category_ids correspond the the rank of the new categories in alphabetical order.

        Args:
            mapping: A dictionnary mapping original categories to new categories.

        Returns:
            The DetectionDataset instance. This allows for method cascading.
        """

        data = self.set_format(index="bbox").reset_index()
        data["category"] = data.loc[:, "category"].map(mapping)
        data = data[~data.category.isna()]

        categories = sorted(data.category.unique())
        data["category_id"] = data.loc[:, "category"].apply(lambda cat: categories.index(cat))

        self._data = data.set_index(["image_id", "bbox_id"])

        return self

    def show(self, image_id: int = None) -> PILImage:
        """Show the image with bounding boxes and labels.

        Args:
            image_id: Id of the image.
                If not provided, a random image is selected.
                Defaults to None.

        Returns:
            Image with bounding boxes and labels.
        """

        data = self.set_format(index="bbox")

        if image_id is None:
            index = np.random.randint(0, len(data))
            image_id = data.reset_index().iloc[index]["image_id"]

        rows = data.loc[image_id]

        image = show_image_bbox(rows=rows)

        print(f"Showing image id {image_id}.")

        return image

    @property
    def n_images(self) -> int:
        """Returns the number of images in the dataset.

        Returns:
            The number of images in the dataset.
        """

        data = self.set_format(index="image")

        return len(data)

    @property
    def n_bbox(self) -> int:
        """Returns the number of images in the dataset.

        Returns:
            The number of images in the dataset.
        """

        data = self.set_format(index="bbox")

        return len(data)

    @property
    def splits(self) -> list[str]:
        """Returns the splits of the dataset.

        Returns:
            The splits present in the dataset.
        """

        return self._data.split.unique().tolist()

    @property
    def split_proportions(self) -> pd.DataFrame:
        """Returns the proportion of images in the train, val and test splits.

        Returns:
            The proportion of images in the train, val and test splits.
        """

        data = self.set_format(index="image")

        return pd.DataFrame({s.value: [len(data[data.split == s.value]) / len(data)] for s in Split})

    @property
    def categories(self) -> pd.DataFrame:
        """Creates a DataFrame containing the categories found in the data with their id.

        Returns:
            A dataframe containing the categories with the category_id as index.
        """

        data = self.set_format(index="bbox")

        return (
            data.loc[:, ["category_id", "category"]]
            .drop_duplicates()
            .astype({"category_id": int, "category": "object"})
            .sort_values("category_id")
            .set_index("category_id")
        )

    @property
    def category_names(self) -> list[str]:
        """Returns the categories names.

        Returns:
            The categories names.
        """

        return list(self.categories["category"].unique())

    @property
    def n_categories(self) -> int:
        """Returns the number of categories.

        Returns:
            The number of categories.
        """

        return self.categories["category"].nunique()
