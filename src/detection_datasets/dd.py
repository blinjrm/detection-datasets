from typing import List, Tuple, Union

import numpy as np
import pandas as pd
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, Sequence, Value, load_dataset
from huggingface_hub import HfApi

from detection_datasets.utils import reader_factory, writer_factory
from detection_datasets.utils.enums import Split

api = HfApi()

ORGANISATION = "detection-datasets"


class DetectionDataset:

    COLUMNS = [
        "image_id",
        "image",
        "width",
        "height",
        "split",
        "bbox_id",
        "category_id",
        "category",
        # "supercategory",
        "bbox",
        "area",
        # "image_name",
        # "image_path",
    ]

    _data = pd.DataFrame(columns=COLUMNS).set_index(["image_id", "bbox_id"])

    def __init__(self, data: pd.DataFrame = None) -> None:
        """Initialize the dataset."""

        self._format = "bbox"

        if data is not None:
            data = data[data.columns.intersection(self.COLUMNS)]
            self.concat(data)

    def concat(self, other_data: pd.DataFrame) -> pd.DataFrame:
        """Concatenate the existing data with new data."""

        self.set_format(index="bbox")
        self._data = pd.concat([self.data, other_data])
        # return self._data

    @property
    def data(self, index: str = None) -> pd.DataFrame:
        if index:
            self.set_format(index=index)

        return self._data

    def from_hub(self, name: str) -> None:
        """Load a dataset from the Hugging Face Hub.

        Currently only datasets from the 'detection-datasets' organisation can be loaded.

        Args:
            name: name of the dataset, without the organisation's prefix.

        Returns:
            A DetectionDataset instance containing the loaded data.
        """

        if name not in self.available_in_hub:
            raise ValueError(
                f"""{name} is not available on the Hub.
            Use `DetectionDataset.available_in_hub() to get the list of available datasets."""
            )

        path = "/".join([ORGANISATION, name])
        ds = load_dataset(path=path)

        df_splits = []
        for key in ds.keys():
            df_split = ds[key].to_pandas()
            df_split["split"] = key
            df_splits.append(df_split)

        df = pd.concat(df_splits)

        objects = pd.json_normalize(df["objects"])
        objects = objects.rename(columns={"id": "bbox_id"})

        data = df.join(objects)
        data = data.drop(columns=["objects"])

        self.concat(other_data=data)

    @property
    @staticmethod
    def available_in_hub() -> List[str]:
        """List the datasets available in the Hugging Face Hub.

        Returns:
            List of names of datasets registered in the Hugging Face Hub, under the 'detection-datasets' organisation.
        """

        datasets = api.list_datasets(author=ORGANISATION)
        return [dataset.id.split("/")[-1] for dataset in datasets]

    @classmethod
    def from_disk(self, dataset_format: str, path: str, **kwargs) -> None:
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
            A DetectionDataset instance containing the loaded data.
        """

        reader = reader_factory.get(dataset_format=dataset_format, path=path, **kwargs)
        data = reader.load()

        self.concat(other_data=data)

    def to_hub(self, dataset_name: str, repo_name: str = ORGANISATION, **kwargs) -> None:
        """Push the dataset to the hub as a Parquet dataset.

        This method wraps Hugging Face's DatasetDict.push_to_hub() method, check here for reference:
        https://huggingface.co/docs/datasets/package_reference/main_classes#datasets.DatasetDict.push_to_hub

        The dataset is pushed as a DatasetDict, meaning the each split (train, val, test), if present,
        will be a separate Dataset instance inside this DatasetDict.

        Args:
            dataset_name: name of the dataset inside the user/organisation's repository.
            repo_name: user of organisation to push the dataset to.
        """

        repo_id = "/".join([repo_name, dataset_name])

        hf_dataset_dict = DatasetDict()
        data = self.data_by_image

        for split in data.unique("split"):
            split_data = data[data.split == split]
            images_data = []

            for _, row in split_data.iterrows():
                objects = {}
                objects["id"] = row["bbox_id"]
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

            features = Features(
                {
                    "image_id": Value(dtype="int64"),
                    "image": Image(decode=True),
                    "width": Value(dtype="int64"),
                    "height": Value(dtype="int64"),
                    "objects": Sequence(
                        {
                            "area": Value(dtype="int64"),
                            "bbox": Sequence(feature=Value(dtype="float64"), length=4),
                            "category": ClassLabel(names=self.category_names),
                            "id": Value(dtype="int64"),
                        }
                    ),
                }
            )

            hf = Dataset.from_pandas(df=df, features=features, split=split)
            hf_dataset_dict[split] = hf

        hf_dataset_dict.push_to_hub(repo_id=repo_id, **kwargs)
        return hf_dataset_dict

    def to_disk(self, dataset_format: str, name: str, path: str) -> None:
        """Write the dataset to disk.

        This is a factory method that can write the dataset to disk in the selected format (e.g. COCO, MMDET, YOLO)

        Args:
            dataset_format: Format of the dataset.
                Currently supported formats:
                - "yolo": YOLO format
                - "mmdet": MMDET internal format, see:
                    https://mmdetection.readthedocs.io/en/latest/tutorials/customize_dataset.html#reorganize-new-data-format-to-middle-format
            name: Name of the dataset to be created in the "path" directory.
            path: Path to the directory where the dataset will be created.
            **kwargs: Keyword arguments specific to the dataset_format.
        """

        writer = writer_factory.get(dataset_format=dataset_format, dataset=self, name=name, path=path)
        writer.write()

    def set_format(self, index: str) -> pd.DataFrame:
        if index == self._format:
            pass
        elif index == "image":
            self._data_by_image()
        elif index == "bbox":
            self._data_by_bbox()
        else:
            raise ValueError(f"The index must be either 'image' or 'bbox', not '{index}'.")

        return self._data

    @property
    def _data_by_image(self) -> pd.DataFrame:
        """Returns the data grouped by image.

        Returns:
            A DataFrame grouped by image, meaning that each may contain data related to multiple bboxes.
        """

        data = self.data.reset_index().groupby("image_id")
        self._data = pd.DataFrame(
            {
                "bbox_id": data["bbox_id"].apply(list),
                "category_id": data["category_id"].apply(list),
                "category": data["category"].apply(list),
                "supercategory": data["supercategory"].apply(list),
                "bbox": data["bbox"].apply(list),
                "width": data["width"].first(),
                "height": data["height"].first(),
                "area": data["area"].apply(list),
                "image_name": data["image_name"].first(),
                "image_path": data["image_path"].first(),
                "split": data["split"].first(),
            }
        ).reset_index()

        self._format = "image"

    def _data_by_bbox(
        self,
    ) -> pd.DataFrame:
        """Converts a DataFrame arranged by image to a DataFrame arranged by bbox.

        This method reverses the effect of calling self._data_by_image().

        Args:
            data: Dataframe to explode.

        Returns:
            A DataFrame arranged by bbox instead of images.
        """

        self._data = self._data.explode(
            ["bbox_id", "category_id", "category", "supercategory", "area", "bbox"]
        ).set_index(["image_id", "bbox_id"])

        self._format = "bbox"

    def select(self, n_images: int) -> None:
        """Limits the number of images to n_images.

        Args:
            n_images: Number of images to include in the dataset.
                The original proportion of images between splits will be respected.
        """

        data_by_image = self.set_format(index="image")

        if self.n_images > len(data_by_image):
            raise ValueError(
                "The number of images to include in the dataset is greater than the number of existing images."
            )

        split_data = []

        for split in Split:
            if split.value in data_by_image.unique("split"):
                sample_size = int(n_images * self.split_proportions[split.value])
                split_data.append(
                    data_by_image.loc[data_by_image.split == split.value, :]
                    .sample(frac=sample_size, random_state=42)
                    .reset_index(drop=True)
                )

        self._data = pd.concat(split_data)

    def shuffle(self, seed: int = 42) -> None:
        """Shuffles the dataset.

        Args:
            seed: Random seed.
        """

        data_by_image = self.set_format(index="image")

        split_data = []

        for split in Split:
            split_data.append(
                data_by_image.loc[data_by_image.split == split.value, :]
                .sample(frac=1, random_state=seed)
                .reset_index(drop=True)
            )

        self._data = pd.concat(split_data)

    def split(self, splits: Tuple[Union[int, float]]) -> None:
        """Splits the dataset into train, val and test.

        Args:
            splits: Iterable containing the proportion of images to include in the train, val and test splits.
                The sum of the values in the iterable must be equal to 1. The original splits will be overwritten.
        """

        if len(splits) != 3:
            raise ValueError(f"The splits must be a tuple of 3 elements, here it is: {splits}.")

        if sum(splits) != 1:
            raise ValueError(f"The sum of the proportion for each split must be equal to 1, here it is: {sum(splits)}.")

        if not all([isinstance(x, float) for x in splits]):
            raise TypeError("Splits must be either int or float, here it is: {}.".format(*[type(s) for s in splits]))

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

    @property
    def n_images(self) -> int:
        """Returns the number of images in the dataset.

        Returns:
            The number of images in the dataset.
        """

        self.set_format(index="image")

        return len(self._data)

    @property
    def splits(self) -> List[str]:
        """Returns the splits of the dataset.

        Returns:
            The splits present in the dataset.
        """

        return self.data.split.unique().tolist()

    @property
    def split_proportions(self) -> Tuple[float, float, float]:
        """Returns the proportion of images in the train, val and test splits.

        Returns:
            The proportion of images in the train, val and test splits.
        """

        data = self._data

        return pd.DataFrame({s.value: [len(data[data.split == s.value]) / len(data)] for s in Split})

    @property
    def categories(self) -> None:
        """Creates a DataFrame containing the categories found in the data with their id."""

        return (
            # self.data.loc[:, ["category_id", "category", "supercategory"]]
            self.data.loc[:, ["category_id", "category"]]
            .drop_duplicates()
            .sort_values("category_id")
            .reset_index(drop=True)
        )

    @property
    def category_names(self) -> List[str]:
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
