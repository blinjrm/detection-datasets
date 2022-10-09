import json
import os
from typing import Dict, Tuple

import pandas as pd

from detection_datasets.bbox import Bbox
from detection_datasets.readers import BaseReader


class CocoReader(BaseReader):
    """Read a dataset from disk in the COCO format.

    You don't need to use the reader directly, use the `from_disk(dataset_format="coco", *args, **kwargs)` method
    of the DetectionDataset class instead.
    This is equivalent to first calling the reader, and then initializing the DetectionDataset instance
    with the data returned by the reader.

    Example:
        ```Python
        config = {
            "path": "PATH/TO/DATASET",
            "splits": {
                "train": (train_annotations.json, 'train'),
                "val": (test_annotations.json, 'test'),
            },
        }
        reader = CocoReader(**config)
        data = reader.read()
        dd = DetectionDataset(data=data)
        ```
    """

    def __init__(self, path: str, splits: Dict[str, Tuple[str, str]]) -> None:
        """Load a dataset from disk.

        This is a factory method that can read the dataset from different formats,
        when the dataset is already in a local directory.

        Args:
            path: Path to the dataset on the local filesystem.
            splits: Dictionnary indicating how to read the data.
                - The key is the name of the split
                - The value is a tuple containing the name of the annotation file for this split,
                    and the directory containing the images for this split.
        """

        super().__init__(path)
        self.splits = splits

    def read(self) -> pd.DataFrame():
        """Read the dataset from disk.

        Returns:
            DataFrame containing the data for all splits, with one row per annotation.
        """

        annotation_dataframes = []
        for split, (annotation_file, images_dir) in self.splits.items():
            images_path_prefix = os.path.join(self.path, images_dir)

            json = self._read_json(self.path, annotation_file)

            annotation_dataframe = self._read_annotations(json_data=json)
            annotation_dataframe["image_path"] = annotation_dataframe["image_name"].apply(
                lambda x: os.path.join(images_path_prefix, x)
            )
            annotation_dataframe["split"] = split
            annotation_dataframes.append(annotation_dataframe)

        annotation_by_bbox = pd.concat(annotation_dataframes, axis=0, ignore_index=True)
        annotation_by_bbox["bbox"] = [
            Bbox.from_coco(row.bbox, row.width, row.height, row.bbox_id) for _, row in annotation_by_bbox.iterrows()
        ]

        return annotation_by_bbox

    @staticmethod
    def _read_json(path: str, file: str) -> json:
        path_to_file = os.path.join(path, file)
        with open(path_to_file) as f:
            return json.load(f)

    def _read_annotations(self, json_data: json) -> pd.DataFrame:
        annotations = pd.DataFrame(json_data["annotations"])
        annotations = annotations[["image_id", "category_id", "bbox", "area", "id"]]
        annotations = annotations.rename(columns={"id": "bbox_id"})

        images = pd.DataFrame(json_data["images"])
        images = images[["id", "file_name", "height", "width"]]
        images = images.rename(columns={"id": "image_id", "file_name": "image_name"})

        categories = pd.DataFrame(json_data["categories"])
        categories = categories[["id", "name"]]
        categories = categories.rename(columns={"id": "category_id", "name": "category"})
        categories = categories.sort_values("category_id")
        self.categories = list(categories.category.unique())

        data = pd.merge(annotations, images, on="image_id", how="left")
        data = pd.merge(data, categories, on="category_id", how="left")

        return data
