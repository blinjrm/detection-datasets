from __future__ import annotations

import json
import os

import pandas as pd

from detection_dataset.bbox import Bbox
from detection_dataset.readers import BaseReader


class CocoReader(BaseReader):
    def __init__(self, path: str, splits: dict[str, tuple[str, str]]) -> None:
        super().__init__(path, splits)

    def _load(self) -> pd.DataFrame:
        annotation_dataframes = []
        for split, (annotation_file, images_dir) in self.splits.items():
            images_path_prefix = os.path.join(self.path, images_dir)
            annotation_dataframe = self._json_to_dataframe(self._read_json(self.path, annotation_file))
            annotation_dataframe["image_path"] = annotation_dataframe["filename"].apply(
                lambda x: os.path.join(images_path_prefix, x)
            )
            annotation_dataframe["split"] = split
            annotation_dataframes.append(annotation_dataframe)

        annotation_by_bbox = pd.concat(annotation_dataframes, axis=0, ignore_index=True)
        annotation_by_bbox["bbox"] = [
            Bbox.from_coco(row.bbox, row.width, row.height) for _, row in annotation_by_bbox.iterrows()
        ]
        return annotation_by_bbox

    @staticmethod
    def _read_json(path: str, file: str) -> json:
        path_to_file = os.path.join(path, file)
        with open(path_to_file) as f:
            return json.load(f)

    @staticmethod
    def _json_to_dataframe(json_data: json) -> pd.DataFrame:
        annotations = pd.DataFrame(json_data["annotations"])
        annotations = annotations.drop(columns=["segmentation", "iscrowd"], errors="ignore")
        annotations = annotations.rename(columns={"id": "bbox_id"})

        images = pd.DataFrame(json_data["images"])
        images = images.drop(
            columns=["license", "time_captured", "original_url", "isstatic", "kaggle_id"], errors="ignore"
        )
        images = images.rename(columns={"id": "image_id", "file_name": "filename"})

        data = pd.merge(annotations, images, on="image_id", how="left")
        return data
