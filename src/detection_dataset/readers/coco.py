from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from detection_dataset.bbox import Bbox
from detection_dataset.readers import BaseReader


class CocoReader(BaseReader):
    def __init__(self, path: str) -> None:
        super().__init__(path)

    def _load(self) -> pd.DataFrame:
        annotation_files = Path(self.path).glob("*.json")
        annotation_dataframes = [self._json_to_dataframe(self._read_json(file)) for file in annotation_files]
        annotation_by_bbox = pd.concat(annotation_dataframes, axis=0, ignore_index=True)
        annotation_by_bbox["bbox"] = [
            Bbox.from_coco(row.bbox, row.width, row.height) for _, row in annotation_by_bbox.iterrows()
        ]
        return annotation_by_bbox

    @staticmethod
    def _read_json(path: str) -> json:
        with open(path) as f:
            return json.load(f), path.stem

    @staticmethod
    def _json_to_dataframe(json: json) -> pd.DataFrame:
        json_data, split = json
        annotations = pd.DataFrame(json_data["annotations"])
        annotations = annotations.drop(columns=["segmentation", "iscrowd"])
        annotations = annotations.rename(columns={"id": "bbox_id"})

        images = pd.DataFrame(json_data["images"])
        images = images.drop(columns=["license", "time_captured", "original_url", "isstatic", "kaggle_id"])
        images = images.rename(columns={"id": "image_id", "file_name": "filename"})

        data = pd.merge(annotations, images, on="image_id", how="left")
        data["split"] = split
        return data

    def _annotation_by_image(self):
        """Returns the dataframe grouped by image."""

        data = self.annotation_by_bbox.groupby(["filename"])

        return pd.DataFrame(
            {
                "width": data["width"].first(),
                "height": data["height"].first(),
                "category": data["category"].apply(list),
                "attributes": data["attributes"].apply(list),
                "area": data["area"].apply(list),
                "bbox_coco": data["bbox"].apply(list),
                "split": data["split"].first(),
            }
        ).reset_index()
