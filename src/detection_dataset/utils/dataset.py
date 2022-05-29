from typing import List

import pandas as pd


class Dataset:
    def __init__(self, data: pd.DataFrame = None, categories: List[str] = None) -> None:
        self.data = (
            data
            if data is not None
            else pd.DataFrame(
                columns=[
                    "image_id",
                    "bbox_id",
                    "category_id",
                    "category",
                    "bbox",
                    "width",
                    "height",
                    "area",
                    "image_name",
                    "image_path",
                    "split",
                ]
            )
        )
        self.categories = categories or []

    def concat(self, other: "Dataset") -> None:
        self.data = pd.concat([self.data, other.data])
        self.categories = list(set(self.categories + other.categories))
        return self
