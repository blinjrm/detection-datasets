import pandas as pd


class Dataset:
    def __init__(self, data: pd.DataFrame = None) -> None:
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

    def concat(self, other: "Dataset") -> None:
        self.data = pd.concat([self.data, other.data])
        return self
