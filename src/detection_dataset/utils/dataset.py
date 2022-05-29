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
                    "supercategory",
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
        self._clean_dategories()

    def concat(self, other: "Dataset") -> None:
        self.data = pd.concat([self.data, other.data])
        self._clean_dategories()
        return self

    def _clean_dategories(self) -> None:
        """Returns a DataFrame containing the categories found in the data with their id."""

        self.categories = (
            self.data.loc[:, ["category_id", "category", "supercategory"]]
            .drop_duplicates()
            .sort_values("category_id")
            .reset_index(drop=True)
        )

    @property
    def class_names(self) -> list:
        return self.categories.category.tolist()
