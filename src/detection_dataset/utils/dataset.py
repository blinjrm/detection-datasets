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
        # return self

    def map_categories(self, mapping: pd.DataFrame) -> None:
        """Maps the categories to the new categories."""

        mapping = mapping.loc[:, ["category_id", "category", "new_category_id", "new_category"]]

        data = self.data.copy()
        data = data.merge(mapping, on=["category_id", "category"], how="left", validate="m:1")
        data = data[data.new_category_id >= 0]
        self.data = data.rename(
            columns={
                "category_id": "category_id_original",
                "category": "category_original",
                "new_category_id": "category_id",
                "new_category": "category",
            }
        )

        self._clean_dategories()

    def _clean_dategories(self) -> None:
        """Creates a DataFrame containing the categories found in the data with their id."""

        self.categories = (
            self.data.loc[:, ["category_id", "category", "supercategory"]]
            .drop_duplicates()
            .sort_values("category_id")
            .reset_index(drop=True)
        )

    @property
    def class_names(self) -> list:
        return self.categories.category.unique()
