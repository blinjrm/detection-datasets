from typing import List, Tuple, Union

import pandas as pd


class Dataset:

    COLUMNS = [
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

    data = pd.DataFrame(columns=COLUMNS)

    def __init__(self, data: pd.DataFrame = None) -> None:
        """Initializes the dataset."""

        if data is not None:
            data = data[data.columns.intersection(self.COLUMNS)]
            self.concat(data)

    def concat(self, other_data: pd.DataFrame) -> None:
        """Concatenates the existing data with new data."""

        self.data = pd.concat([self.data, other_data])

    def map_categories(self, mapping: pd.DataFrame) -> None:
        """Maps the categories to the new categories.

        Args:
            category_mapping: A DataFrame mapping original categories to new categories.
        """

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

    def limit_images(self, n_images: int) -> None:
        """Limits the number of images to n_images.

        TODO: add documentation
        """

        # keep original splits
        raise NotImplementedError

    def split(self, splits: Tuple[Union[int, float]]) -> None:
        """Splits the dataset into train, val and test.

        TODO: add documentation
        """

        # keep original splits
        raise NotImplementedError

    @property
    def dategories(self) -> None:
        """Creates a DataFrame containing the categories found in the data with their id."""

        return (
            self.data.loc[:, ["category_id", "category", "supercategory"]]
            .drop_duplicates()
            .sort_values("category_id")
            .reset_index(drop=True)
        )

    @property
    def class_names(self) -> List[str]:
        """Returns the class names."""

        return self.categories.category.unique()

    @property
    def n_classes(self) -> int:
        """Returns the number of classes."""

        return self.categories.category.nunique()
