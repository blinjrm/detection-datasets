from typing import List, Tuple, Union

import numpy as np
import pandas as pd

from detection_dataset.utils.enums import Split


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

    data = pd.DataFrame(columns=COLUMNS).set_index(["image_id", "bbox_id"])

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
                Schema:
                    - category_id: Original category id
                    - category: Original category name
                    - new_category_id: New category id
                    - new_category: New category name
        """

        mapping = mapping.loc[:, ["category_id", "category", "new_category_id", "new_category"]]

        data = (
            self.data.reset_index()
            .merge(mapping, on=["category_id", "category"], how="left", validate="m:1")
            .set_index(["image_id", "bbox_id"])
        )
        data = data[data.new_category_id >= 0]
        self.data = data.rename(
            columns={
                "category_id": "category_id_original",
                "category": "category_original",
                "new_category_id": "category_id",
                "new_category": "category",
            }
        )

    def split(self, splits: Tuple[Union[int, float]]) -> None:
        """Splits the dataset into train, val and test.

        Args:
            splits: Tuple containing, depending on the type of the values (int or float):
                - Proportion of images to include in the train, val and test splits, if specified as floats.
                  The original splits will be overwritten.
                  The sum of the values in the tuple can be lower than 1, then the dataset size will be reduced.
                - Number of images to include in the each split, if specified as integers.
                  The original splits will be respected.
        """

        data = self.data.copy()

        if len(splits) != 3:
            raise ValueError(f"The splits must be a tuple of 3 elements, here it is: {splits}.")

        if all([isinstance(x, float) for x in splits]):
            assert sum(splits) <= 1, "The sum of the splits must lower than or equal to 1."

            n_train = int(splits[0] * len(data))
            n_val = int(n_train + splits[1] * len(data))
            n_test = int(n_val + splits[2] * len(data))
            data_train, data_val, data_test, _ = np.split(data, [n_train, n_val, n_test])

            data_train["split"] = Split.train.value
            data_val["split"] = Split.val.value
            data_test["split"] = Split.test.value

        elif all([isinstance(x, int) for x in self.splits]):
            data_train = data.loc[data.split == Split.train.value, :].sample(self.splits[0])
            data_val = data.loc[data.split == Split.val.value, :].sample(self.splits[1])
            data_test = data.loc[data.split == Split.test.value, :].sample(self.splits[2])

        else:
            raise ValueError("Splits must be either int or float")

        return pd.concat([data_train, data_val, data_test])

    def limit_images(self, n_images: int) -> None:
        """Limits the number of images to n_images.

        Args:
            n_images: Number of images to include in the dataset.
                The original proportion of images between splits will be respected.
        """

        if self.n_images > len(self.data):
            raise ValueError(
                "The number of images to include in the dataset is greater than the number of images present."
            )

        data = self.data.copy()
        split_data = []
        for split in Split:
            split_data.append(
                data.loc[data.split == split.value, :].sample(
                    int(n_images / self.split_proportions[split]), random_state=42
                )
            )

        self.data = pd.concat(split_data)

    @property
    def n_images(self) -> int:
        """Returns the number of images in the dataset."""

        return len(self.data)

    @property
    def splits(self) -> List[str]:
        """Returns the splits of the dataset."""

        return self.data.split.unique().tolist()

    @property
    def split_proportions(self) -> Tuple[float, float, float]:
        """Returns the proportion of images in the train, val and test splits."""

        # proportion = self.data.split.value_counts() / len(self.data)

        data = self.data.copy()
        return pd.DataFrame({s.value: [len(data[data.split == s.value]) / len(data)] for s in Split})

    @property
    def categories(self) -> None:
        """Creates a DataFrame containing the categories found in the data with their id."""

        return (
            self.data.loc[:, ["category_id", "category", "supercategory"]]
            .drop_duplicates()
            .sort_values("category_id")
            .reset_index(drop=True)
        )

    @property
    def categorie_names(self) -> List[str]:
        """Returns the categories names."""

        return self.categories.category.unique()

    @property
    def n_categories(self) -> int:
        """Returns the number of categories."""

        return self.categories.category.nunique()
