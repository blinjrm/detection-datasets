import os
import shutil
from abc import ABC, abstractmethod

import pandas as pd
import wandb

from detection_dataset.utils import Dataset


class BaseWriter(ABC):
    def __init__(
        self,
        dataset: Dataset,
        name: str,
        path: str,
        destination: str,
    ) -> None:
        """Base class for writing datasets to disk.

        Args:
            dataset: Dataframe containing the dataset to write to disk.
            name: Name of the dataset to be created in the "path" directory.
            path: Path to the directory where the dataset will be created.
            wandb_upload: Whether to upload the dataset to W&B artifacts.
        """

        self.data = dataset.data_by_image
        self.name = name
        self.path = path
        self.destination = destination
        self.dataset_dir = os.path.join(self.path, self.name)
        self.class_names = dataset.category_names
        self.n_classes = dataset.n_categories

    @abstractmethod
    def write(self) -> pd.DataFrame:
        """Writes the dataset to disk."""

    def upload_to_wandb(self) -> None:
        """Uploads the dataset to W&B artifacts."""

        # import inspect
        # inspect.signature(wandb.init)

        try:
            run = wandb.init(
                project="detection-dataset",
                name="dataset_upload",
                resume=True,
                settings=wandb.Settings(start_method="fork"),
            )

            # Log the entire dataset
            artifact = wandb.Artifact(self.name, type="dataset")
            artifact.add_dir(self.dataset_dir)

            # Log a table for dataset visualization
            # table = self.make_wandb_table()
            # artifact.add_file(table, name='table.csv')

            # Upload to W&B
            run.log_artifact(artifact, aliases=[self.format])
        except Exception as e:
            print(e)

        # Delete local dataset
        self.delete_local_dataset()

    def make_wandb_table(self) -> wandb.Table:
        """Creates a W&B table containing the dataset."""

        # data = self.data.copy()

    def delete_local_dataset(self) -> None:
        """Deletes the local dataset."""

        shutil.rmtree(self.dataset_dir)
