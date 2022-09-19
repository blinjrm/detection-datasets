from typing import List

from huggingface_hub import HfApi

api = HfApi()

ORGANISATION = "detection-datasets"


def available_in_hub(repo_name: str = ORGANISATION) -> List[str]:
    """List the datasets available in the Hugging Face Hub.

    Args:
        repo_name: user or organisation where the dataset is stored on the Hub.

    Returns:
        List of names of datasets registered in the Hugging Face Hub, under the 'detection-datasets' organisation.

    Example:
        >>> available_in_hub()
        ['fashionpedia']
    """

    datasets = api.list_datasets(author=repo_name)

    return [dataset.id.split("/")[-1] for dataset in datasets]
