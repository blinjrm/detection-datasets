<div align="center">

<img src="images/dd_logo.png" alt="logo" width="100"/>

<br>

# Detection datasets

<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/-Python 3.8-blue?style=flat-square&logo=python&logoColor=white"></a>
<a href="https://black.readthedocs.io/en/stable/"><img alt="Code style: black" src="https://img.shields.io/badge/code%20style-black-black.svg?style=flat-square&labelColor=gray"></a>
<a href="https://github.com/blinjrm/detection-datasets/actions/workflows/ci.yaml"><img src="https://img.shields.io/github/workflow/status/blinjrm/detection-datasets/CI?label=CI&style=flat-square"/></a>
<a href="https://github.com/blinjrm/detection-datasets/actions/workflows/pypi.yaml"><img src="https://img.shields.io/github/workflow/status/blinjrm/detection-datasets/Python%20package?label=Build&style=flat-square"/></a>
<a href="https://pypi.org/project/detection-datasets/"><img src="https://img.shields.io/pypi/status/detection-datasets?style=flat-square"/></a>

<br>

*Easily load and transform datasets for object detection.*

</div>
<br>

---

**Documentation**: https://blinjrm.github.io/detection-datasets/

**Source Code**: https://github.com/blinjrm/detection-datasets

**Datasets on Hugging Face Hub**: https://huggingface.co/detection-datasets

---

<br>

`detection_datasets` aims to make it easier to work with detection datasets.
The main features are:
* **Read** the dataset :
    * from disk if it has already been downloaded.
    * directly from the Hugging Face Hub if it [already exist](https://huggingface.co/detection-datasets).
* **Transform** the dataset:
    * Select a subset of data.
    * Remap categories.
    * Create new train-val-test splits.
* **Visualize** the annotations.
* **write** the dataset:
    * to disk, selecting the target detection format: `COCO`, `YOLO` and more to come.
    * to the Hugging Face Hub for easy reuse in a different environment and share with the community.

## Requirements

Python 3.8+

`detection_datasets` is upon the great work of:

* <a href="https://pandas.pydata.org/" class="external-link" target="_blank">Pandas</a> for manipulating data.
* <a href="https://huggingface.co/" class="external-link" target="_blank">Hugging Face</a> to store and load datasets from the Hub.

## Installation

<div class="termy">

```console
$ pip install detection_datasets
```

# Examples

```Python
from detection_datasets import DetectionDataset
```

## 1. Read

From local files:

```Python
config = {
    'dataset_format': 'coco',                   # the format of the dataset on disk
    'path': 'path/do/data/on/disk',             # where the dataset is located
    'splits': {                                 # how to read the files
        'train': ('train.json', 'train'),
        'test': ('test.json', 'test'),
    },
}

dd = DetectionDataset()
dd.from_disk(**config)

# note that you can use method cascading as well:
# dd = DetectionDataset().from_disk(**config)
```

From the Hugging Face Hub:

```Python
dd = DetectionDataset().from_hub(name='fashionpedia')
```
Currently supported format for reading datasets are:
- COCO
- *more to come*

The list of datasets available from the Hub is given by:
```Python
DetectionDataset().available_in_hub
```

## 2. Transform

Here we select a subset of 10.000 images and create new train-val-test splits, overwritting the splits from the original dataset:
```Python
dd = DetectionDataset()\
    .from_hub(name='fashionpedia')\
    .select(n_images=10000)\
    .split(splits=[0.8, 0.1, 0.1])
```

## 3. Visualize

The `DetectionDataset` objects contains several properties to analyze your data:


```Python
dd.data
# This is equivlent to calling `dd.get_data('image')`, and returns a DataFrame with 1 row per image

dd.get_data('bbox')     # Returns a DataFrame with 1 row per annotation

dd.n_images             # Number of images

dd.n_bbox               # Number of annotations

dd.splits               # List of split names

dd.split_proportions    # DataFrame with the % of iamges in each split

dd.categories           # DataFrame with the categories and thei ids

dd.category_names       # List of categories

dd.n_categories         # Number of categories

```

You can also visualize a image with its annotations in a notebook:
```Python
dd.show()                   # Shows a random image from the dataset
dd.show(image_id=42)        # Shows the select image based on image_id
```

<div align="center">
<img src="images/show.png" alt="hub viewer" width="500"/>
</div>

## 4. Write

Once the dataset is ready, you can write it to the local filesystem in a given format:

```Python
dd.to_disk(
    dataset_format='yolo',
    name='MY_DATASET_NAME',
    path='DIRECTORY_TO_WRITE_TO',
)
```

Currently supported format for writing datasets are:
- YOLO
- MMDET
- *more to come*

The dataset can also be easily uploaded to the Hugging Face Hub, for reuse later on or in a different environment:
```Python
dd.to_hub(dataset_name='MY_DATASET_NAME', repo_name='MY_REPO_OR_ORGANISATION')
```
The dataset viewer on the Hub will work out of the box, and we encourage you to update the README in your new repo to make it easier for the comminuty to use it.

<div align="center">
<img src="images/hub.png" alt="hub viewer" width="800"/>
</div>
