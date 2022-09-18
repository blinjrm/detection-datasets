<div align="center">

<img src="images/dd_logo.png" width="100"/>

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


<br>

## Requirements

Python 3.8+

detection_datasets is upon the great work of:

* <a href="https://pandas.pydata.org/" class="external-link" target="_blank">Pandas</a> for manipulating data.
* <a href="https://huggingface.co/" class="external-link" target="_blank">Hugging Face</a> to store and load datasets from the Hub.

## Installation

<div class="termy">

```console
$ pip install detection_datasets
```

<br>

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
dd = DetectionDataset().from_hub('fashionpedia')
```

The list of datasets available from the Hub is given by:
```Python
DetectionDataset().available_in_hub
```
