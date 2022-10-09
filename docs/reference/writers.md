# Base writer class

This is the base class that other writers inherit from.  
It enforces the use of the `write()` method in all writers.

::: detection_datasets.writers.base.BaseWriter

<br>

# Mmdet writer

This writer saves a dataset to disk in the MMdetection format, that is called "middle format" on the MMdetection documentation.  
For more details check [Tutorial 2: Customize Datasets](https://mmdetection.readthedocs.io/en/latest/tutorials/customize_dataset.html#reorganize-new-data-format-to-middle-format) on the MMdetection documentation.

::: detection_datasets.writers.mmdet.MmdetWriter

<br>

# YOLO writer

This writer saves a dataset to disk in the YOLO format.  

::: detection_datasets.writers.yolo.YoloWriter

<br>

# COCO writer

This writer saves a dataset to disk in the COCO format.  

::: detection_datasets.writers.coco.CocoWriter
