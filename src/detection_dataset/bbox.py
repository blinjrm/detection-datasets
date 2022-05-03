from __future__ import annotations


class Bbox:
    """Class for manipulating bounding boxes.

    All bounding boxes are converted to internally the VOC format: [xmin, ymin, xmax, ymax], and can be exported to the
    VOC, COCO and YOLO formats.
    """

    def __init__(self, bbox: list[float], width: float, height: float) -> None:
        self.bbox = bbox
        self.width = width
        self.height = height
        self._validate_bbox()

    @classmethod
    def from_voc(cls, bbox: list[float], width: float, height: float) -> Bbox:
        """Keep the bbox in VOC format: xmin, ymin, xmax, ymax."""

        return Bbox(bbox, width, height)

    @classmethod
    def from_coco(cls, bbox: list[float], width: float, height: float) -> Bbox:
        """Convert the bbox from COCO format: xmin, ymin, w, h."""

        bbox = [bbox[0], bbox[1], bbox[0] + bbox[2], bbox[1] + bbox[3]]
        return Bbox(bbox, width, height)

    @classmethod
    def from_yolo(cls, bbox: list[float], width: float, height: float) -> Bbox:
        """Convert the bbox from YOLO format: relative xc, yc, w, h."""

        assert bbox[0] < 1 and bbox[1] < 1 and bbox[2] < 1 and bbox[3] < 1, "yolo bbox must be relative"

        bbox = [bbox[0] - bbox[2] / 2, bbox[1] - bbox[3] / 2, bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2]
        bbox = [bbox[0] * width, bbox[1] * height, bbox[2] * width, bbox[3] * height]
        return Bbox(bbox, width, height)

    def to_voc(self) -> list[float]:
        """Bbox is already in VOC format internally."""

        return self.bbox

    def to_coco(self) -> list[float]:
        """Convert the bbox to COCO format: xmin, ymin, w, h."""

        return [self.bbox[0], self.bbox[1], self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1]]

    def to_yolo(self) -> list[float]:
        """Convert the bbox to YOLO format: relative xc, yc, w, h."""

        bbox = self.to_coco()
        bbox = [bbox[0] / self.width, bbox[1] / self.height, bbox[2] / self.width, bbox[3] / self.height]
        return bbox

    def _validate_bbox(self) -> None:
        """Asserts that the bbox to the correct size."""

        assert self.bbox[2] >= self.bbox[0] and self.bbox[3] >= self.bbox[1], "bbox must be a rectangle"
        assert self.bbox[2] <= self.width and self.bbox[3] <= self.height, "bbox must be inside the image"
