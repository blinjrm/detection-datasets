import pandas as pd
from PIL import Image, ImageDraw


def show_image_bbox(rows: pd.DataFrame) -> Image:
    """Show the image with bounding boxes and labels.

    Args:
        rows: DataFrame with one row per annotation.

    Returns:
        Image with bounding boxes and labels.
    """

    image = Image.open(rows.reset_index().loc[0, "image_path"])
    draw = ImageDraw.Draw(image)

    for _, row in rows.iterrows():
        bbox = row["bbox"].bbox
        draw.rectangle(xy=[bbox[0], bbox[1], bbox[2], bbox[3]], outline="red")
        draw.text(xy=(bbox[0], bbox[1]), text=row.category, fill="red")

    return image
