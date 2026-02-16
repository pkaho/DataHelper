import json
import shutil
from pathlib import Path

import typer
from PIL import Image
from rich.progress import track

from tools.utils import SUPPORTED_IMAGE_EXTENSIONS
from tools.utils import create_output_directory

cli = typer.Typer(help="LabelMe 标签转 YOLO 标签 (分割)")


def normalize_polygon(polygon, img_width, img_height):
    normalized = []
    for point in polygon:
        x = point[0] / img_width
        y = point[1] / img_height
        normalized.extend([x, y])
    return normalized


def convert_labelme_to_yolo_seg(json_path, txt_path, classes, img_width, img_height):
    with open(json_path, "r") as f:
        data = json.load(f)

    with open(txt_path, "w") as f:
        for shape in data["shapes"]:
            label = shape["label"]

            class_id = classes.index(label)
            polygon = shape["points"]
            normalized_polygon = normalize_polygon(polygon, img_width, img_height)

            # Write to file: class_id x1 y1 x2 y2 ...
            line = f"{class_id} " + " ".join(
                [f"{coord:.6f}" for coord in normalized_polygon]
            )
            f.write(line + "\n")


@cli.command()
def process_labelme_to_yolo_seg(
    image_path: Path = typer.Argument(..., help="图片目录"),
    class_path: str = typer.Argument(..., help="classes.txt"),
    label_path: Path = typer.Option(None, "--label_path", "-l", help="标签目录"),
    output_path: Path = typer.Option(None, "--output_path", "-o", help="输出目录"),
):
    images = [f for f in image_path.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
    label_path = label_path or image_path
    output_path = create_output_directory(output_path, image_path, "json2yolo_seg")

    classes = []
    with open(class_path, "r") as f:
        classes = f.read().splitlines()

    for img_file in track(images, description="Converting to YOLO segmentation..."):
        img = Image.open(img_file)
        base_name = img_file.stem
        json_file = label_path / f"{base_name}.json"
        txt_file = output_path / f"{base_name}.txt"

        if json_file.exists():
            convert_labelme_to_yolo_seg(
                json_file, txt_file, classes, img.width, img.height
            )
        shutil.copy(img_file, output_path)

    shutil.copy(class_path, output_path / "classes.txt")


if __name__ == "__main__":
    cli()
