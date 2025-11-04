import json
import shutil
from pathlib import Path

import typer
from PIL import Image
from rich.progress import track

cli = typer.Typer()

IMAGE_FORMAT = [".jpg", ".png", ".jpeg", ".webp", ".tiff", ".bmp"]
DEFAULT_JSON_TEMPLATE = {
    "version": "5.3.1",
    "flags": {},
    "shapes": [],
    "imagePath": None,
    "imageData": None,
    "imageHeight": None,
    "imageWidth": None,
}


def create_output_directory(output_dir, source_path, folder_name):
    if output_dir is None:
        output_dir = Path(source_path).resolve().parent / folder_name
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def xywh2xyxy(box, img_width, img_height):
    class_id, x, y, w, h = map(float, box.split())
    x_min = (x - w / 2) * img_width
    y_min = (y - h / 2) * img_height
    x_max = (x + w / 2) * img_width
    y_max = (y + h / 2) * img_height
    return (class_id, x_min, y_min, x_max, y_max)


def convert_yolo_to_labelme(txt_path, json_path, classes, img_width, img_height):
    with open(txt_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    json_data = DEFAULT_JSON_TEMPLATE.copy()
    json_data.update(
        {
            "imagePath": txt_path.name.replace(".txt", ".jpg"),
            "imageHeight": img_height,
            "imageWidth": img_width,
            "shapes": [],
        }
    )

    for line in lines:
        class_id, x_min, y_min, x_max, y_max = xywh2xyxy(line, img_width, img_height)
        json_data["shapes"].append(
            {
                "label": classes[int(class_id)],
                "points": [[x_min, y_min], [x_max, y_max]],
                "group_id": None,
                "shape_type": "rectangle",
                "flags": {},
            }
        )

    with open(json_path, "w") as f:
        json.dump(json_data, f, indent=4)


@cli.command()
def process_yolo_det_to_labelme(
    image_path: str = typer.Argument(..., help="图片目录"),
    class_path: str = typer.Argument(..., help="classes.txt"),
    label_path: str = typer.Option(None, "--label_path", "-l", help="标签目录"),
    output_path: Path = typer.Option(None, "--output_path", "-o", help="输出目录"),
):
    images = [f for f in Path(image_path).iterdir() if f.suffix.lower() in IMAGE_FORMAT]
    label_path = label_path or image_path
    output_path = create_output_directory(output_path, image_path, "yolo2json_det")
    classes = []
    with open(class_path, "r") as f:
        classes = f.read().splitlines()

    for img_file in track(images, description="Converting to JSON..."):
        img = Image.open(img_file)
        base_name = img_file.stem
        txt_file = Path(label_path) / f"{base_name}.txt"
        json_file = output_path / f"{base_name}.json"

        if txt_file.exists():
            convert_yolo_to_labelme(txt_file, json_file, classes, img.width, img.height)
        shutil.copy(img_file, output_path)


if __name__ == "__main__":
    cli()
