import json
import shutil
from pathlib import Path

import typer
from PIL import Image
from rich.progress import track

cli = typer.Typer()

IMAGE_FORMAT = [".jpg", ".png", ".jpeg", ".webp", ".tiff", ".bmp"]


def create_output_directory(output_dir, source_path, folder_name):
    if output_dir is None:
        output_dir = Path(source_path).resolve().parent / folder_name
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def xyxy2xywh(box, img_width, img_height):
    x_center = (box[0] + box[2]) / 2.0 / img_width
    y_center = (box[1] + box[3]) / 2.0 / img_height
    width = abs(box[2] - box[0]) / img_width
    height = abs(box[3] - box[1]) / img_height
    return (x_center, y_center, width, height)


def convert_labelme_to_yolo(json_path, txt_path, classes, img_width, img_height):
    with open(json_path, "r") as f:
        data = json.load(f)

    with open(txt_path, "w") as f:
        for shape in data["shapes"]:
            points = shape["points"]
            box = [points[0][0], points[0][1], points[1][0], points[1][1]]
            yolo_box = xyxy2xywh(box, img_width, img_height)
            class_id = classes.index(shape["label"])
            f.write(
                f"{class_id} " + " ".join(map(lambda x: f"{x:.6f}", yolo_box)) + "\n"
            )


@cli.command()
def process_labelme_to_yolo_det(
    image_path: Path = typer.Argument(..., help="图片目录"),
    class_path: str = typer.Argument(..., help="classes.txt"),
    label_path: Path = typer.Option(None, "--label_path", "-l", help="标签目录"),
    output_path: Path = typer.Option(None, "--output_path", "-o", help="输出目录"),
):
    images = [f for f in image_path.iterdir() if f.suffix.lower() in IMAGE_FORMAT]
    label_path = label_path or image_path
    output_path = create_output_directory(output_path, image_path, "json2yolo_det")
    classes = []

    with open(class_path, "r") as f:
        classes = f.read().splitlines()

    for img_file in track(images, description="Converting to YOLO..."):
        img = Image.open(img_file)
        base_name = img_file.stem
        json_file = label_path / f"{base_name}.json"
        txt_file = output_path / f"{base_name}.txt"

        if json_file.exists():
            convert_labelme_to_yolo(json_file, txt_file, classes, img.width, img.height)
        shutil.copy(img_file, output_path)

    shutil.copy(class_path, output_path / "classes.txt")


if __name__ == "__main__":
    cli()
