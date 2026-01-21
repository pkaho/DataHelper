import copy
import json
from enum import Enum
from pathlib import Path

import typer
from PIL import Image
from rich.progress import track

cli = typer.Typer(help="生成空标签文件，支持 txt/json 格式")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
JSON_FORMAT = {
    "version": "5.3.1",
    "flags": {},
    "shapes": [],
    "imagePath": None,
    "imageData": None,
    "imageHeight": None,
    "imageWidth": None,
}


class LabelType(str, Enum):
    txt = "txt"
    json = "json"


@cli.command()
def generate_empty_file(
    path: Path = typer.Argument(..., help="图片存放目录"),
    file_type: LabelType = typer.Argument(
        LabelType.txt, help="要生成的标签文件类型 [txt, json]"
    ),
):
    for img_file in track(
        path.iterdir(), description="Generating empty label files..."
    ):
        if img_file.suffix.lower() not in IMAGE_EXTENSIONS:
            continue
        filename = Path(img_file.parent) / f"{img_file.stem}.{file_type}"

        if file_type == "json":
            with open(filename, "w") as f:
                img = Image.open(img_file)
                data = copy.deepcopy(JSON_FORMAT)
                data["imagePath"] = img_file.name
                data["imageHeight"] = img.height
                data["imageWidth"] = img.width
                json.dump(data, f, indent=4)
        else:
            with open(filename, "w") as f:
                pass


if __name__ == "__main__":
    cli()
