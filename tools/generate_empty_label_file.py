import copy
import json
from enum import Enum
from pathlib import Path

import typer
from PIL import Image
from rich.progress import track

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
JSON_FORMAT = {
    "version": "5.3.1",
    "flags": {},
    "shapes": [],
    "imagePath": None,
    "imageData": None,
    "imageHeight": None,
    "imageWidth": None,
}


cli = typer.Typer(rich_markup_mode="rich", help="生成空标签文件，支持 txt/json 格式")


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
    """
    为目录下所有图片生成同名的空标签文件 (txt 或 json)

    说明:
        1. 遍历图片目录, 为每张图片生成同名标签文件 (不覆盖已存在的)
        2. txt 模式: 生成空文件 (用于占位)
        3. json 模式: 生成标准 LabelMe 空标注 (含图片宽高, shapes 为空)

    使用示例:
        1. 【生成空 txt 标签】默认类型
            python generate_empty_label_file.py ./images

        2. 【生成空 json 标注】
            python generate_empty_label_file.py ./images json
    """
    for img_file in track(
        path.iterdir(), description="Generating empty label files..."
    ):
        if img_file.suffix.lower() not in SUPPORTED_IMAGE_EXTENSIONS:
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
