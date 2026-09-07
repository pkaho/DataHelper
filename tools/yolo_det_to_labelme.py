import json
import shutil
from pathlib import Path

import typer
from PIL import Image
from rich.progress import track

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
DEFAULT_JSON_TEMPLATE = {
    "version": "5.3.1",
    "flags": {},
    "shapes": [],
    "imagePath": None,
    "imageData": None,
    "imageHeight": None,
    "imageWidth": None,
}


cli = typer.Typer(rich_markup_mode="rich", help="YOLO 标签转 LabelMe 标签 (目标检测)")


def create_output_directory(output_dir, source_path, folder_name) -> Path:
    output_dir = output_dir or source_path.resolve().parent / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def xywh2xyxy(box, img_width, img_height):
    # 兼容 5 字段 (class x y w h) 与 6 字段 (class x y w h conf)
    parts = box.split()
    class_id, x, y, w, h = map(float, parts[:5])
    x_min = (x - w / 2) * img_width
    y_min = (y - h / 2) * img_height
    x_max = (x + w / 2) * img_width
    y_max = (y + h / 2) * img_height
    return (class_id, x_min, y_min, x_max, y_max)


def convert_yolo_to_labelme(
    txt_path, json_path, classes, img_width, img_height, image_name
):
    with open(txt_path, "r") as f:
        lines = [line.strip() for line in f if line.strip()]

    json_data = DEFAULT_JSON_TEMPLATE.copy()
    json_data.update(
        {
            "imagePath": image_name,
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
    image_path: Path = typer.Argument(..., help="图片目录"),
    class_path: str = typer.Argument(..., help="classes.txt"),
    label_path: Path = typer.Option(None, "--label_path", "-l", help="标签目录"),
    output_path: Path = typer.Option(None, "--output_path", "-o", help="输出目录"),
):
    """
    将 YOLO 目标检测标签 (.txt) 转换为 LabelMe 标注 (.json)

    说明:
        1. 遍历图片目录, 每张图片查找同名的 .txt 标签文件
        2. 标签格式: class_id x_center y_center width height (归一化坐标),
           兼容带置信度的 6 字段格式, 多余字段忽略
        3. classes.txt 每行一个类别名, 行号即 class_id, 用于还原类别名
        4. 没有对应 txt 的图片仅拷贝, 不生成 json
        5. 转换结果默认输出到图片目录同级的 yolo2json_det 文件夹

    使用示例:
        1. 【基本转换】标签与图片在同一目录
            python yolo_det_to_labelme.py ./images ./classes.txt

        2. 【标签目录分离】标签在 labels 目录
            python yolo_det_to_labelme.py ./images ./classes.txt -l ./labels

        3. 【指定输出目录】
            python yolo_det_to_labelme.py ./images ./classes.txt -o ./labelme
    """
    images = [
        f
        for f in image_path.iterdir()
        if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    label_path = label_path or image_path
    output_path = create_output_directory(output_path, image_path, "yolo2json_det")

    classes = []
    with open(class_path, "r") as f:
        classes = f.read().splitlines()

    for img_file in track(images, description="Converting to JSON..."):
        img = Image.open(img_file)
        base_name = img_file.stem
        txt_file = label_path / f"{base_name}.txt"
        json_file = output_path / f"{base_name}.json"

        if txt_file.exists():
            convert_yolo_to_labelme(
                txt_file, json_file, classes, img.width, img.height, img_file.name
            )
        shutil.copy(img_file, output_path)


if __name__ == "__main__":
    cli()
