import json
import shutil
from pathlib import Path

import typer
from PIL import Image
from rich.progress import track

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def create_output_directory(output_dir, source_path, folder_name) -> Path:
    output_dir = output_dir or source_path.resolve().parent / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


cli = typer.Typer(rich_markup_mode="rich", help="LabelMe 标签转 YOLO 标签 (目标检测)")


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
    """
    将 LabelMe 标注转换为 YOLO 目标检测格式 (class_id x_center y_center width height)

    说明:
        1. 遍历图片目录, 每张图片查找同名的 .json 标注文件
        2. 每个标注取其 points 的前两点作为对角坐标, 归一化为 xywh
        3. classes.txt 每行一个类别名, 行号即 class_id, 必须与标注 label 完全一致
        4. 没有对应 json 的图片仅拷贝, 不生成 txt
        5. 转换结果默认输出到图片目录同级的 json2yolo_det 文件夹

    使用示例:
        1. 【基本转换】标注 json 与图片在同一目录
            python labelme_to_yolo_det.py ./images ./classes.txt

        2. 【标签目录分离】标注 json 在 labels 目录
            python labelme_to_yolo_det.py ./images ./classes.txt -l ./labels

        3. 【指定输出目录】
            python labelme_to_yolo_det.py ./images ./classes.txt -o ./yolo_det
    """
    images = [
        f
        for f in image_path.iterdir()
        if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
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
