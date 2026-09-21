import json
import shutil
from pathlib import Path

import typer
from PIL import Image
from rich.console import Console
from rich.progress import track

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
SUPPORTED_OBB_SHAPES = {"oriented_rectangle", "rectangle"}

console = Console()


def create_output_directory(output_path, source_path, folder_name) -> Path:
    output_path = output_path or source_path.resolve().parent / folder_name
    output_path.mkdir(parents=True, exist_ok=True)

    return output_path


cli = typer.Typer(rich_markup_mode="rich", help="LabelMe 标签转 YOLO 标签 (OBB, 旋转矩形)")


def normalize_points(points, img_width, img_height):
    """四角点像素坐标 -> 归一化浮点 (x1,y1,x2,y2,x3,y3,x4,y4)。"""
    norm = []
    for x, y in points:
        norm.append(x / img_width)
        norm.append(y / img_height)
    return norm


def expand_axis_aligned_corners(points):
    """轴对齐 rectangle 的两点对角 -> 顺时针四角。"""
    (x1, y1), (x2, y2) = points[0], points[1]
    return [
        [x1, y1],
        [x2, y1],
        [x2, y2],
        [x1, y2],
    ]


def convert_labelme_to_yolo_obb(json_path, txt_path, classes, img_width,
                                img_height, missing_labels=None):
    with open(json_path, "r") as f:
        data = json.load(f)

    lines = []
    skipped = 0
    for shape in data["shapes"]:
        stype = shape.get("shape_type", "")
        points = shape.get("points", [])
        label = shape.get("label", "")

        if stype == "oriented_rectangle" and len(points) == 4:
            corners = points
        elif stype == "rectangle" and len(points) == 2:
            corners = expand_axis_aligned_corners(points)
        else:
            skipped += 1
            console.log(
                f"[yellow]跳过[/yellow] {json_path.name}: label={label!r} "
                f"shape_type={stype!r}（非 OBB，不做最小外接矩形近似）"
            )
            continue

        if label not in classes:
            skipped += 1
            if missing_labels is not None:
                missing_labels[label] = missing_labels.get(label, 0) + 1
            continue

        class_id = classes.index(label)
        norm = normalize_points(corners, img_width, img_height)
        lines.append(
            f"{class_id} " + " ".join(map(lambda x: f"{x:.6f}", norm)) + "\n"
        )

    with open(txt_path, "w") as f:
        f.writelines(lines)
    return skipped


@cli.command()
def process_labelme_to_yolo_obb(
    image_path: Path = typer.Argument(..., help="图片目录"),
    class_path: Path = typer.Argument(..., help="classes.txt（每行一个类别名）"),
    label_path: Path = typer.Option(None, "--label_path", "-l", help="标签 JSON 目录（默认与图片同目录）"),
    output_path: Path = typer.Option(None, "--output_path", "-o", help="输出目录"),
):
    """
    将 LabelMe 标注转换为 YOLO-OBB 格式 (class_id x1 y1 x2 y2 x3 y3 x4 y4)

    说明:
        1. 遍历图片目录, 每张图片查找同名的 .json 标注文件
        2. oriented_rectangle 的四个角点原样写出; rectangle(轴对齐)展开成四角;
           多边形/圆/掩码等一律跳过并警告（不做最小外接矩形近似）
        3. 所有坐标按图片宽高归一化, 保留 6 位小数
        4. classes.txt 每行一个类别名, 行号即 class_id, 必须与标注 label 完全一致
        5. 没有对应 json 的图片仅拷贝, 不生成 txt
        6. 转换结果默认输出到图片目录同级的 json2yolo_obb 文件夹

    使用示例:
        1. 【基本转换】标注 json 与图片在同一目录
            python labelme_to_yolo_obb.py ./obb_labeled ./classes.txt

        2. 【标签目录分离】标注 json 在 labels 目录
            python labelme_to_yolo_obb.py ./images ./classes.txt -l ./labels

        3. 【指定输出目录】
            python labelme_to_yolo_obb.py ./images ./classes.txt -o ./yolo_obb
    """
    images = [
        f
        for f in image_path.iterdir()
        if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    label_path = label_path or image_path
    output_path = create_output_directory(output_path, image_path, "json2yolo_obb")

    classes = []
    with open(class_path, "r") as f:
        classes = f.read().splitlines()

    total_skipped = 0
    missing_labels = {}
    for img_file in track(images, description="Converting to YOLO-OBB..."):
        img = Image.open(img_file)
        base_name = img_file.stem
        json_file = label_path / f"{base_name}.json"
        txt_file = output_path / f"{base_name}.txt"

        if json_file.exists():
            total_skipped += convert_labelme_to_yolo_obb(
                json_file, txt_file, classes, img.width, img.height,
                missing_labels
            )
        shutil.copy(img_file, output_path)

    shutil.copy(class_path, output_path / "classes.txt")
    for label, n in missing_labels.items():
        console.log(
            f"[red]警告[/red]: label={label!r} 不在 classes.txt 中，共跳过 {n} 个"
        )
    console.log(
        f"[green]完成[/green] 共处理 {len(images)} 张图, 跳过 {total_skipped} 个非 OBB 标注, "
        f"输出目录: {output_path}"
    )


if __name__ == "__main__":
    cli()
