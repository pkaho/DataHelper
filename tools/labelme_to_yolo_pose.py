import json
import shutil
from pathlib import Path

import typer
from PIL import Image
from rich.progress import track

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# YOLO pose 可见性约定: 0=缺失/未标注, 1=遮挡, 2=可见
KEYPOINT_VISIBILITY = 2
MISSING_KEYPOINT_VISIBILITY = 0


def create_output_directory(output_dir, source_path, folder_name) -> Path:
    output_dir = output_dir or source_path.resolve().parent / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


cli = typer.Typer(rich_markup_mode="rich", help="LabelMe 标签转 YOLO 标签 (关键点检测)")


def xyxy2xywh(box, img_width, img_height):
    x_center = (box[0] + box[2]) / 2.0 / img_width
    y_center = (box[1] + box[3]) / 2.0 / img_height
    width = abs(box[2] - box[0]) / img_width
    height = abs(box[3] - box[1]) / img_height
    return (x_center, y_center, width, height)


def collect_keypoint_names(label_path, images):
    """未提供关键点名称文件时, 从所有 json 的 point 类型 shape 中收集关键点名称"""
    names = set()
    for img_file in images:
        json_file = label_path / f"{img_file.stem}.json"
        if not json_file.exists():
            continue
        with open(json_file, "r") as f:
            data = json.load(f)
        for shape in data["shapes"]:
            if shape["shape_type"] == "point":
                names.add(shape["label"])
    return sorted(names)


def convert_labelme_to_yolo_pose(
    json_path, txt_path, classes, keypoint_names, img_width, img_height
):
    """将单个 LabelMe json 转换为 YOLO pose 标注。

    约定:
    - 目标框: shape_type 为 rectangle 的 shape, label 为类别名
    - 关键点: shape 的 label 为关键点名称(通常 shape_type 为 point)
    - 框与关键点通过 group_id 匹配: 同一 group_id 的框和点构成一个实例,
      组内没有关键点的框, 关键点输出全 0 (只检测不带点)
    - 未设置 group_id 的 shape 合并为一个实例
    - 输出格式: class_id x_center y_center width height x1 y1 v1 x2 y2 v2 ...
      缺失的关键点输出 0 0 0
    """
    with open(json_path, "r") as f:
        data = json.load(f)

    # 按 group_id 分组, group_id 为 None 的 shape 合并为一个实例
    groups = {}
    for shape in data["shapes"]:
        group_id = shape.get("group_id")
        key = group_id if group_id is not None else "__default__"
        groups.setdefault(key, []).append(shape)

    with open(txt_path, "w") as f:
        for group in groups.values():
            bbox_shapes = [s for s in group if s["shape_type"] == "rectangle"]
            point_map = {
                s["label"]: s["points"][0]
                for s in group
                if s["shape_type"] != "rectangle" and s["label"] in keypoint_names
            }

            for bbox_shape in bbox_shapes:
                class_id = classes.index(bbox_shape["label"])
                box = [
                    bbox_shape["points"][0][0],
                    bbox_shape["points"][0][1],
                    bbox_shape["points"][1][0],
                    bbox_shape["points"][1][1],
                ]
                yolo_box = xyxy2xywh(box, img_width, img_height)

                keypoints = []
                for name in keypoint_names:
                    point = point_map.get(name)
                    if point is None:
                        keypoints.extend([0.0, 0.0, MISSING_KEYPOINT_VISIBILITY])
                    else:
                        keypoints.extend(
                            [
                                point[0] / img_width,
                                point[1] / img_height,
                                KEYPOINT_VISIBILITY,
                            ]
                        )

                numbers = list(yolo_box) + keypoints
                f.write(
                    f"{class_id} " + " ".join(map(lambda x: f"{x:.6f}", numbers)) + "\n"
                )


@cli.command()
def process_labelme_to_yolo_pose(
    image_path: Path = typer.Argument(..., help="图片目录"),
    class_path: str = typer.Argument(..., help="classes.txt"),
    keypoint_path: Path = typer.Option(
        None,
        "--keypoint_path",
        "-k",
        help="关键点名称文件(每行一个), 不传则从数据中自动收集",
    ),
    label_path: Path = typer.Option(None, "--label_path", "-l", help="标签目录"),
    output_path: Path = typer.Option(None, "--output_path", "-o", help="输出目录"),
):
    """
    将 LabelMe 标注转换为 YOLO pose 格式 (class_id bbox x1 y1 v1 x2 y2 v2 ...)

    说明:
        1. 目标框: shape_type 为 rectangle 的 shape, label 为类别名
        2. 关键点: shape 的 label 为关键点名称 (通常 shape_type 为 point)
        3. 框与关键点通过 group_id 匹配: 同一 group_id 的框和点构成一个实例
        4. 组内没有关键点的框, 关键点输出全 0 (如 circle 只检测不带点)
        5. 未设置 group_id 的 shape 合并为一个实例
        6. keypoint_names.txt 每行一个关键点名称, 顺序即输出顺序;
           不传 -k 时按字母序从数据中自动收集
        7. 可见性: 已标注=2, 缺失=0

    使用示例:
        1. 【基本转换】关键点顺序显式指定
            python labelme_to_yolo_pose.py ./images ./classes.txt -k ./keypoint_names.txt

        2. 【标签目录分离】标注 json 在 labels 目录
            python labelme_to_yolo_pose.py ./images ./classes.txt -k ./keypoint_names.txt -l ./labels

        3. 【自动收集关键点】
            python labelme_to_yolo_pose.py ./images ./classes.txt
    """
    images = [
        f
        for f in image_path.iterdir()
        if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    label_path = label_path or image_path
    output_path = create_output_directory(output_path, image_path, "json2yolo_pose")

    classes = []
    with open(class_path, "r") as f:
        classes = f.read().splitlines()

    if keypoint_path is None:
        keypoint_names = collect_keypoint_names(label_path, images)
        if not keypoint_names:
            raise typer.BadParameter(
                "未找到关键点, 请通过 --keypoint_path 指定关键点名称文件"
            )
    else:
        with open(keypoint_path, "r") as f:
            keypoint_names = f.read().splitlines()
    typer.echo(f"关键点顺序({len(keypoint_names)}): {', '.join(keypoint_names)}")

    for img_file in track(images, description="Converting to YOLO pose..."):
        img = Image.open(img_file)
        base_name = img_file.stem
        json_file = label_path / f"{base_name}.json"
        txt_file = output_path / f"{base_name}.txt"

        if json_file.exists():
            convert_labelme_to_yolo_pose(
                json_file, txt_file, classes, keypoint_names, img.width, img.height
            )
        shutil.copy(img_file, output_path)

    shutil.copy(class_path, output_path / "classes.txt")
    if keypoint_path is not None:
        shutil.copy(keypoint_path, output_path / "keypoint_names.txt")


if __name__ == "__main__":
    cli()
