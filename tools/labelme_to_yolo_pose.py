import json
import shutil
from pathlib import Path

import typer
from PIL import Image
from rich.progress import track

from tools.show_pose import show
from tools.utils import SUPPORTED_IMAGE_EXTENSIONS
from tools.utils import create_output_directory

cli = typer.Typer(help="LabelMe 标签转 YOLO 标签 (关键点)")


def xyxy2xywh(box, img_width, img_height):
    x_center = (box[0] + box[2]) / 2.0 / img_width
    y_center = (box[1] + box[3]) / 2.0 / img_height
    width = abs(box[2] - box[0]) / img_width
    height = abs(box[3] - box[1]) / img_height
    return (x_center, y_center, width, height)


def convert_labelme_to_yolo(
    json_path, txt_path, classes, point_order, img_width, img_height
):
    with open(json_path, "r") as f:
        data = json.load(f)

    retangles = []
    points = []

    p_order = {po: None for po in point_order}

    for shape in data["shapes"]:
        if int(shape["group_id"]) > 2:
            raise ValueError(
                f"{json_path} 可见性不符合规范 [0: 不可见, 1: 部分可见, 2: 全部可见]"
            )

        if shape["group_id"] is None:
            shape["group_id"] = 2

        shape_type = shape["shape_type"]
        infos = {"label": shape["label"]}

        if shape_type == "rectangle":
            infos["xyxy"] = shape["points"][0] + shape["points"][1]
            infos.update(p_order)
            retangles.append(infos)
        elif shape_type == "point":
            infos["vis"] = shape["group_id"]
            infos["xy"] = shape["points"][0]
            points.append(infos)

    point_in_rectangle = {tuple(point["xy"]): False for point in points}

    for point in points:
        x, y = point["xy"]
        label = point["label"]
        for retangle in retangles:
            x1, y1, x2, y2 = retangle["xyxy"]
            x_min, x_max = min(x1, x2), max(x1, x2)
            y_min, y_max = min(y1, y2), max(y1, y2)
            if x_min <= x <= x_max and y_min <= y <= y_max:
                # if retangle.get(label) is not None:
                #     raise f"{json_path} {label} has more than one point in one rectangle"
                retangle[label] = point
                point_in_rectangle[(x, y)] = True

    outside_points = [
        point for point, is_inside in point_in_rectangle.items() if not is_inside
    ]
    if outside_points:
        print(
            f"{json_path} contains {len(outside_points)} points not in any rectangle:"
        )

    with open(txt_path, "w") as f:
        for retangle in retangles:
            xywh = xyxy2xywh(retangle["xyxy"], img_width, img_height)
            yolo_box = [round(i, 6) for i in xywh]
            for po in point_order:
                if retangle[po]:
                    x, y = retangle[po]["xy"]
                    visibility = retangle[po]["vis"]
                    yolo_box.extend(
                        [round(x / img_width, 6), round(y / img_height, 6), visibility]
                    )
                else:
                    yolo_box.extend([0, 0, 0])

            class_id = classes.index(retangle["label"])
            f.write(
                f"{class_id} " + " ".join(map(lambda x: f"{x:.6f}", yolo_box)) + "\n"
            )


@cli.command()
def process_labelme_to_yolo_pose(
    image_path: Path = typer.Argument(..., help="图片目录"),
    class_path: str = typer.Argument(..., help="classes.txt"),
    label_path: Path = typer.Option(None, "--label_path", "-l", help="标签目录"),
    output_path: Path = typer.Option(None, "--output_path", "-o", help="输出目录"),
):
    images = [f for f in image_path.iterdir() if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS]
    label_path = label_path or image_path
    output_path = create_output_directory(output_path, image_path, "json2yolo_pose")

    classes = []
    with open(class_path, "r") as f:
        classes = f.read().splitlines()
        split_idx = classes.index("")
        classes, point_order = classes[:split_idx], classes[split_idx + 1 :]

    print("主体类别: ", classes)
    print("关键点顺序: ", point_order)

    for img_file in track(images, description="Converting to POSE..."):
        img = Image.open(img_file)
        base_name = img_file.stem
        json_file = label_path / f"{base_name}.json"
        txt_file = output_path / f"{base_name}.txt"

        if json_file.exists():
            convert_labelme_to_yolo(
                json_file, txt_file, classes, point_order, img.width, img.height
            )
        shutil.copy(img_file, output_path)

    shutil.copy(class_path, output_path / "classes.txt")
    show_result = input("是否要显示结果? (y/n): ")
    if show_result.lower() == "y":
        show(output_path, output_path / "classes.txt", output_path)


if __name__ == "__main__":
    cli()
