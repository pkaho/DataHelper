import json
import shutil
from pathlib import Path

import cv2
import numpy as np
import typer
from PIL import Image, ImageDraw
from rich.progress import track

from tools.show_pose import show

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
COLORS_RGB = [
    (255, 0, 0),  # 红色
    (0, 255, 0),  # 绿色
    (0, 0, 255),  # 蓝色
    (255, 255, 0),  # 黄色
    (0, 255, 255),  # 青色
    (255, 0, 255),  # 品红
    (0, 0, 0),  # 黑色
    (255, 255, 255),  # 白色
    (128, 128, 128),  # 灰色
    (255, 165, 0),  # 橙色
    (128, 0, 128),  # 紫色
    (255, 192, 203),  # 粉色
    (165, 42, 42),  # 棕色
    (128, 128, 0),  # 橄榄色
    (0, 0, 139),  # 深蓝色
    (135, 206, 235),  # 天蓝色
    (255, 127, 80),  # 珊瑚色
    (255, 215, 0),  # 金色
    (192, 192, 192),  # 银色
    (152, 255, 152),  # 薄荷绿
    (230, 230, 250),  # 薰衣草紫
    (183, 110, 121),  # 玫瑰金
    (0, 71, 171),  # 孔雀蓝
    (255, 219, 88),  # 芥末黄
    (86, 130, 3),  # 牛油果绿
    (176, 196, 222),  # 雾霾蓝
    (232, 180, 184),  # 脏粉色
]

cli = typer.Typer(help="关键点可视化，yolo 格式")


def draw_pose(pil_image, data, classes, point_order):
    """
    在 PIL 图像上绘制关键点检测结果

    Args:
        pil_image: PIL 图像对象
        data: 包含关键点检测结果的列表
        classes: 类别列表
        point_order: 关键点顺序列表
    """
    draw = ImageDraw.Draw(pil_image)
    width, height = pil_image.size

    for detection in data:
        parts = detection.strip().split()
        if len(parts) < 1:
            continue

        cls_id = int(parts[0])
        color = COLORS_RGB[cls_id % len(COLORS_RGB)]

        center_x = float(parts[1]) * width
        center_y = float(parts[2]) * height
        box_width = float(parts[3]) * width
        box_height = float(parts[4]) * height

        x1 = center_x - box_width / 2
        y1 = center_y - box_height / 2
        x2 = center_x + box_width / 2
        y2 = center_y + box_height / 2
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)

        label = f"{classes[cls_id]}" if cls_id < len(classes) else str(cls_id)
        draw.text((x1, y1 - 10), label, fill=color)

        keypoints = parts[5:]
        for i in range(0, len(keypoints), 3):
            if i + 2 >= len(keypoints):
                break

            x = float(keypoints[i]) * width
            y = float(keypoints[i + 1]) * height
            v = int(float(keypoints[i + 2]))
            draw.text((x, y), point_order[i // 3], fill=color)

            if v > 0:
                draw.ellipse([x - 3, y - 3, x + 3, y + 3], fill=color)

    return pil_image


@cli.command()
def show(
    image_path: Path = typer.Argument(..., help="图片目录"),
    class_path: Path = typer.Argument(
        ..., help="classes.txt, 目标分类和关键点分类(按实际顺序排列)中间用空行分隔"
    ),
    label_path: Path = typer.Option(None, "--label_path", "-l", help="标签目录"),
):
    label_path = image_path if label_path is None else label_path
    images = sorted(
        [
            f
            for f in image_path.iterdir()
            if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
    )
    if not images:
        print("No images found in the specified directory.")
        return

    with open(class_path, "r") as f:
        classes = f.read().splitlines()
        split_idx = classes.index("") if "" in classes else len(classes)
        classes, point_order = classes[:split_idx], classes[split_idx + 1 :]

    current_idx = 0
    while True:
        img_file = images[current_idx]
        base_name = img_file.stem
        txt_file = Path(label_path) / f"{base_name}.txt"

        pil_img = Image.open(img_file)

        if txt_file.exists():
            with open(txt_file, "r") as f:
                lines = f.readlines()
            pil_img = draw_pose(pil_img, lines, classes, point_order)
        else:
            print(f"Label file not found: {txt_file}")

        cv_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

        info_text = f"{img_file.name} ({current_idx + 1}/{len(images)})"
        cv2.putText(
            cv_img, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2
        )

        help_text = "Press: [a]prev [d]next [q]quit"
        cv2.putText(
            cv_img,
            help_text,
            (10, cv_img.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 255),
            2,
        )

        cv2.imshow("YOLO Pose Visualization", cv_img)

        key = cv2.waitKey(0) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("d"):
            current_idx = (current_idx + 1) % len(images)
        elif key == ord("a"):
            current_idx = (current_idx - 1) % len(images)

    cv2.destroyAllWindows()


def create_output_directory(output_dir, source_path, folder_name) -> Path:
    output_dir = output_dir or source_path.resolve().parent / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


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
