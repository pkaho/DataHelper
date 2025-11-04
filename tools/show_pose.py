from pathlib import Path

import cv2
import numpy as np
import typer
from PIL import Image, ImageDraw

cli = typer.Typer()

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
IMAGE_FORMAT = [".jpeg", ".jpg", ".png", ".webp", ".tiff", ".bmp"]


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
        [f for f in Path(image_path).iterdir() if f.suffix.lower() in IMAGE_FORMAT]
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


if __name__ == "__main__":
    cli()
