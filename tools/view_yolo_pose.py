import cv2
import typer
from pathlib import Path
from rich.progress import track

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

# 类别调色板 (BGR), 相邻类别色差尽量拉大
PALETTE = [
    (255, 0, 0),      # 蓝
    (0, 215, 255),    # 黄
    (0, 128, 255),    # 橙
    (60, 180, 75),    # 绿
    (128, 0, 128),    # 紫
    (203, 192, 255),  # 粉
    (0, 0, 255),      # 红
    (200, 160, 0),    # 青
]

# 自动缩放窗口上限
MAX_WINDOW_WIDTH = 1280
MAX_WINDOW_HEIGHT = 960

cli = typer.Typer(rich_markup_mode="rich", help="实时预览 YOLO pose 标注")


def parse_pose_line(line):
    """解析一行 pose 标注: class_id cx cy w h [x y v ...]

    兼容两种写法:
        3 字段: x y v (v 为可见性, 2=已标注, 0=缺失)
        2 字段: x y (不带可见性, 默认视为已标注)
    """
    parts = line.strip().split()
    if len(parts) < 5:
        return None
    class_id = int(parts[0])
    box = tuple(float(x) for x in parts[1:5])
    rest = parts[5:]

    kps = []
    if len(rest) % 3 == 0:
        for i in range(0, len(rest), 3):
            kps.append((float(rest[i]), float(rest[i + 1]), int(float(rest[i + 2]))))
    elif len(rest) % 2 == 0:
        for i in range(0, len(rest), 2):
            kps.append((float(rest[i]), float(rest[i + 1]), 2))
    else:
        return None

    return class_id, box, kps


def draw_pose(img, class_id, box, kps, class_names, kp_name_list):
    """在图上绘制一个检测框及其关键点"""
    h, w = img.shape[:2]
    color = PALETTE[class_id % len(PALETTE)]

    # 检测框 (xywh -> x1y1x2y2)
    x1 = int((box[0] - box[2] / 2) * w)
    y1 = int((box[1] - box[3] / 2) * h)
    x2 = int((box[0] + box[2] / 2) * w)
    y2 = int((box[1] + box[3] / 2) * h)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)

    # 类别名
    label = class_names[class_id] if class_id < len(class_names) else str(class_id)
    cv2.putText(img, label, (x1, max(y1 - 6, 15)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # 关键点
    for i, (px, py, v) in enumerate(kps):
        cx, cy = int(px * w), int(py * h)
        if v == 0:
            # 缺失点: 灰色小叉
            cv2.drawMarker(img, (cx, cy), (180, 180, 180), cv2.MARKER_CROSS, 14, 2)
            continue
        # 已标注: 实心圆 + 外圈 + 名称
        cv2.circle(img, (cx, cy), 5, color, -1)
        cv2.circle(img, (cx, cy), 9, color, 1)
        name = kp_name_list[i] if i < len(kp_name_list) else str(i + 1)
        cv2.putText(img, name, (cx + 9, cy - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(img, name, (cx + 9, cy - 7), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


def load_name_file(path):
    """读取名称文件 (每行一个), 不存在时返回空列表"""
    if path is None:
        return []
    p = Path(path)
    if not p.exists():
        raise typer.BadParameter(f"文件不存在: {path}")
    return [line.strip() for line in p.read_text(encoding="utf-8").splitlines() if line.strip()]


def render_one(img_file, label_dir, class_names, kp_name_list):
    """读取单张图片并绘制标注, 看一张绘制一张"""
    img = cv2.imread(str(img_file))
    if img is None:
        return None

    txt = label_dir / f"{img_file.stem}.txt"
    if txt.exists():
        for line in txt.read_text(encoding="utf-8").splitlines():
            parsed = parse_pose_line(line)
            if parsed is None:
                continue
            class_id, box, kps = parsed
            draw_pose(img, class_id, box, kps, class_names, kp_name_list)
    return img


@cli.command()
def view_pose(
    images: Path = typer.Argument(..., help="图片目录(或单张图片)"),
    labels: Path = typer.Option(None, "--labels", "-l", help="txt 标签目录(默认与图片同目录)"),
    classes: Path = typer.Option(None, "--classes", "-c", help="classes.txt 路径(可选, 不传显示类别编号)"),
    kp_names: Path = typer.Option(None, "--kp-names", "-k", help="关键点名称文件(每行一个, 可选, 不传显示序号)"),
    scale: float = typer.Option(0.0, "--scale", help="显示缩放比例, 如 0.5 缩小一半 (默认自动适配窗口)"),
):
    """
    实时预览 YOLO pose 标注: 把检测框和关键点画到图片上逐张查看

    按键操作:
        a / ←   上一张
        d / →   下一张
        Esc / q 退出

    说明:
        1. 看一张绘制一张, 不预加载不保存
        2. 每行格式: class_id cx cy w h x1 y1 v1 x2 y2 v2 ...
           (x/y 为归一化坐标, v 为可见性: 2=已标注, 0=缺失)
        3. 兼容不带可见性的写法 (x y, 默认视为已标注)
        4. 已标注关键点画实心圆+名称, 缺失点画灰色小叉
        5. 每个类别一个颜色, 框左上角标注类别名
        6. 大图默认自动缩放适配窗口, --scale 可手动控制

    使用示例:
        1. 【实时预览】txt 与图片同目录
            python view_yolo_pose.py ./json2yolo_pose -c ./classes.txt

        2. 【标签目录分离 + 关键点名称】
            python view_yolo_pose.py ./images -l ./labels -c ./classes.txt -k ./kp_names.txt

        3. 【手动缩放预览】
            python view_yolo_pose.py ./images -c ./classes.txt --scale 0.5
    """
    class_names = load_name_file(classes)
    kp_name_list = load_name_file(kp_names)

    # 输入: 单张图片或目录
    if images.is_file():
        img_files = [images]
        label_dir = labels or images.parent
    else:
        img_files = [
            f
            for f in images.iterdir()
            if f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
        ]
        if not img_files:
            raise typer.BadParameter(f"目录中没有支持的图片: {images}")
        label_dir = labels or images

    # 只看带标签的 (仅文件级扫描, 不读图)
    img_files = [f for f in img_files if (label_dir / f"{f.stem}.txt").exists()]
    if not img_files:
        raise typer.BadParameter(f"没有找到任何带标签的图片: {images}")

    total = len(img_files)
    print(f"共 {total} 张带标注图片, 按 a/← 上一张, d/→ 下一张, Esc/q 退出")

    idx = 0
    window = "YOLO Pose Viewer"
    while True:
        img_file = img_files[idx]
        img = render_one(img_file, label_dir, class_names, kp_name_list)

        if img is None:
            print(f"跳过无法读取的图片: {img_file.name}")
            idx = (idx + 1) % total
            continue

        disp = img
        # 手动或自动缩放
        if scale > 0:
            s = scale
        else:
            h, w = img.shape[:2]
            s = min(1.0, MAX_WINDOW_WIDTH / w, MAX_WINDOW_HEIGHT / h)
        if s < 1.0:
            disp = cv2.resize(img, (int(img.shape[1] * s), int(img.shape[0] * s)))

        cv2.imshow(window, disp)
        cv2.setWindowTitle(window, f"{idx + 1}/{total} {img_file.name}")
        key = cv2.waitKey(0) & 0xFF

        if key in (ord("a"), 37):  # a 或 ←
            idx = (idx - 1) % total
        elif key in (ord("d"), 39):  # d 或 →
            idx = (idx + 1) % total
        elif key in (27, ord("q")):  # Esc 或 q
            break

    cv2.destroyAllWindows()
    print("预览结束")


if __name__ == "__main__":
    cli()
