import random
import shutil
from pathlib import Path

import typer
from rich.progress import track

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


cli = typer.Typer(rich_markup_mode="rich", help="划分数据集")


def create_output_directory(output_dir, source_path, folder_name) -> Path:
    output_dir = output_dir or source_path.resolve().parent / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


@cli.command()
def split_dataset(
    image_path: Path = typer.Argument(..., help="图片目录"),
    label_path: Path = typer.Option(None, "--label_path", "-l", help="标签目录"),
    output_path: Path = typer.Option(None, "--output_path", "-o", help="输出目录"),
    ratio: float = typer.Option(0.1, "--ratio", "-r", help="验证集占比(默认0.1)"),
):
    """
    将数据集按比例随机划分为 train/val 两套 (images 与 labels 目录结构, 供 YOLO 训练使用)

    说明:
        1. 遍历图片目录, 随机打乱后按 ratio 比例划分验证集 (其余为训练集)
        2. 标签目录与图片目录可以分离, 标签按同名查找, 支持 .txt 或 .json
        3. 输出目录结构: images/train, images/val, labels/train, labels/val
        4. 没有对应标签文件的图片仅拷贝图片
        5. 默认输出到图片目录同级的 splitdata 文件夹

    使用示例:
        1. 【基本划分】验证集占 10%
            python splitdata.py ./images

        2. 【标签目录分离】标签在 labels 目录
            python splitdata.py ./images -l ./labels

        3. 【自定义比例与输出】验证集占 20%, 输出到 ./split
            python splitdata.py ./images -r 0.2 -o ./split
    """
    if not 0 < ratio < 1:
        raise typer.BadParameter("ratio 必须在 0 到 1 之间")

    label_path = label_path or image_path
    output_path = create_output_directory(output_path, image_path, "splitdata")

    train_image_dir = Path(output_path, "images", "train")
    val_image_dir = Path(output_path, "images", "val")
    train_label_dir = Path(output_path, "labels", "train")
    val_label_dir = Path(output_path, "labels", "val")

    for d in (train_image_dir, val_image_dir, train_label_dir, val_label_dir):
        d.mkdir(parents=True, exist_ok=True)

    image_list = [
        file
        for file in image_path.iterdir()
        if file.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]
    random.shuffle(image_list)

    split_index = int(len(image_list) * ratio)
    train_files = image_list[split_index:]
    val_files = image_list[:split_index]

    def copy_with_label(img_files, img_dir, label_dir, description):
        for img_file in track(img_files, description=description):
            shutil.copy(img_file, img_dir)
            txt_label = label_path / f"{img_file.stem}.txt"
            json_label = label_path / f"{img_file.stem}.json"
            if txt_label.exists():
                shutil.copy(txt_label, label_dir)
            elif json_label.exists():
                shutil.copy(json_label, label_dir)

    copy_with_label(train_files, train_image_dir, train_label_dir, "SplitTrain...")
    copy_with_label(val_files, val_image_dir, val_label_dir, "SplitVal...")

    typer.echo(
        f"Finished! train: {len(train_files)}, val: {len(val_files)}, saved in {output_path}"
    )


if __name__ == "__main__":
    cli()
