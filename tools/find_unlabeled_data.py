import json
import shutil
from enum import Enum
from pathlib import Path

import typer
from rich.progress import track

SUPPORTED_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}

cli = typer.Typer(rich_markup_mode="rich", help="查找未/空标注数据")


class Mode(str, Enum):
    single = "single"
    nolabel = "nolabel"
    all = "all"


def create_output_directory(output_dir, source_path, folder_name) -> Path:
    output_dir = output_dir or source_path.resolve().parent / folder_name
    output_dir.mkdir(parents=True, exist_ok=True)

    return output_dir


def move_or_copy(src_file: Path, dst_path: Path, copy: bool) -> None:
    try:
        if copy:
            shutil.copy2(src_file, dst_path)
        else:
            shutil.move(src_file, dst_path)
    except OSError as e:
        print(f"无法处理 {src_file.name}: {e}")


def is_nolabel_file(label_file: Path) -> bool:
    if not label_file.exists() or label_file.stat().st_size == 0:
        return True

    try:
        with open(label_file, "r", encoding="utf-8") as f:
            if label_file.suffix == ".txt":
                lines = [line.strip() for line in f if line.strip()]
                if not lines:
                    return True
                # 每行应至少有5个字段（class + 4 coords）
                return any(len(line.split()) < 5 for line in lines)

            elif label_file.suffix == ".json":
                data = json.load(f)
                # 支持常见 YOLO-JSON 或 LabelMe 格式
                shapes = data.get("shapes", [])
                return not shapes
    except (json.JSONDecodeError, UnicodeDecodeError, OSError) as e:
        print(f"无法解析标签文件 {label_file}: {e}")
        return True  # 视为无效

    return False


@cli.command()
def process_data(
    image_path: Path = typer.Argument(..., help="图片目录"),
    label_path: Path = typer.Option(None, "--label_path", "-l", help="标签目录"),
    output_path: Path = typer.Option(None, "--output_path", "-o", help="输出目录"),
    copy: bool = typer.Option(False, "--copy", "-c", help="复制或是移动"),
    mode: Mode = typer.Option(
        Mode.all,
        "--mode",
        "-m",
        help="处理模式 [single: 没有标签文件, nolabel: 空标签文件, all: 同时两种]",
    ),
):
    """
    查找并处理未标注或空标注的数据

    说明:
        1. single 模式: 查找没有任何标签文件 (.txt/.json) 的图片
        2. nolabel 模式: 查找标签文件为空或无效 (txt 空文件/字段不足5, json 无 shapes) 的图片
        3. all 模式: 同时处理以上两种
        4. 默认移动, 使用 --copy 改为复制; 输出到图片目录同级的
           find_single / find_nolabel 文件夹
        5. nolabel 模式下标签文件会随图片一起移动/复制

    使用示例:
        1. 【查找无标签图片】移动到 find_single
            python find_unlabeled_data.py ./images --mode single

        2. 【查找空标签图片】标签在 labels 目录
            python find_unlabeled_data.py ./images -l ./labels --mode nolabel

        3. 【复制而非移动】同时处理两种模式, 复制到输出目录
            python find_unlabeled_data.py ./images --mode all --copy
    """
    img_dir = image_path.resolve()
    label_dir = label_path.resolve() if label_path else img_dir

    if not img_dir.is_dir():
        raise ValueError(f"图片路径不存在或不是目录: {img_dir}")
    if not label_dir.is_dir():
        raise ValueError(f"标签路径不存在或不是目录: {label_dir}")

    output_paths = {}
    if mode in (Mode.single, Mode.all):
        output_paths["single"] = create_output_directory(
            output_path, img_dir, "find_single"
        )
    if mode in (Mode.nolabel, Mode.all):
        output_paths["nolabel"] = create_output_directory(
            output_path, img_dir, "find_nolabel"
        )

    image_files = [
        f
        for f in img_dir.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    ]

    processed = 0
    for img_file in track(image_files, description="Processing images..."):
        stem = img_file.stem
        txt_label = label_dir / f"{stem}.txt"
        json_label = label_dir / f"{stem}.json"

        has_txt = txt_label.exists()
        has_json = json_label.exists()

        # 情况1: 无任何标签文件
        if not (has_txt or has_json):
            if output_paths["single"]:
                move_or_copy(img_file, output_paths["single"], copy)
                processed += 1
            continue

        # 情况2: 有标签但为空/无效
        label_file = txt_label if has_txt else json_label
        if is_nolabel_file(label_file):
            if output_paths["nolabel"]:
                move_or_copy(img_file, output_paths["nolabel"], copy)
                move_or_copy(label_file, output_paths["nolabel"], copy)
                processed += 1

    # 清理空输出目录
    for out_dir in [output_paths["single"], output_paths["nolabel"]]:
        if out_dir and out_dir.exists():
            if not any(out_dir.iterdir()):
                shutil.rmtree(out_dir)

    typer.echo(f"处理完成: 共检查 {len(image_files)} 张图像, 操作 {processed} 张图片")


if __name__ == "__main__":
    cli()
