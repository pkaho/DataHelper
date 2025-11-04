import json
import shutil
from enum import Enum
from pathlib import Path


import typer
from rich.progress import track

cli = typer.Typer()

IMAGE_FORMATS = {
    ".bmp",
    ".dng",
    ".jpeg",
    ".jpg",
    ".mpo",
    ".png",
    ".tif",
    ".tiff",
    ".webp",
    ".pfm",
}


class Mode(str, Enum):
    single = "single"
    nolabel = "nolabel"
    all = "all"


def create_output_directory(output_dir, source_path, folder_name) -> Path:
    if output_dir is None:
        output_dir = Path(source_path).resolve().parent / folder_name
    output_dir = Path(output_dir).resolve()
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
    img_dir = Path(image_path).resolve()
    label_dir = Path(label_path).resolve() if label_path else img_dir

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
        if f.is_file() and f.suffix.lower() in IMAGE_FORMATS
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
                processed += 2

    # 清理空输出目录
    for out_dir in [output_paths["single"], output_paths["nolabel"]]:
        if out_dir and out_dir.exists():
            if not any(out_dir.iterdir()):
                shutil.rmtree(out_dir)

    typer.echo(f"处理完成: 共检查 {len(image_files)} 张图像, 操作 {processed} 个文件")


if __name__ == "__main__":
    cli()
