import random
import shutil
from pathlib import Path

import typer
from rich.progress import track

cli = typer.Typer()

IMAGE_FORMATS = [
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
]


@cli.command()
def split_dataset(
    image_path: Path = typer.Argument(..., help="图片目录"),
    label_path: Path = typer.Option(None, "--label_path", "-l", help="标签目录"),
    output_path: Path = typer.Option(None, "--output_path", "-o", help="输出目录"),
    ratio: float = typer.Option(0.1, "--ratio", "-r", help="分割比例(val集占比)"),
):
    output_path = output_path or image_path.resolve().parent / "splitdata"
    output_path.mkdir(parents=True, exist_ok=True)
    label_path = label_path or image_path

    train_image_dir = Path(output_path, "images", "train")
    val_image_dir = Path(output_path, "images", "val")
    train_label_dir = Path(output_path, "labels", "train")
    val_label_dir = Path(output_path, "labels", "val")

    train_image_dir.mkdir(parents=True, exist_ok=True)
    val_image_dir.mkdir(parents=True, exist_ok=True)
    train_label_dir.mkdir(parents=True, exist_ok=True)
    val_label_dir.mkdir(parents=True, exist_ok=True)

    image_list = [
        file for file in image_path.iterdir() if file.suffix in IMAGE_FORMATS
    ]
    random.shuffle(image_list)

    split_index = int(len(image_list) * ratio)

    train_files = image_list[split_index:]
    val_files = image_list[:split_index]

    for tr_image_file in track(train_files, description="SplitTrain..."):
        tr_label_file = label_path / Path(tr_image_file.name).stem
        tr_label_file = str(tr_label_file) + ".txt"
        shutil.copy(tr_image_file, train_image_dir)
        shutil.copy(tr_label_file, train_label_dir)

    for val_image_file in track(val_files, description="SplitVal..."):
        val_label_file = label_path / Path(val_image_file.name).stem
        val_label_file = str(val_label_file) + ".txt"
        shutil.copy(val_image_file, val_image_dir)
        shutil.copy(val_label_file, val_label_dir)

    typer.echo(f"Finished! file saved in {output_path}")


if __name__ == "__main__":
    cli()
