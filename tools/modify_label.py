import json
import re
from enum import Enum
from pathlib import Path

import typer
from rich.progress import track

cli = typer.Typer()


class LabelFormat(Enum):
    TXT = ".txt"
    JSON = ".json"


def modify_txt(file, old_str, new_str, all_cls=None):
    with open(file, "r") as bf:
        lines = [i.strip() for i in bf if i.strip()]

    if old_str.isdigit():
        old_str_id = int(old_str)
    else:
        if all_cls is None:
            raise ValueError("classes.txt is required when using string labels!")
        old_str_id = all_cls.index(old_str)

    new_lines = []
    for line in lines:
        label = int(line.split(" ")[0])
        if label != old_str_id:
            new_lines.append(line)
            continue

        if new_str is None:  # 没有就跳过, 等效删除
            continue
        else:
            if new_str.isdigit():
                new_str_id = new_str
            else:
                if all_cls is None:
                    raise ValueError("classes.txt is required when using string labels!")
                new_str_id = str(all_cls.index(new_str))

        parts = line.split(" ")
        parts[0] = new_str_id
        new_line = " ".join(parts)
        new_lines.append(new_line)

    with open(file, "w") as f:
        f.write("\n".join(new_lines))

    return "Modification completed!"


def modify_json(file, old_str, new_str):
    with open(file, "r") as f:
        data = json.load(f)

    if new_str is None:
        data["shapes"] = [
            shape for shape in data["shapes"] if shape["label"] != old_str
        ]

    if new_str is not None:
        for shape in data["shapes"]:
            if shape["label"] == old_str:
                shape["label"] = new_str

    with open(file, "w") as f:
        json.dump(data, f, indent=4)


@cli.command()
def modify_label(
    path: Path = typer.Argument(..., help="标签目录"),
    old_str: str = typer.Argument(..., help="要替换或删除的旧标签名"),
    new_str: str = typer.Option(None, "--new_str", "-n", help="要替换的新标签名"),
    cls_path: str = typer.Option(None, "--cls_path", "-c", help="classes.txt"),
):
    if not path.exists():
        return f"{path} not found!"

    is_txt = LabelFormat.TXT.value
    is_json = LabelFormat.JSON.value

    all_cls = None
    if cls_path is not None:
        with open(cls_path, "r") as af:
            all_cls = [i.strip() for i in af if i.strip()]

    for label_file in track(path.iterdir(), description="Modify..."):
        if label_file.suffix == is_txt:
            if label_file.stem == "classes":
                continue
            modify_txt(label_file, old_str, new_str, all_cls)
        elif label_file.suffix == is_json:
            modify_json(label_file, old_str, new_str)

    typer.echo("Modification completed!")


if __name__ == "__main__":
    cli()
