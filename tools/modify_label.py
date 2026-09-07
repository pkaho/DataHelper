import json
from enum import Enum
from pathlib import Path

import typer
from rich.progress import track

cli = typer.Typer(rich_markup_mode="rich", help="修改标签")


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
                    raise ValueError(
                        "classes.txt is required when using string labels!"
                    )
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
    """
    批量修改标签文件中的类别 (支持 txt 和 json)

    说明:
        1. 支持修改 .txt (YOLO 格式) 和 .json (LabelMe 格式) 标签文件
        2. txt 文件: 通过类别 ID 或类别名定位 (使用类别名时需传 -c classes.txt)
        3. json 文件: 直接按 label 名称匹配
        4. 不传 -n 时表示删除该类别 (等效删除标注)
        5. 自动跳过 classes.txt 文件本身

    使用示例:
        1. 【按类别 ID 替换】把类别 0 改为类别 1 (txt 标签)
            python modify_label.py ./labels 0 -n 1

        2. 【按类别名替换】需要 classes.txt
            python modify_label.py ./labels dog -n cat -c ./classes.txt

        3. 【删除类别】删除所有 dog 标注
            python modify_label.py ./labels dog
    """
    if not path.exists():
        raise typer.BadParameter(f"{path} not found!")

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
