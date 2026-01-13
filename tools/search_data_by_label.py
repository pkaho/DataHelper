import json
import shutil
import operator
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer
from rich.progress import track

cli = typer.Typer(
    help="""根据指定规则筛选 YOLO 或 LabelMe 标签文件，并移动/复制对应图像和标签"""
)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}
OPERATOR_MAP = {
    '>': operator.gt,
    '>=': operator.ge,
    '==': operator.eq,
    '!=': operator.ne,
    '<': operator.lt,
    '<=': operator.le
}


def find_files(path: Path):
    return list(path.glob("*.txt")) + list(path.glob("*.json"))


def load_labels(label_path: Path) -> Optional[Dict[str, int]]:
    labels = {}
    if label_path.suffix == ".txt":
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                class_id = parts[0]
                labels[class_id] = labels.get(class_id, 0) + 1
        return labels
    elif label_path.suffix == ".json":
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for shape in data.get('shapes', []):
            label = shape.get('label', '').strip()
            if label:
                labels[label] = labels.get(label, 0) + 1
        return labels
    return None


def get_image_path(label_path: Path) -> Optional[Path]:
    for ext in IMG_EXTS:
        img = label_path.with_suffix(ext)
        if img.exists():
            return img
    return None


def parse_pair_list(pair_list: List[str], op: str = ">=") -> Dict[str, Tuple[str, int]]:
    d = {}
    for item in pair_list:
        parts = item.split(':')
        if ':' not in item:
            cls, count_str = item, 1
        elif len(parts) == 2:
            cls, count_str = parts
        elif len(parts) == 3:
            cls, op, count_str = parts
        else:
            raise typer.BadParameter( f"格式错误，应为 'cls:op:count' 或 'cls:count', 但得到: '{item}'")

        if op not in OPERATOR_MAP.keys():
            raise typer.BadParameter(f"不支持的操作符 '{op}'，仅支持 >, >=, ==, !=, <, <=")

        try:
            count = int(count_str)
            if count < 0:
                raise ValueError
        except ValueError:
            #TODO: 以后或许可以直接强制 int(abs(count_str))
            raise typer.BadParameter(f"计数必须为正整数: '{item}'")

        d[cls] = (op, count)
    return d


def matches_rules(label_counts: Dict[str, int], **rules) -> bool:
    classes = set(label_counts.keys())
    total_count = sum(label_counts.values())

    if rules['any'] and classes.intersection(set(rules['any'])):
        return True

    if rules['all'] and set(rules['all']).issubset(classes):
        return True

    if rules['exact']:
        category_corr = False
        count_corr = True
        comp_dict = parse_pair_list(rules['exact'])

        if classes == set(comp_dict.keys()):
            category_corr = True

        for cls, (op, count) in comp_dict.items():
            label_count = label_counts.get(cls, 0)
            op_func = OPERATOR_MAP[op]
            if not op_func(label_count, count):
                count_corr = False

        if category_corr and count_corr:
            return True

    if rules['pairs']:
        pairs_dict = parse_pair_list(rules["pairs"])
        for cls, (op, count) in pairs_dict.items():
            label_count = label_counts.get(cls, 0)
            if OPERATOR_MAP[op](label_count, count):
                return True

    if rules['total'] and rules['op'] is not None:
        op = rules['op']
        if OPERATOR_MAP[op](total_count, rules['total']):
            return True

    return False


def safe_copy_or_move(src: Path, dst: Path, action: str):
    if action == "copy":
        shutil.copy2(src, dst)
    elif action == "move":
        shutil.move(str(src), str(dst))


@cli.command()
def main(
    input_path: Path = typer.Argument(..., help="包含标签文件（.txt/.json）和图像的输入目录"),
    output_path: Path = typer.Option(None, "--output_dir", "-o", help="匹配文件输出目录"),
    action: str = typer.Option("move", "--action", "-a", help="操作：'copy' 或 'move'"),
    include_labels: bool = typer.Option(True, "--include-labels/--exclude-labels", help="是否同时包含标签文件"),
    # 规则
    any: Optional[List[str]] = typer.Option(None, "--any", help="包含任一类别"),
    all: Optional[List[str]] = typer.Option(None, "--all", help="包含所有类别"),
    exact: Optional[List[str]] = typer.Option(None, "--exact", help="类别集合完全匹配"),
    pairs: Optional[List[str]] = typer.Option(None, "--pairs", help="类别:比较符:数量"),
    total: Optional[int] = typer.Option(None, "--total", help="类别总数, 配合 --op 使用"),
    op: Optional[str] = typer.Option(">=", "--op", help="总操作符"),
):
    input_path = input_path.resolve()
    output_path = output_path or input_path.parent / "search_data"
    output_path.mkdir(parents=True, exist_ok=True)

    matched_count = 0
    for label_file in track(find_files(input_path), description="Searching..."):
        labels = load_labels(label_file)
        if not labels:
            continue

        if not matches_rules(labels, any=any, all=all, exact=exact, pairs=pairs, total=total, op=op):
            continue

        img_file = next((label_file.with_suffix(ext)
                        for ext in IMG_EXTS
                        if label_file.with_suffix(ext).exists()), None)
        if img_file:
            safe_copy_or_move(img_file, output_path / img_file.name, action)

            if include_labels:
                safe_copy_or_move(label_file, output_path / label_file.name, action)

            matched_count += 1

    typer.secho(
        f"完成！共匹配 {matched_count} 个样本，已{'复制' if action == 'copy' else '移动'}至 {output_path}",
        fg=typer.colors.GREEN
    )


if __name__ == "__main__":
    cli()
