import json
import shutil
from pathlib import Path
from typing import Dict, List, Optional

import typer

cli = typer.Typer(
    help="根据指定规则筛选 YOLO 或 LabelMe 标签文件，并移动/复制对应图像和标签。"
)

IMG_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

def create_output_directory(output_dir, source_path, folder_name) -> Path:
    if output_dir is None:
        output_dir = Path(source_path).resolve().parent / folder_name
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir

def detect_label_format(label_path: Path) -> Optional[str]:
    if label_path.suffix == '.txt':
        return 'yolo'
    elif label_path.suffix == '.json':
        return 'labelme'
    return None


def load_yolo_labels(label_path: Path) -> Dict[str, int]:
    labels = {}
    try:
        with open(label_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if not parts:
                    continue
                # YOLO uses class index; we treat it as string (no class name mapping)
                class_id = parts[0]
                labels[class_id] = labels.get(class_id, 0) + 1
    except Exception as e:
        typer.secho(f"解析 YOLO 标签失败 {label_path}: {e}", fg=typer.colors.YELLOW)
    return labels


def load_labelme_labels(label_path: Path) -> Dict[str, int]:
    labels = {}
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        for shape in data.get('shapes', []):
            label = shape.get('label', '').strip()
            if label:
                labels[label] = labels.get(label, 0) + 1
    except Exception as e:
        typer.secho(f"解析 LabelMe 标签失败 {label_path}: {e}", fg=typer.colors.YELLOW)
    return labels


def get_image_path(label_path: Path) -> Optional[Path]:
    for ext in IMG_EXTS:
        img = label_path.with_suffix(ext)
        if img.exists():
            return img
    return None


def matches_rules(
    label_counts: Dict[str, int],
    any_classes: Optional[List[str]] = None,
    all_classes: Optional[List[str]] = None,
    exact_classes: Optional[List[str]] = None,
    exact_one_classes: Optional[List[str]] = None,
    gte_pairs: Optional[List[str]] = None,
    lte_pairs: Optional[List[str]] = None,
    total_gte: Optional[int] = None,
    total_lte: Optional[int] = None,
) -> bool:
    classes_in_label = set(label_counts.keys())
    total_instances = sum(label_counts.values())

    # Rule: --any
    if any_classes and not classes_in_label.intersection(set(any_classes)):
        return False

    # Rule: --all
    if all_classes and not set(all_classes).issubset(classes_in_label):
        return False

    # Rule: --exact (set equality, ignore counts)
    if exact_classes and classes_in_label != set(exact_classes):
        return False

    # Rule: --exact-one (set equality + each count == 1)
    if exact_one_classes:
        target_set = set(exact_one_classes)
        if classes_in_label != target_set:
            return False
        for cls in target_set:
            if label_counts.get(cls, 0) != 1:
                return False

    # Helper to parse "class:N" pairs
    def parse_pair_list(pair_list: List[str]) -> Dict[str, int]:
        d = {}
        for item in pair_list:
            if ':' not in item:
                raise typer.BadParameter(f"格式错误，应为 'class:count'，但得到 '{item}'")
            cls, cnt_str = item.split(':', 1)
            try:
                cnt = int(cnt_str)
                if cnt < 0:
                    raise ValueError
            except ValueError:
                raise typer.BadParameter(f"计数必须为非负整数：'{item}'")
            d[cls] = cnt
        return d

    # Rule: --gte (>=)
    if gte_pairs:
        gte_dict = parse_pair_list(gte_pairs)
        for cls, min_count in gte_dict.items():
            if label_counts.get(cls, 0) < min_count:
                return False

    # Rule: --lte (<=)
    if lte_pairs:
        lte_dict = parse_pair_list(lte_pairs)
        for cls, max_count in lte_dict.items():
            if label_counts.get(cls, 0) > max_count:
                return False

    # Rule: total count >=
    if total_gte is not None and total_instances < total_gte:
        return False

    # Rule: total count <=
    if total_lte is not None and total_instances > total_lte:
        return False

    return True


def safe_copy_or_move(src: Path, dst: Path, action: str):
    if action == "copy":
        shutil.copy2(src, dst)
    elif action == "move":
        shutil.move(str(src), str(dst))


@cli.command()
def main(
    input_path: Path = typer.Argument(..., help="包含标签文件（.txt/.json）和图像的输入目录"),
    action: str = typer.Option("move", "--action", help="操作：'copy' 或 'move'"),
    output_path: Path = typer.Option(None, "--output_dir", "-o", help="匹配文件输出目录"),
    copy_labels: bool = typer.Option(True, "--copy-labels/--no-copy-labels", help="是否同时复制/移动标签文件"),
    # rules
    any_classes: Optional[List[str]] = typer.Option(None, "--any", help="包含任意指定类别"),
    all_classes: Optional[List[str]] = typer.Option(None, "--all", help="必须包含所有指定类别"),
    exact_classes: Optional[List[str]] = typer.Option(None, "--exact", help="类别集合完全匹配（不限数量）"),
    exact_one_classes: Optional[List[str]] = typer.Option(None, "--exact-one", help="类别集合完全匹配且每类仅1个"),
    gte_pairs: Optional[List[str]] = typer.Option(None, "--gte", help="格式: class:count，要求 ≥ 该数量"),
    lte_pairs: Optional[List[str]] = typer.Option(None, "--lte", help="格式: class:count，要求 ≤ 该数量"),
    total_gte: Optional[int] = typer.Option(None, "--total-gte", help="总实例数 ≥ N"),
    total_lte: Optional[int] = typer.Option(None, "--total-lte", help="总实例数 ≤ N"),
):
    input_path = input_path.resolve()
    output_path = create_output_directory(output_path, input_path, "search_data")

    label_files = list(input_path.glob("*.txt")) + list(input_path.glob("*.json"))
    if not label_files:
        typer.secho("输入目录中未找到 .txt 或 .json 标签文件", fg=typer.colors.YELLOW)
        raise typer.Exit(0)

    matched_count = 0

    for label_path in label_files:
        fmt = detect_label_format(label_path)
        if not fmt:
            continue

        if fmt == 'yolo':
            label_counts = load_yolo_labels(label_path)
        else:
            label_counts = load_labelme_labels(label_path)

        if not label_counts:
            continue

        if not matches_rules(
            label_counts,
            any_classes=any_classes,
            all_classes=all_classes,
            exact_classes=exact_classes,
            exact_one_classes=exact_one_classes,
            gte_pairs=gte_pairs,
            lte_pairs=lte_pairs,
            total_gte=total_gte,
            total_lte=total_lte,
        ):
            continue

        img_path = get_image_path(label_path)
        if not img_path:
            typer.secho(f"未找到对应图像文件: {label_path.stem}", fg=typer.colors.YELLOW)
            continue

        # 执行操作
        safe_copy_or_move(img_path, output_path / img_path.name, action)
        if copy_labels:
            safe_copy_or_move(label_path, output_path / label_path.name, action)
            matched_count += 1

        matched_count += 1

    typer.secho(
        f"完成！共匹配 {matched_count} 个样本，已{'复制' if action == 'copy' else '移动'}至 {output_path}",
        fg=typer.colors.GREEN
    )


if __name__ == "__main__":
    cli()
