import json
import operator
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer
from rich.progress import track

from tools.utils import SUPPORTED_IMAGE_EXTENSIONS
from tools.utils import create_output_directory

cli = typer.Typer(rich_markup_mode="rich")

OPERATOR_MAPPING = {
    '>': operator.gt,
    '=': operator.eq,
    '<': operator.lt,
    '>=': operator.ge,
    '!=': operator.ne,
    '<=': operator.le
}


def find_files(search_dir: Path):
    txt_label_files = list(search_dir.glob("*.txt"))
    json_label_files = list(search_dir.glob("*.json"))
    return txt_label_files + json_label_files


def load_labels(label_file_path: Path) -> Optional[Dict[str, int]]:
    label_counts = {}
    if label_file_path.suffix == ".txt":
        with open(label_file_path, 'r') as f:
            for line in f:
                parts = line.strip().split() # 去除首位空白字符并按空格分割
                if not parts:
                    continue
                class_id = parts[0]
                label_counts[class_id] = label_counts.get(class_id, 0) + 1
        return label_counts

    elif label_file_path.suffix == ".json":
        with open(label_file_path, 'r', encoding='utf-8') as f:
            label_data = json.load(f)

        for shape in label_data.get('shapes', []):
            class_name = shape.get('label', '').strip()
            if class_name:
                label_counts[class_name] = label_counts.get(class_name, 0) + 1
        return label_counts

    return None


def get_corressponding_image_path(label_file_path: Path) -> Optional[Path]:
    for ext in SUPPORTED_IMAGE_EXTENSIONS:
        image_file_path = label_file_path.with_suffix(ext) # 替换文件扩展名
        if image_file_path.exists():
            return image_file_path

    return None


def parse_rule_pairs(rule_pairs: List[str], default_operator: str = '>=') -> Dict[str, Tuple[str, int]]:
    parsed_rules = {}
    for rule in rule_pairs:
        operator_symbol = default_operator # 无自定义操作符时，使用默认操作符

        rule_parts = rule.split(':')
        if len(rule_parts) == 1:
            class_name, count_str = rule, 1
        elif len(rule_parts) == 2:
            class_name, count_str = rule_parts
        elif len(rule_parts) == 3:
            class_name, operator_symbol, count_str = rule_parts
        else:
            raise typer.BadParameter(
                f"规则格式错误: '{rule}', 应为 '类别名', '类别名:数量', '类别名:操作符:数量'"
            )

        if operator_symbol not in OPERATOR_MAPPING.keys():
            raise typer.BadParameter(
                f"不支持的操作符: '{operator_symbol}', 仅支持: {list(OPERATOR_MAPPING.keys())}"
            )

        try:
            count = int(count_str)
            if count < 0:
                raise ValueError
        except ValueError:
            raise typer.BadParameter(f"规则 '{rule}' 中的数量来必须为正整数, 当前值: '{count_str}'")

        parsed_rules[class_name] = (operator_symbol, count)

    return parsed_rules


def check_rule_matching(label_counts: Dict[str, int], **rules) -> bool:
    current_classes = set(label_counts.keys()) # 当前包含的类别集合
    current_total_count = sum(label_counts.values()) # 当前标签总数量

    # 包含任一指定类别
    if rules.get('any'):
        any_rules = parse_rule_pairs(rules['any'], default_operator='>=')
        for class_name, (op, count) in any_rules.items():
            current_count = label_counts.get(class_name, 0)
            if OPERATOR_MAPPING[op](current_count, count):
                return True

    # 包含所有指定类别
    if rules.get('all'):
        all_rules = parse_rule_pairs(rules['all'], default_operator='>=')
        all_matched = True

        for class_name, (op, count) in all_rules.items():
            current_count = label_counts.get(class_name, 0)
            if not OPERATOR_MAPPING[op](current_count, count):
                all_matched = False
                break

        if all_matched:
            return True

    # 类别集合和数量完全匹配
    if rules.get('exact'):
        exact_rules = parse_rule_pairs(rules['exact'], default_operator='>=')

        # 检查类别集合是否完全一致
        if current_classes == set(exact_rules.keys()):
            exact_matched = True

            # 检查每个类别的数量是否满足条件
            for class_name, (op, count) in exact_rules.items():
                current_class_count = label_counts.get(class_name, 0)
                if not OPERATOR_MAPPING[op](current_class_count, count):
                    exact_matched = False
                    break

            if exact_matched:
                return True

    # 总标签数量满足阈值
    if rules.get('total'):
        total_rule = rules['total']
        virtual_total_rule = f"__TOTAL__:{total_rule}"

        total_rule = parse_rule_pairs([virtual_total_rule], default_operator='>=')
        op, count = total_rule['__TOTAL__']

        if OPERATOR_MAPPING[op](current_total_count, count):
            return True

    return False


def safe_copy_or_move(src: Path, dst: Path, action: str):
    if action == "copy":
        shutil.copy2(src, dst)
    elif action == "move":
        shutil.move(str(src), str(dst))


@cli.command()
def main(
    input_path: Path = typer.Argument(..., help="输入目录路径, 包含标签文件(.txt/.json)和图像"),
    output_path: Path = typer.Option(
        None,
        "--output_dir",
        "-o",
        help="匹配文件输出目录，未指定则在输入目录同级生成 search_data 文件夹"
    ),
    action: str = typer.Option("move", "--action", "-a", help="对匹配文件执行的操作：copy、move(默认)"),
    include_labels: bool = typer.Option(
        True,
        "--include-labels/--exclude-labels",
        help="是否同步处理标签文件(默认包含), 仅需处理图像时使用 --exclude-labels"
    ),
    # 规则
    any: Optional[List[str]] = typer.Option(None, "--any", help="匹配任一指定类别"),
    all: Optional[List[str]] = typer.Option(None, "--all", help="匹配所有指定类别"),
    exact: Optional[List[str]] = typer.Option(None, "--exact", help="精确匹配类别集合和数量"),
    total: Optional[int] = typer.Option(None, "--total", help="匹配总标签数量规则"),
):
    """
    根据指定的标签规则查找并处理对应的图像和标签文件

    匹配规则说明, 同时也是优先级顺序：
        1. --any: 任一规则满足即匹配（OR逻辑）
        2. --all: 所有规则都满足才匹配（AND逻辑）
        3. --exact: 类别集合和数量完全匹配（无额外类别）
        4. --total: 总标注数量满足指定条件

    使用示例:
        1. 【任一条件满足】查找包含至少2个dog标签 或 超过3个cat标签的数据
            python search_data_by_label.py ./data --any dog:>=:2 --any cat:>:3

        2. 【所有条件满足】查找同时包含至少2个dog标签 和 超过3个cat标签的数据
            python search_data_by_label.py ./data --all dog:>=:2 --all cat:>:3

        3. 【精确匹配类别+数量】查找仅包含cat和dog两类标签，且满足至少2个dog、超过3个cat的数据
            python search_data_by_label.py ./data --exact dog:>=:2 --exact cat:>:3

        4. 【总数量条件】查找所有标签的总数量大于5的数据
            python search_data_by_label.py ./data --total >:5
    """
    input_path = input_path.resolve()
    output_path = create_output_directory(output_path, input_path, "search_data")

    matched_count = 0
    for label_file in track(find_files(input_path), description="Searching..."):
        label_counts = load_labels(label_file)
        if not label_counts:
            continue

        if not check_rule_matching(label_counts, any=any, all=all, exact=exact, total=total):
            continue

        img_file = next((label_file.with_suffix(ext)
                        for ext in SUPPORTED_IMAGE_EXTENSIONS
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
