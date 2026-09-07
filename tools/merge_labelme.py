import json
import shutil
from pathlib import Path

import typer
from rich.progress import track

cli = typer.Typer(rich_markup_mode="rich", help="合并两份 LabelMe 标注为一个 json")


def merge_labelme_json(first_data, second_data):
    """将两份 LabelMe 标注合并为一个 labelme 结构。

    - 两份标注的 shapes 全部保留并直接拼接 (框、关键点、分割等类型都合并)
    - group_id 不做任何修改或计算, 原样搬过去
    """
    merged_shapes = list(first_data["shapes"]) + list(second_data["shapes"])

    return {
        "version": first_data.get("version", ""),
        "flags": first_data.get("flags", {}),
        "shapes": merged_shapes,
        "imagePath": Path(first_data.get("imagePath", "")).name,
        "imageData": None,
        "imageHeight": first_data.get("imageHeight"),
        "imageWidth": first_data.get("imageWidth"),
    }


@cli.command()
def merge_labelme(
    first_path: Path = typer.Argument(..., help="第一份标注目录 (如 frames)"),
    second_path: Path = typer.Argument(..., help="第二份标注目录 (如 frames-kp)"),
    output_path: Path = typer.Option(
        None, "--output_path", "-o", help="输出目录, 不指定则直接合并到 first_path"
    ),
):
    """
    将两份 LabelMe 标注合并为一个 json (如框标注 + 关键点标注, 或未来的分割标注)

    说明:
        1. 两份标注的 shapes 全部保留并直接拼接 (框、关键点、分割等类型都合并)
        2. group_id 不做任何修改或计算, 原样搬过去
        3. 以 first_path 中同名 json 的字段为准 (图片宽高、路径等)
        4. 图片从 first_path 拷贝到输出目录
        5. 不指定 -o 时, 合并结果直接写入 first_path (覆盖原 json)

    使用示例:
        1. 【原地合并】结果写入 frames 目录, 覆盖原 json
            python merge_labelme.py ./frames ./frames-kp

        2. 【输出到新目录】不修改原标注
            python merge_labelme.py ./frames ./frames-kp -o ./merged
    """
    output_path = output_path or first_path
    output_path.mkdir(parents=True, exist_ok=True)

    first_jsons = sorted(first_path.glob("*.json"))
    merged_count = 0
    skipped = []
    for first_json in track(first_jsons, description="Merging labels..."):
        second_json = second_path / first_json.name
        if not second_json.exists():
            skipped.append(first_json.name)
            continue

        with open(first_json, "r", encoding="utf-8") as f:
            first_data = json.load(f)
        with open(second_json, "r", encoding="utf-8") as f:
            second_data = json.load(f)

        merged = merge_labelme_json(first_data, second_data)

        with open(output_path / first_json.name, "w", encoding="utf-8") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)

        for img_file in first_path.iterdir():
            if img_file.is_file() and img_file.stem == first_json.stem:
                img_dest = output_path / img_file.name
                if img_dest.resolve() != img_file.resolve():
                    shutil.copy(img_file, img_dest)
                break

        merged_count += 1

    typer.echo(f"合并完成: {merged_count} 个")
    if skipped:
        typer.echo(f"跳过 {len(skipped)} 个 (缺少对应的第二份标注): {skipped}")


if __name__ == "__main__":
    cli()
