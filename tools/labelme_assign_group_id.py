import json
import shutil
from pathlib import Path

import typer
from rich.progress import track

cli = typer.Typer(rich_markup_mode="rich")


def normalize_rect(points):
    """将矩形两点对角坐标归一化为 (minx, miny, maxx, maxy)"""
    (x0, y0), (x1, y1) = points
    return (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))


def point_in_rect(point, rect):
    """判断点是否在矩形内, 边界算在内"""
    px, py = point["points"][0]
    minx, miny, maxx, maxy = normalize_rect(rect["points"])
    return minx <= px <= maxx and miny <= py <= maxy


def assign_group_ids(data, override):
    """为矩形和点分配 group_id, 返回 (是否正常, 信息)。

    返回:
    - 正常: (True, {"removed": 删除的重叠框label列表})
    - 异常: (False, {"multi": 多框包含的点, "orphan": 无框包含的点, "no_rect": 只有点没框})
    """
    shapes = data["shapes"]
    rects = [s for s in shapes if s["shape_type"] == "rectangle"]
    points = [s for s in shapes if s["shape_type"] == "point"]

    # 完全重叠的矩形删除一个 (保留第一个)
    removed_labels = []
    seen = []
    to_remove = set()
    for rect in rects:
        norm = normalize_rect(rect["points"])
        if any(norm == normalize_rect(r["points"]) for r in seen):
            to_remove.add(id(rect))
            removed_labels.append(rect["label"])
        else:
            seen.append(rect)
    if to_remove:
        shapes = [s for s in shapes if id(s) not in to_remove]
        data["shapes"] = shapes
        rects = [r for r in rects if id(r) not in to_remove]

    # 每个点找包含它的矩形
    multi_points, orphan_points = [], []
    point_rect = {}
    for p in points:
        containing = [r for r in rects if point_in_rect(p, r)]
        if len(containing) == 0:
            orphan_points.append(p["label"])
        elif len(containing) > 1:
            multi_points.append(p["label"])
        else:
            point_rect[id(p)] = containing[0]

    no_rect = len(rects) == 0 and len(points) > 0
    if multi_points or orphan_points or no_rect:
        return False, {
            "multi": multi_points,
            "orphan": orphan_points,
            "no_rect": no_rect,
            "removed": removed_labels,
        }

    # 分配 group_id: 有点的框与框内点同组, 没点的框保持不动
    existing = (
        set()
        if override
        else {s.get("group_id") for s in shapes if s.get("group_id") is not None}
    )
    next_gid = max(existing) + 1 if existing else 0

    for rect in rects:
        group_points = [p for p in points if point_rect.get(id(p)) is rect]
        if not group_points:
            continue

        if override or rect.get("group_id") is None:
            gid = next_gid
            next_gid += 1
            rect["group_id"] = gid
        else:
            gid = rect["group_id"]

        for p in group_points:
            if override or p.get("group_id") is None:
                p["group_id"] = gid

    return True, {"removed": removed_labels}


def find_image(label_path, stem):
    for img_file in label_path.iterdir():
        if img_file.is_file() and img_file.stem == stem and img_file.suffix.lower() != ".json":
            return img_file
    return None


@cli.command()
def labelme_assign_group_id(
    label_path: Path = typer.Argument(..., help="标注目录 (json + 图片)"),
    output_path: Path = typer.Option(None, "--output_path", "-o", help="输出目录, 不指定则原地修改"),
    exceptions_path: Path = typer.Option(
        None, "--exceptions", "-e", help="异常文件夹, 不指定则默认 <输出目录>/exceptions"
    ),
    override: bool = typer.Option(
        False, "--override", help="覆盖已有 group_id, 全部重新分配 (默认保留已有分组)"
    ),
):
    """
    根据空间包含关系为 LabelMe 标注中的矩形框和关键点分配 group_id

    分配规则说明:
        1. 遍历标注中的 rectangle 和 point, 判断关键点是否落在矩形内 (边界算框内)
        2. 每个包含关键点的矩形, 与框内所有关键点分配同一个 group_id (从 0 起递增)
        3. 没有包含任何关键点的矩形保持不动 (group_id 不变, 保持 null)
        4. 已有 group_id 的 shape 默认保留 (null 的关键点会继承所属矩形的 group_id),
           使用 --override 可忽略已有分组, 全部重新分配
        5. 完全重叠的矩形 (坐标完全相同) 自动删除一个并打印警告

    异常处理 (自动移动到异常文件夹, 需人工处理):
        1. 关键点同时落在多个矩形内 (如点落在嵌套的 circle 中心圆内)
        2. 关键点不在任何矩形内 (孤立点, 可能标出界或漏标框)
        3. 文件只有关键点没有矩形

    使用示例:
        1. 【原地分配】对 merged 目录的标注直接分配 group_id
            python labelme_assign_group_id.py ./merged

        2. 【输出到新目录】不修改原标注, 结果输出到 grouped 目录
            python labelme_assign_group_id.py ./merged -o ./grouped

        3. 【覆盖已有分组】忽略已有 group_id, 全部重新分配
            python labelme_assign_group_id.py ./merged --override

        4. 【自定义异常文件夹】异常数据移动到 ./manual_check
            python labelme_assign_group_id.py ./merged -e ./manual_check
    """
    output_path = output_path or label_path
    output_path.mkdir(parents=True, exist_ok=True)
    exceptions_path = exceptions_path or (output_path / "exceptions")
    exceptions_path.mkdir(parents=True, exist_ok=True)

    assigned_count = 0
    abnormal_files = []
    for json_file in track(sorted(label_path.glob("*.json")), description="Assigning group ids..."):
        with open(json_file, "r", encoding="utf-8") as f:
            data = json.load(f)

        ok, info = assign_group_ids(data, override)
        if not ok:
            # 异常: 整张图 (json + 图片) 挪到异常文件夹, 人工分配
            shutil.move(json_file, exceptions_path / json_file.name)
            img_file = find_image(label_path, json_file.stem)
            if img_file is not None:
                shutil.move(img_file, exceptions_path / img_file.name)
            abnormal_files.append((json_file.name, info))
            continue

        if info["removed"]:
            typer.echo(f"[warn] {json_file.name}: 删除完全重叠的矩形: {info['removed']}")

        with open(output_path / json_file.name, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        if output_path.resolve() != label_path.resolve():
            img_file = find_image(label_path, json_file.stem)
            if img_file is not None:
                shutil.copy(img_file, output_path)

        assigned_count += 1

    typer.echo(f"分配完成: {assigned_count} 个正常")
    if abnormal_files:
        typer.echo(f"异常 {len(abnormal_files)} 个, 已移动到 {exceptions_path}:")
        for name, info in abnormal_files:
            reasons = []
            if info["no_rect"]:
                reasons.append("只有点没有框")
            if info["multi"]:
                reasons.append(f"点被多个框包含: {info['multi']}")
            if info["orphan"]:
                reasons.append(f"点不在任何框内: {info['orphan']}")
            typer.echo(f"  {name}: {'; '.join(reasons)}")


if __name__ == "__main__":
    cli()
