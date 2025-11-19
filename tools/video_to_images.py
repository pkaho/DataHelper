import subprocess
from pathlib import Path

import typer

cli = typer.Typer()

VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v', '.3gp', '.mpg', '.mpeg', '.ts'}

def create_output_directory(output_dir, source_path, folder_name):
    if output_dir is None:
        output_dir = Path(source_path).resolve().parent / folder_name
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def get_video_files_iterator(path: str):
    """优化的迭代器版本"""
    input_path = Path(path)

    if input_path.is_file():
        if input_path.suffix.lower() in VIDEO_EXTENSIONS:
            yield input_path
        else:
            print(f"文件 {path} 不是支持的视频格式")
        return

    # 对于目录，使用迭代器避免一次性加载所有文件
    for file_path in sorted(input_path.iterdir()):  # 保持排序
        if file_path.is_file() and file_path.suffix.lower() in VIDEO_EXTENSIONS:
            yield file_path

@cli.command()
def extract_frames(
    path: str = typer.Argument(..., help="视频文件路径或包含视频文件的文件夹路径"),
    gap: int = typer.Option(50, "-gap", "-g", help="间隔多少帧保存一次"),
    output_path: Path = typer.Option(None, "--output_path", "-o", help="输出目录"),
) -> None:
    video_files = get_video_files_iterator(path)

    processed_count = 0
    found_files = False

    for video_file in video_files:
        if not found_files:
            found_files = True

        processed_count += 1
        print(f"\n正在处理第 {processed_count} 个视频文件: {video_file.name}")

        video_output_path = create_output_directory(
            output_path, str(video_file), f"video2img_{gap}_{video_file.name}"
        )

        output_pattern = f"{str(video_output_path)}/{video_file.stem}_%05d.jpg"

        ffmpeg_command = [
            "ffmpeg",
            "-i",
            str(video_file),  # 确保路径是字符串
            "-vf",
            f"select='not(mod(n,{gap}))'",
            "-vsync",
            "vfr",
            output_pattern,
        ]

        try:
            subprocess.run(ffmpeg_command, check=True)
        except subprocess.CalledProcessError as e:
            print(f"处理视频 {video_file.name} 时出错: {e}")
        except FileNotFoundError:
            print("错误: 未找到 ffmpeg，请确保已安装并添加到系统路径")
            return

        print(f"视频 {video_file.name} 处理完成！文件保存在 {video_output_path}")

    if processed_count > 0:
        print(f"\n总共处理了 {processed_count} 个视频文件。")
    elif not found_files:
        print("\n没有找到任何视频文件进行处理。")

if __name__ == "__main__":
    cli()
