import subprocess
from pathlib import Path
from typing import Optional

import cv2
import typer
from rich.progress import Progress

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

def extract_frames_with_ffmpeg(video_path: Path, output_dir: Path, gap: int, video_name: str) -> bool:
    """使用ffmpeg提取帧"""
    try:
        output_pattern = f"{str(output_dir)}/{video_path.stem}_%05d.jpg"

        ffmpeg_command = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-vf",
            f"select='not(mod(n,{gap}))'",
            "-vsync",
            "vfr",
            output_pattern,
        ]

        subprocess.run(ffmpeg_command, check=True, capture_output=True, text=True)
        print(f"使用ffmpeg处理完成: {video_name}")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        print(f"ffmpeg处理失败: {e}")
        return False

def extract_frames_with_opencv(video_path: Path, output_dir: Path, gap: int, video_name: str) -> bool:
    """使用OpenCV提取帧"""
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            print(f"无法打开视频文件: {video_name}")
            return False

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        frame_count = 0
        saved_count = 0

        update_rate = int(total_frames/gap)
        with Progress() as progress:
            task = progress.add_task(f"[cyan]提取 {video_name[:15]}...[/cyan]", total=total_frames)
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                if frame_count % gap == 0:
                    progress.update(task, advance=update_rate)
                    output_file = output_dir / f"{video_path.stem}_{saved_count:05d}.jpg"
                    success = cv2.imwrite(str(output_file), frame)
                    if success:
                        saved_count += 1
                    else:
                        print(f"  - 警告: 无法保存帧 {saved_count:05d}")
                if (total_frames - frame_count) / gap < 1:
                    break
                frame_count += 1

        cap.release()
        print(f"使用OpenCV处理完成: {video_name}")
        print(f"  - 共处理 {frame_count} 帧")
        print(f"  - 保存 {saved_count} 张图片")
        return True

    except Exception as e:
        print(f"OpenCV处理失败: {e}")
        return False

@cli.command()
def extract_frames(
    path: str = typer.Argument(..., help="视频文件路径或包含视频文件的文件夹路径"),
    gap: int = typer.Option(50, "-gap", "-g", help="间隔多少帧保存一次"),
    output_path: Optional[Path] = typer.Option(None, "--output_path", "-o", help="输出目录"),
) -> None:
    """提取视频帧，默认使用ffmpeg，如果没有ffmpeg则使用OpenCV"""
    video_files = get_video_files_iterator(path)

    processed_count = 0
    found_files = False
    use_ffmpeg = None

    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        use_ffmpeg = True
    except (FileNotFoundError, subprocess.CalledProcessError):
        use_ffmpeg = False

    for video_file in video_files:
        if not found_files:
            found_files = True

        processed_count += 1
        print(f"\n正在处理第 {processed_count} 个视频文件: {video_file.name}")

        video_output_path = create_output_directory(
            output_path, str(video_file), f"video2img_{gap}_{video_file.stem}"
        )

        success = False

        if use_ffmpeg:
            success = extract_frames_with_ffmpeg(video_file, video_output_path, gap, video_file.name)
        else:
            success = extract_frames_with_opencv(video_file, video_output_path, gap, video_file.name)

        if success:
            print(f"视频 {video_file.name} 处理完成！")
            print(f"文件保存在: {video_output_path}")
        else:
            print(f"视频 {video_file.name} 处理失败！")

    print("="*60)
    if processed_count > 0:
        print(f"\n总共处理了 {processed_count} 个视频文件")
    elif not found_files:
        print("\n没有找到任何视频文件进行处理。")

if __name__ == "__main__":
    cli()
