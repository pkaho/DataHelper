import subprocess
from pathlib import Path

import cv2
import typer

cli = typer.Typer()


def create_output_directory(output_dir, source_path, folder_name):
    if output_dir is None:
        output_dir = Path(source_path).resolve().parent / folder_name
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


@cli.command()
def extract_frames(
    path: str = typer.Argument(..., help="视频路径"),
    gap: int = typer.Option(50, "-gap", "-g", help="间隔多少帧保存一次"),
    output_path: Path = typer.Option(None, "--output_path", "-o", help="输出目录"),
) -> None:
    cap = cv2.VideoCapture(path)
    fps = cap.get(7)
    name_length = len(str(int(fps)))

    basename = Path(path).name
    output_path = create_output_directory(
        output_path, path, f"video2img_{gap}_{basename}"
    )

    try:
        input_path = Path(path)
        if not input_path.is_file():
            raise FileNotFoundError(f"输入的视频文件 {path} 不存在。")

        ffmpeg_command = [
            "ffmpeg",
            "-i",
            path,
            "-vf",
            f"select='not(mod(n,{gap}))'",
            "-vsync",
            "vfr",
            f"{str(output_path)}/%{name_length}d.jpg",
        ]

        print("执行的 FFmpeg 命令：", " ".join(ffmpeg_command))

        subprocess.run(ffmpeg_command, check=True)
    except FileNotFoundError as file_err:
        print(f"文件错误：{file_err}")
    except subprocess.CalledProcessError as process_err:
        print(f"执行 FFmpeg 命令时出错：{process_err}")
    except Exception as general_err:
        print(f"发生未知错误：{general_err}")

    typer.echo(f"Finished! file saved in {output_path}")


if __name__ == "__main__":
    cli()
