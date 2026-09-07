import asyncio
import concurrent.futures
import re
import time
from datetime import datetime
from pathlib import Path

import cv2
import requests
import schedule
import typer
from requests.auth import HTTPDigestAuth

cli = typer.Typer(rich_markup_mode="rich", help="海康摄像头定时抓拍")


def build_image_name(ip, preset):
    """生成抓拍文件名: {IP去点}_ISAPI_p{预置点}_{日期时间_毫秒}.jpg

    ISAPI 即海康摄像头的 HTTP 接口协议, 预置点调用走
    /ISAPI/PTZCtrl/channels/{channel}/presets/{preset}/goto
    """
    ip_safe = ip.replace(".", "_")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
    return f"{ip_safe}_ISAPI_p{preset}_{timestamp}.jpg"


def control_ptz(ip, auth, channel, preset, timeout):
    """调用摄像头预置点"""
    try:
        resp = requests.put(
            url=f"http://{ip}:80/ISAPI/PTZCtrl/channels/{channel}/presets/{preset}/goto",
            auth=auth,
            data=f"<PTZData><presetId>{preset}</presetId></PTZData>",
            headers={"Content-Type": "application/xml"},
            timeout=timeout,
        )
        success = resp.status_code == 200
        print(f"PTZ调用 {'成功' if success else '失败'}: {ip} (状态码: {resp.status_code})")
        return success
    except Exception as e:
        print(f"PTZ调用异常: {ip} - {e}")
        return False


def capture_camera(
    ip, user, password, output_dir, preset, ptz_timeout, ptz_wait, capture_timeout
):
    """捕获单个摄像头图像: 调预置点 -> 等待到位 -> RTSP 抓帧保存"""
    auth = HTTPDigestAuth(user, password)

    if not control_ptz(ip, auth, channel=1, preset=preset, timeout=ptz_timeout):
        return False

    time.sleep(ptz_wait)

    try:
        cap = cv2.VideoCapture(
            f"rtsp://{user}:{password}@{ip}:554/h264/ch1/main/av_stream"
        )
        start_time = time.time()
        ret, frame = False, None

        # 循环读取直到超时或成功
        while time.time() - start_time < capture_timeout and not ret:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)

        if ret and frame is not None:
            filename = build_image_name(ip, preset)
            cv2.imwrite(str(output_dir / filename), frame)
            print(f"图片保存成功: {filename}")
            cap.release()
            return True

        print(f"无法读取图像: {ip}")
        cap.release()
        return False

    except Exception as e:
        print(f"捕获图像异常: {ip} - {e}")
        return False


async def capture_all_cameras(
    ip_list, user, password, output_dir, preset, ptz_timeout, ptz_wait, capture_timeout, max_workers
):
    """异步并行捕获所有摄像头"""
    loop = asyncio.get_running_loop()
    success_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        tasks = [
            loop.run_in_executor(
                executor,
                capture_camera,
                ip,
                user,
                password,
                output_dir,
                preset,
                ptz_timeout,
                ptz_wait,
                capture_timeout,
            )
            for ip in ip_list
        ]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for ip, result in zip(ip_list, results):
            if isinstance(result, Exception):
                print(f"处理失败: {ip} - {result}")
            elif result:
                success_count += 1
                print(f"处理成功: {ip}")
            else:
                print(f"处理失败: {ip}")

    print(f"任务完成: 成功 {success_count}/{len(ip_list)}")
    return success_count


def run_capture_task(
    ip_list, user, password, output_dir, preset, ptz_timeout, ptz_wait, capture_timeout, max_workers
):
    """封装异步任务执行"""
    asyncio.run(
        capture_all_cameras(
            ip_list, user, password, output_dir, preset, ptz_timeout, ptz_wait, capture_timeout, max_workers
        )
    )


@cli.command()
def run_capture(
    ips: str = typer.Option("192.168.180.0", "--ips", help="摄像头 IP 列表(逗号分隔)"),
    directory: Path = typer.Option("images", "--dir", "-d", help="图片保存目录"),
    user: str = typer.Option("admin", "--user", "-u", help="摄像头用户名"),
    password: str = typer.Option("hxzh2019", "--password", help="摄像头密码"),
    preset: int = typer.Option(1, "--preset", "-p", help="PTZ 预置点编号"),
    cron_times: str = typer.Option(
        ":20,:50", "--cron", help="定时抓拍时间(逗号分隔, 每小时执行)"
    ),
    max_workers: int = typer.Option(10, "--max-workers", help="并发抓拍线程数"),
    ptz_timeout: int = typer.Option(5, "--ptz-timeout", help="PTZ 请求超时(秒)"),
    ptz_wait: int = typer.Option(2, "--ptz-wait", help="PTZ 到位等待时间(秒)"),
    capture_timeout: int = typer.Option(
        5, "--capture-timeout", help="RTSP 抓帧超时(秒)"
    ),
):
    """
    海康摄像头定时抓拍: 调用 PTZ 预置点后通过 RTSP 抓取一帧保存

    说明:
        1. 启动后立即抓拍一次, 之后按 --cron 指定的分钟在每小时定点抓拍
        2. 图片命名: {IP去点}_ISAPI_p{预置点}_{日期时间_毫秒}.jpg
           (ISAPI 即海康预置点调用接口)
        3. 多个摄像头并发抓拍 (--max-workers)
        4. 按下 Ctrl+C 停止

    使用示例:
        1. 【默认配置】抓拍 192.168.180.0, 每小时 :20/:50 执行
            python get_image.py

        2. 【多个摄像头与自定义目录】
            python get_image.py --ips 192.168.180.0,192.168.180.1 -d ./captures

        3. 【修改预置点】
            python get_image.py --preset 2

        4. 【自定义定时时间】
            python get_image.py --cron ":20,:50"
    """
    output_dir = directory
    output_dir.mkdir(parents=True, exist_ok=True)
    ip_list = [ip.strip() for ip in ips.split(",") if ip.strip()]
    cron_list = [t.strip() for t in cron_times.split(",") if t.strip()]

    # 校验定时时间格式 (MM:SS 或 :SS), 提前给出友好报错
    for t in cron_list:
        if not re.match(r"^([0-5]\d)?:[0-5]\d$", t):
            raise typer.BadParameter(
                f"非法定时时间 '{t}', 格式应为 MM:SS 或 :SS (如 :20, 12:30)"
            )

    # 立即执行一次
    run_capture_task(
        ip_list, user, password, output_dir, preset, ptz_timeout, ptz_wait, capture_timeout, max_workers
    )

    # 设置定时任务
    for cron_time in cron_list:
        schedule.every().hour.at(cron_time).do(
            run_capture_task,
            ip_list,
            user,
            password,
            output_dir,
            preset,
            ptz_timeout,
            ptz_wait,
            capture_timeout,
            max_workers,
        )

    print(f"调度器已启动, 将在每小时的{cron_list}执行捕获任务")

    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n程序已停止")


if __name__ == "__main__":
    cli()
