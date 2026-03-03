import asyncio
import cv2
import requests
from requests.auth import HTTPDigestAuth
from datetime import datetime
import os
import schedule
import time
import concurrent.futures

# 配置项（集中管理常量，便于维护）
CONFIG = {
    "IMG_DIR": "images",
    "AUTH": HTTPDigestAuth("admin", "hxzh2019"),
    "PTZ_TIMEOUT": 5,
    "CAPTURE_TIMEOUT": 5,
    "PTZ_WAIT": 2,
    "MAX_WORKERS": 10,
    "IP_LIST": [f"192.168.180.{i}" for i in range(0, 1)],
    "CRON_TIMES": [":20", ":50"]
}

# 初始化目录
os.makedirs(CONFIG["IMG_DIR"], exist_ok=True)

def control_ptz(ip, channel=1, preset=1):
    """调用摄像头预置点（简化函数名，精简逻辑）"""
    try:
        resp = requests.put(
            url=f"http://{ip}:80/ISAPI/PTZCtrl/channels/{channel}/presets/{preset}/goto",
            auth=CONFIG["AUTH"],
            data=f"<PTZData><presetId>{preset}</presetId></PTZData>",
            headers={"Content-Type": "application/xml"},
            timeout=CONFIG["PTZ_TIMEOUT"]
        )
        success = resp.status_code == 200
        print(f"PTZ调用 {'成功' if success else '失败'}: {ip} (状态码: {resp.status_code})")
        return success
    except Exception as e:
        print(f"PTZ调用异常: {ip} - {str(e)}")
        return False

def capture_camera(ip):
    """捕获单个摄像头图像（精简逻辑，合并重复判断）"""
    # 1. 调用预置点
    if not control_ptz(ip):
        return False

    # 2. 等待摄像头到位
    time.sleep(CONFIG["PTZ_WAIT"])

    # 3. 捕获图像
    try:
        cap = cv2.VideoCapture(f"rtsp://admin:hxzh2019@{ip}:554/h264/ch1/main/av_stream")
        start_time = time.time()
        ret, frame = False, None

        # 循环读取直到超时或成功
        while time.time() - start_time < CONFIG["CAPTURE_TIMEOUT"] and not ret:
            ret, frame = cap.read()
            if not ret:
                time.sleep(0.1)

        # 保存图片
        if ret and frame is not None:
            filename = f"{ip}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]}.jpg"
            cv2.imwrite(os.path.join(CONFIG["IMG_DIR"], filename), frame)
            print(f"图片保存成功: {filename}")
            cap.release()
            return True

        print(f"无法读取图像: {ip}")
        cap.release()
        return False

    except Exception as e:
        print(f"捕获图像异常: {ip} - {str(e)}")
        return False

async def capture_all_cameras(ip_list):
    """异步并行捕获所有摄像头（精简任务创建和结果统计）"""
    loop = asyncio.get_running_loop()
    success_count = 0

    with concurrent.futures.ThreadPoolExecutor(max_workers=CONFIG["MAX_WORKERS"]) as executor:
        # 批量创建任务并执行
        tasks = [loop.run_in_executor(executor, capture_camera, ip) for ip in ip_list]
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # 统计结果
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

def run_capture_task():
    """封装异步任务执行逻辑（简化调用）"""
    asyncio.run(capture_all_cameras(CONFIG["IP_LIST"]))

def main():
    """主函数（精简定时任务配置）"""
    # 立即执行一次
    run_capture_task()

    # 设置定时任务
    for cron_time in CONFIG["CRON_TIMES"]:
        schedule.every().hour.at(cron_time).do(run_capture_task)

    print(f"调度器已启动，将在每小时的{CONFIG['CRON_TIMES']}执行捕获任务")

    # 运行调度器
    try:
        while True:
            schedule.run_pending()
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n程序已停止")

if __name__ == "__main__":
    # 修改 CONFIG["IP_LIST"]
    main()
