# 项目名称

图像识别任务的数据处理工具

---

## 目录结构

```
.
├── detect.py                   # 主检测脚本（基于 YOLO 模型）
├── models/                     # 模型权重文件
│   └── <your mode>.pt          # 预训练/微调后的模型权重
├── tools/                      # 数据处理与格式转换工具集
│   ├── find_unlabeled_data.py       # 查找未标注数据
│   ├── generate_empty_label_file.py # 生成空标签文件
│   ├── labelme_to_yolo_det.py       # LabelMe 转 YOLO 目标检测格式
│   ├── labelme_to_yolo_pose.py      # LabelMe 转 YOLO 姿态估计格式
│   ├── labelme_to_yolo_seg.py       # LabelMe 转 YOLO 分割格式
│   ├── modify_label.py              # 修改标签内容
│   ├── show_pose.py                 # 可视化姿态标注
│   ├── splitdata.py                 # 划分训练/验证/测试集
│   ├── video_to_images.py           # 视频抽帧为图像
│   └── yolo_det_to_labelme.py       # YOLO 检测结果转回 LabelMe 格式
├── pyproject.toml          # 项目依赖与构建配置（兼容 Poetry / uv 等）
└── pyrightconfig.jsonc     # Pyright 类型检查配置
```

---

## 快速开始

### 1. 安装依赖

使用 [`uv`](https://github.com/astral-sh/uv) 进行快速依赖管理，根据情况选择命令：

```bash
uv sync --extra cu118
uv sync --extra cpu
```

### 2. 使用工具脚本

所有数据处理工具均位于 `tools/` 目录，使用 `--help` 查看详细用法

```bash
uv run tools/xxx.py --help
```

