# TJ4DRadSet 视觉伪标签构建

本项目用于构建“视觉 3D 教师 -> 雷达坐标系伪标签 -> 雷达学生模型”的可复现实验流程。

项目已完成第一轮端到端基线：

1. 4D雷达、同步图像、标定和KITTI标签解析；
2. 相机/雷达坐标转换及双视图几何验证；
3. YOLO26s实例分割与Depth Anything V2户外公制深度；
4. 掩码反投影、雷达距离聚类校正与质量评分；
5. 可断点续跑的批量伪标签生成；
6. 统一雷达BEV学生模型的真值、伪标签和等量真值训练；
7. 在官方真值验证集上的2m/4m中心距离AP评测。

## 当前样例数据

完整数据统一路径：`D:\BaiduNetdiskDownload\TJ4DRadSet_Full`

其中 `velodyne/*.bin` 实际为4D雷达点云，每点8个float32特征：

```text
X, Y, Z, V_r, Range, Power, Alpha, Beta
```

## 环境

PyCharm解释器选择：

```text
C:\Users\lth\miniconda3\envs\radar_pseudo\python.exe
```

环境可通过 `environment.yml` 重建。已验证 PyTorch 2.11.0+cu128 可在 RTX 5060 Laptop GPU (`sm_120`) 上实际计算。

## 快速验证与可视化

在项目根目录运行：

```powershell
$env:PYTHONPATH = "src"
$python = "C:\Users\lth\miniconda3\envs\radar_pseudo\python.exe"
& $python -m pytest -q
& $python -m radar_pseudo.visualize --dataset-root D:\BaiduNetdiskDownload\TJ4DRadSet_Full --frame-id 000000
```

## 单帧伪标签

```powershell
& $python -m radar_pseudo.run_pseudo `
  --dataset-root D:\BaiduNetdiskDownload\TJ4DRadSet_Full `
  --frame-id 000000
```

## 批量生成

```powershell
& $python -m radar_pseudo.batch_generate `
  --dataset-root D:\BaiduNetdiskDownload\TJ4DRadSet_Full `
  --split-file D:\BaiduNetdiskDownload\TJ4DRadSet_Full\ImageSets\train.txt `
  --output-root outputs\pseudo_q06_train `
  --quality-threshold 0.6
```

输出包括与官方标签相同15字段格式的 `label_2/*.txt`，以及保存置信度、雷达支持和质量分数的 `metadata/*.json`。

## 实验结果

第一轮实验的完整配置、结果和限制见 [docs/results_baseline.md](docs/results_baseline.md)。当前结果表明：雷达联合校正显著改善3D中心定位，车辆伪标签有效性较好，但弱类召回不足，尚未达到整体替代真值标签的水平。

## 伪标签可视化对比

每张图按四个视图展示：图像平面真值/伪标签投影、雷达鸟瞰图叠加、3D点云真值框、3D点云伪标签框。红色实线表示真值，青色虚线表示伪标签。

![帧250091的点云与伪标签对比](docs/assets/pseudo-comparison-250091.png)

![帧030187的点云与伪标签对比](docs/assets/pseudo-comparison-030187.png)

![帧250176的点云与伪标签对比](docs/assets/pseudo-comparison-250176.png)

仓库不包含TJ4DRadSet原始数据、批量输出和模型权重。上述示例含数据集相机画面，因此远程仓库默认保持私有。
