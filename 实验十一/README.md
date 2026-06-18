# 计算机视觉实验：基于MediaPipe + Skeleton Transformer 的羽毛球击球动作识别
[![Python 3.12](https://img.shields.io/badge/Python-3.12-blue.svg)]
[![PyTorch 2.x](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)]
[![MediaPipe 0.10.21](https://img.shields.io/badge/MediaPipe-0.10.21-green.svg)]
[![OpenCV 4.9.0](https://img.shields.io/badge/OpenCV-4.9.0-red.svg)]
[![License](https://img.shields.io/badge/License-Course%20Experiment-lightgrey.svg)]

## 一、项目概述
### 1.1 实验背景
传统视频动作识别直接将原始RGB视频输入网络，存在计算量大、显存占用高、背景信息冗余等问题，不适合普通CPU/WSL教学环境。
本实验采用**轻量化骨架时序方案**：
> 原始视频 → MediaPipe Pose 提取人体33关键点 → 标准化骨架时间序列 → Transformer Encoder 时序分类

仅保留与动作相关的人体运动特征，大幅压缩数据维度，CPU即可完成完整的训练、测试、推理全流程，兼顾可解释性与运行效率。

### 1.2 实验目标
1. 掌握 MediaPipe Pose 逐帧提取人体 33 个关键点的方法，理解骨架特征的构成；
2. 实现视频帧重采样、骨架坐标平移+尺度归一化的完整预处理流程；
3. 理解「单帧骨架作为时序Token」的建模思路，搭建轻量 Skeleton Transformer；
4. 完成 6 类羽毛球击球动作的分类训练、测试评估、混淆矩阵分析；
5. 实现单视频端到端推理，输出动作类别与置信度；
6. 总结骨架时序方案在动作识别任务中的优势与局限性。

## 二、数据集说明
数据集来源：Kaggle `badminton_storke_video`
数据集链接：https://www.kaggle.com/datasets/shenhuichang/badminton-storke-video

包含 6 类标准羽毛球击球动作短视频：

| 标签ID | 英文类别 | 中文说明 |
|--------|----------|----------|
| 0 | forehand drive | 正手平抽 |
| 1 | forehand lift | 正手挑球 |
| 2 | forehand net shot | 正手网前小球 |
| 3 | forehand clear | 正手高远球 |
| 4 | backhand drive | 反手平抽 |
| 5 | backhand net shot | 反手网前小球 |

## 三、整体技术流水线
```
Kaggle 羽毛球原始视频
        ↓ OpenCV 逐帧读取
MediaPipe Pose 提取单帧 33 个人体关键点
        ↓ 每帧展平为 33×4 = 132 维特征 (x,y,z,visibility)
        ↓ 时序统一重采样至固定 30 帧
        ↓ 骨架归一化（双髋中心平移 + 肩宽尺度缩放）
        ↓ 8:2 划分训练/测试集，保存为 .npy 时序文件
        ↓ Skeleton Transformer Encoder 时序分类模型
        ↓ 模型训练 + 测试集评估（准确率/混淆矩阵/分类报告）
        ↓ 单段视频端到端推理，输出动作类别与置信度
```

## 四、项目目录结构
```
experiment11/
├── raw_video/              # 原始视频数据集（6个类别子文件夹）
│   ├── forehand_drive/
│   ├── forehand_lift/
│   ├── forehand_net/
│   ├── forehand_clear/
│   ├── backhand_drive/
│   └── backhand_net/
├── skeleton_npy/           # 预处理输出：骨架时序数据集
│   ├── X_train.npy         # 训练集特征 (N_train, 30, 132)
│   ├── y_train.npy         # 训练集标签 (N_train,)
│   ├── X_test.npy          # 测试集特征 (N_test, 30, 132)
│   ├── y_test.npy          # 测试集标签 (N_test,)
│   └── label_map.json      # 标签ID与动作名称映射
├── model_output/           # 训练好的模型权重
│   └── badminton_transformer.pth
├── pic_output/             # 可视化结果图片
│   ├── train_curve.png     # 训练/测试 Loss & Accuracy 曲线
│   └── confusion_matrix.png# 测试集混淆矩阵热力图
├── preprocess.py           # 【脚本1】视频转骨架时序预处理
├── train.py                # 【脚本2】Skeleton Transformer 训练+评估
├── infer.py                # 【脚本3】单视频端到端推理
├── demo.mp4                # 推理测试视频（自行放入根目录）
├── README.md               # 项目说明文档
└── 实验报告.docx           # 课程实验报告
```

## 五、环境部署
### 5.1 激活虚拟环境
本项目使用 `.venv-dl` 虚拟环境，WSL 终端执行：
```bash
cd /home/hql/cv-course
source .venv-dl/bin/activate
```
激活成功标识：终端行首出现 `(.venv-dl)`

### 5.2 安装依赖（已修复版本冲突）
以下版本经过验证，无 numpy/mediapipe/opencv 兼容性问题：
```bash
# 安装核心依赖（指定版本，避免兼容问题）
pip install mediapipe==0.10.21 opencv-python==4.9.0.80 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 安装其他依赖
pip install torch torchvision numpy matplotlib seaborn scikit-learn -i https://pypi.tuna.tsinghua.edu.cn/simple
```

> 说明：
> - `mediapipe==0.10.21`：最后一个保留 `mp.solutions` 旧接口的版本，适配 Python 3.12；
> - `opencv-python==4.9.0.80`：兼容 numpy 1.x，无版本冲突；
> - 清华镜像源解决 WSL 网络超时问题。

## 六、分步运行指南
### 步骤1：准备数据集
1. 从 Kaggle 下载羽毛球视频数据集；
2. 解压后将 6 个类别文件夹完整放入 `raw_video/` 目录；
3. 支持视频格式：`.mp4 / .avi / .mov / .mkv`。

### 步骤2：执行视频骨架预处理
```bash
cd experiment11
python preprocess.py
```
**执行效果**：
- 逐类统计视频数量，逐帧提取人体关键点；
- 自动完成时序重采样、骨架归一化；
- 按 8:2 划分训练/测试集，输出至 `skeleton_npy/` 目录；
- 自动生成标签映射文件 `label_map.json`。

### 步骤3：模型训练与评估
```bash
python train.py
```
**执行效果**：
- 逐轮打印训练损失、训练准确率、测试损失、测试准确率；
- 自动保存测试精度最高的模型权重至 `model_output/`；
- 训练结束自动生成训练曲线、混淆矩阵图片至 `pic_output/`；
- 终端打印完整分类报告（精确率、召回率、F1分数）。

> 代码已修复：matplotlib subplot 参数错误、动态适配实际类别数量、WSL 无图形界面自动存图。

### 步骤4：单视频推理
1. 将待测试的羽毛球视频重命名为 `demo.mp4`，放在项目根目录（与 `infer.py` 同级）；
2. 执行推理脚本：
```bash
python infer.py
```
**输出示例**：
```
正在处理视频：demo.mp4
========================================
Predicted class: forehand clear
Confidence: 0.87
========================================
```

## 七、模型结构与超参数
### 7.1 网络结构
输入形状：`[Batch, 30, 132]`（批次 × 时序帧数 × 单帧骨架维度）
1. **线性嵌入层**：132 维骨架特征 → 128 维模型隐藏维度
2. **正弦时序位置编码**：注入帧的先后时序信息
3. **2层 Transformer Encoder**：4头注意力 + 前馈网络 + Dropout
4. **全局均值池化**：聚合 30 帧的全部时序特征
5. **MLP 分类头**：128 → 64 → 6，输出 6 类动作的 logits

### 7.2 核心超参数
| 参数项 | 取值 | 说明 |
|--------|------|------|
| input_dim | 132 | 单帧骨架特征维度（33点×4特征） |
| target_frames | 30 | 统一视频时序长度 |
| d_model | 128 | Transformer 基础维度 |
| nhead | 4 | 多头注意力头数 |
| num_layers | 2 | Encoder 堆叠层数 |
| dim_feedforward | 256 | FFN 隐藏层维度 |
| num_classes | 6 | 羽毛球动作总类别数 |
| dropout | 0.1 | 随机失活，防止过拟合 |
| batch_size | 16 | 训练批次大小 |
| learning_rate | 1e-3 | Adam 优化器学习率 |
| epochs | 20 | 总训练轮数 |
| loss | CrossEntropyLoss | 多分类标准损失 |
| optimizer | Adam | 自适应学习率优化器 |

## 八、输出文件说明
| 目录 | 文件 | 说明 |
|------|------|------|
| skeleton_npy/ | X_train.npy / y_train.npy | 训练集骨架序列与标签，可复用 |
| skeleton_npy/ | X_test.npy / y_test.npy | 测试集骨架序列与标签 |
| skeleton_npy/ | label_map.json | 标签ID与动作名称映射 |
| model_output/ | badminton_transformer.pth | 最优测试精度模型权重 |
| pic_output/ | train_curve.png | 训练/测试损失、准确率曲线 |
| pic_output/ | confusion_matrix.png | 测试集混淆矩阵热力图 |

## 九、实验结论
### 9.1 方案优势
1. **数据轻量化**：剔除背景冗余信息，仅保留人体运动骨架，CPU即可完成训练；
2. **时序建模能力强**：Transformer 自注意力可捕捉长距离的挥拍时序依赖（引拍-击球-收拍完整过程）；
3. **鲁棒性好**：骨架归一化消除了人物位置、画面远近的干扰，泛化性更强；
4. **可解释性高**：可可视化每一帧的人体关键点，便于定位模型误判的原因。

### 9.2 方案局限
1. **遮挡敏感**：肢体遮挡时 MediaPipe 关键点会漂移，引入时序噪声；
2. **缺少小球信息**：仅使用人体骨架，丢失羽毛球的位置轨迹，细微动作区分难度大；
3. **固定采样损失细节**：统一 30 帧重采样会丢失高速挥拍的瞬时动作细节。

### 9.3 优化方向
1. 对骨架序列做平滑滤波，降低帧间关键点抖动；
2. 融合羽毛球目标检测特征，补充小球运动轨迹；
3. 加入时序数据增强（随机帧扰动、时序缩放）扩充训练样本；
4. 采用「时序CNN + Transformer」混合结构，兼顾局部细节与全局依赖。

## 十、常见问题 FAQ
### Q1: `AttributeError: module 'mediapipe' has no attribute 'solutions'`
A：mediapipe 0.10.30+ 移除了旧接口，请安装指定的 0.10.21 版本。

### Q2: `TypeError: Pose.__init__() got an unexpected keyword argument 'static_image'`
A：参数名错误，正确参数为 `static_image_mode`，代码已修复。

### Q3: `ValueError: Number of classes does not match size of target_names`
A：数据集中部分类别无有效样本，代码已改为动态适配实际类别数量，自动匹配。

### Q4: pip 安装超时、下载失败
A：添加清华镜像源参数 `-i https://pypi.tuna.tsinghua.edu.cn/simple`。

### Q5: matplotlib 不显示图片 / 无图形界面报错
A：代码已配置 `matplotlib.use("Agg")`，图片自动保存到 `pic_output/`，直接打开文件查看即可。

### Q6: numpy 与 opencv 版本冲突警告
A：使用指定的 opencv-python==4.9.0.80，与 numpy 1.x 完全兼容，无冲突。

## 十一、课程提交材料清单
1. 预处理代码：`preprocess.py`
2. 训练代码：`train.py`
3. 推理代码：`infer.py`
4. 项目说明：`README.md`
5. 实验报告：`实验报告.docx`
6. 结果图片：`pic_output/` 内的训练曲线、混淆矩阵