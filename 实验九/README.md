# 计算机视觉第10次实验：CNN训练分析与可视化
## 实验简介
本实验基于PyTorch复用CNN模型，完成：
优化器对比、学习率对比、卷积核可视化、特征图可视化、错误样本分析、混淆矩阵绘制。

## 实验环境
- 系统：WSL Ubuntu 24.04
- Python：3.x
- 依赖：torch torchvision numpy matplotlib seaborn
- 数据集：MNIST手写数字

## 项目文件结构
experiment10/
├── cnn_analysis.py # 完整实验代码
├── optimizer_compare.png # 优化器对比曲线
├── lr_compare.png # 学习率对比曲线
├── conv_kernel.png # 卷积核可视化
├── feature_map.png # 特征图可视化
├── error_samples.png # 错误样本
├── confusion_matrix.png # 混淆矩阵
├── README.md
└── 实验报告.docx
