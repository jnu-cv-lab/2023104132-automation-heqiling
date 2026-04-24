# OpenCV 图像基础处理实验（Experiment 2）

## 实验概述
本实验基于 OpenCV C++ 实现了图像的基础处理操作，完成了**图像读取、信息输出、显示、灰度化、保存、像素操作、区域裁剪**等核心任务，是计算机视觉入门的基础实践。

---

## 实验环境
- 操作系统：WSL2 Ubuntu 22.04
- 开发工具：VS Code + CMake + g++
- 依赖库：OpenCV 4.x
- 编程语言：C++11

---

## 实验任务与实现
### 1. 任务清单
| 序号 | 任务内容 | 实现方式 |
|------|----------|----------|
| 1 | 读取本地图像文件 | `cv::imread()` |
| 2 | 输出图像基本信息（尺寸、通道数、像素类型） | `src_img.cols`/`rows`/`channels()` + 自定义类型转换函数 |
| 3 | 显示原始图像 | `cv::namedWindow()` + `cv::imshow()` + `cv::waitKey()` |
| 4 | 将彩色图像转换为灰度图像 | `cv::cvtColor()` + `COLOR_BGR2GRAY` |
| 5 | 保存灰度图像到本地 | `cv::imwrite()` |
| 6 | 读取图像中心像素值（BGR格式） | `src_img.at<cv::Vec3b>(cy, cx)` |
| 7 | 裁剪图像左上角200×200区域并保存 | `cv::Rect()` + `cv::imwrite()` |
| 8 | 显示灰度图、裁剪图 | 同原始图像显示逻辑 |

### 2. 核心代码说明
- **`main.cpp`**：主程序文件，实现所有实验任务
- **`CMakeLists.txt`**：CMake 构建配置文件，用于编译项目
- **`test_image.jpg`**：实验用原始输入图像

---

## 编译与运行步骤
### 1. 环境准备
```bash
# 安装OpenCV依赖
sudo apt update && sudo apt install -y libopencv-dev