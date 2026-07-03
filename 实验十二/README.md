# 相机标定实验 —— 使用棋盘格进行张正友标定法

计算机视觉实验，使用 OpenCV 对手机摄像头进行棋盘格标定，估计内参矩阵 K 和畸变参数 D。

## 📁 文件说明

| 文件 | 说明 |
|------|------|
| `calibrate_camera.py` | **相机标定主程序** —— 张正友标定法完整实现 |
| `generate_checkerboard.py` | 棋盘格标定板生成程序 |
| `checkerboard_9x6_25mm.png` | 预生成的棋盘格图片（9×6 内角点，25mm 方格） |
| `requirements.txt` | Python 依赖（`opencv-python numpy`） |
| `相机标定实验报告.docx` | 完整实验报告（Word） |
| `相机标定实验报告.pdf` | 完整实验报告（PDF） |

## 🚀 使用方法

### 1. 准备棋盘格

- **打印版**：用 A4 纸打印 `checkerboard_9x6_25mm.png`，选择"实际大小/100% 缩放"，贴在硬板上
- **屏幕版**：在平板/电脑屏幕上全屏显示，需实测方格边长

### 2. 拍摄标定图片

用同一相机从不同角度拍摄至少 15 张棋盘格照片，放入 `calib_images/` 文件夹。

### 3. 安装依赖

```bash
pip install -r requirements.txt
```

### 4. 修改参数

在 `calibrate_camera.py` 中修改：

```python
INNER_CORNERS_X = 9      # 水平方向内角点数量
INNER_CORNERS_Y = 6      # 垂直方向内角点数量
SQUARE_SIZE_MM  = 25.0   # 方格实际边长（mm）
```

### 5. 运行标定

```bash
python calibrate_camera.py
```

## 📊 输出结果

- **内参矩阵 K**（fx, fy, cx, cy）
- **畸变参数 D** = [k1, k2, p1, p2, k3]
- **重投影误差**（总体 + 每张图片）
- **角点检测可视化**：`output/corners/`
- **去畸变对比图**：`output/undistorted/`

## 📐 实现流程

1. 定义 3D 角点（z=0 平面，`cv2.findChessboardCorners` 检测 → `cv2.cornerSubPix` 亚像素优化）
2. 相机标定（`cv2.calibrateCamera`）
3. 重投影误差计算（`cv2.projectPoints`）
4. 去畸变（`cv2.undistort` + `cv2.getOptimalNewCameraMatrix`）

**标定模型**：`s·p = K · [R | t] · P`

## 📝 实验结果

| 项目 | 数值 |
|------|------|
| 内角点 | 9 × 6 |
| 方格边长 | 40 mm |
| 相机 | 手机摄像头，1706 × 1279 |
| fx | 1210.82 |
| fy | 1199.09 |
| cx | 858.12 |
| cy | 634.56 |
| 总体 RMS 重投影误差 | 1.09 pixels |
| 畸变参数 | [-3.84e-02, -3.24e-02, -3.89e-03, +5.49e-03, -4.57e-02] |

## 📄 参考资料

- Zhang, Z. "A Flexible New Technique for Camera Calibration." IEEE TPAMI, 2000.
- OpenCV Documentation: [Camera Calibration and 3D Reconstruction](https://docs.opencv.org/)
