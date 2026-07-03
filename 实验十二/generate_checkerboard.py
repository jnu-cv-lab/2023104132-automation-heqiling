#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
生成棋盘格标定板图片
====================
内角点数量: 9 × 6（即黑白方格为 10 × 7）
方格边长: 25 mm（打印时需确保实际尺寸一致）

用法：
    python generate_checkerboard.py

输出：
    checkerboard_9x6_25mm.png  -- 棋盘格图片（建议用 A4 纸打印）
"""

import cv2
import numpy as np
import os

# ==================== 参数设置 ====================
INNER_CORNERS_X = 9      # 水平方向内角点数量
INNER_CORNERS_Y = 6      # 垂直方向内角点数量
SQUARE_SIZE_MM  = 25     # 方格边长（毫米）
DPI             = 300    # 打印分辨率（每英寸像素数）

# 图像边距（像素）
MARGIN_PX = 100

# ==================== 计算图像尺寸 ====================
# 方格数量 = 内角点 + 1（每个方向）
NUM_SQUARES_X = INNER_CORNERS_X + 1  # 10 个黑白格
NUM_SQUARES_Y = INNER_CORNERS_Y + 1  # 7 个黑白格

# 将 mm 转换为像素（基于 DPI）
# 1 inch = 25.4 mm
SQUARE_SIZE_INCH = SQUARE_SIZE_MM / 25.4
SQUARE_SIZE_PX = int(round(SQUARE_SIZE_INCH * DPI))

print(f"方格边长: {SQUARE_SIZE_MM} mm = {SQUARE_SIZE_INCH:.4f} inch = {SQUARE_SIZE_PX} px @ {DPI} DPI")
print(f"棋盘格黑/白格数量: {NUM_SQUARES_X} × {NUM_SQUARES_Y}")
print(f"内角点数量: {INNER_CORNERS_X} × {INNER_CORNERS_Y}")

# 棋盘格的像素尺寸
board_width_px  = NUM_SQUARES_X * SQUARE_SIZE_PX
board_height_px = NUM_SQUARES_Y * SQUARE_SIZE_PX

# 加上边距的总图像尺寸
img_width  = board_width_px  + 2 * MARGIN_PX
img_height = board_height_px + 2 * MARGIN_PX

# ==================== 生成棋盘格 ====================
# 创建白色背景
img = np.ones((img_height, img_width), dtype=np.uint8) * 255

# 绘制黑色方格
for i in range(NUM_SQUARES_Y):
    for j in range(NUM_SQUARES_X):
        if (i + j) % 2 == 0:
            x1 = MARGIN_PX + j * SQUARE_SIZE_PX
            y1 = MARGIN_PX + i * SQUARE_SIZE_PX
            x2 = x1 + SQUARE_SIZE_PX
            y2 = y1 + SQUARE_SIZE_PX
            img[y1:y2, x1:x2] = 0

# ==================== 保存图片 ====================
script_dir = os.path.dirname(os.path.abspath(__file__)) if "__file__" in dir() else os.getcwd()
output_path = os.path.join(script_dir,
                           f"checkerboard_{INNER_CORNERS_X}x{INNER_CORNERS_Y}_{SQUARE_SIZE_MM}mm.png")
cv2.imwrite(output_path, img)
print(f"\n棋盘格图片已保存至: {output_path}")

# ==================== 打印提示 ====================
print(f"\n【打印设置】")
print(f"  1. 请使用 A4 纸打印")
print(f"  2. 打印时选择 '实际大小' 或 '100% 缩放'，不要缩放")
print(f"  3. 打印后测量每个方格边长应为 {SQUARE_SIZE_MM} mm")
print(f"  4. 将打印好的棋盘格贴在平整的硬板（如硬纸板、木板）上")
print(f"")
print(f"【屏幕显示替代方案】")
print(f"  若无法打印，也可在平板或电脑屏幕上全屏显示此图片")
print(f"  注意：使用屏幕时需实际测量方格边长（mm），并在标定程序中填入正确数值")
