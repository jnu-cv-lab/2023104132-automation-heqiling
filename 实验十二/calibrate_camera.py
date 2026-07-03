#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
相机标定程序 —— 使用棋盘格进行张正友标定法
===========================================
兼容中文路径：使用 np.fromfile + cv2.imdecode 替代 cv2.imread
"""

import cv2
import numpy as np
import os
import sys

# ============================================================================
# ==================== 参数设置 ====================
# ============================================================================

INNER_CORNERS_X = 9
INNER_CORNERS_Y = 6
SQUARE_SIZE_MM  = 40.0          # 实测方格边长 = 4 cm
IMAGE_FOLDER    = "./calib_images"

# ============================================================================
# ==================== 辅助函数 ====================
# ============================================================================

def imread_unicode(path):
    """兼容中文路径的图片读取"""
    with open(path, "rb") as f:
        data = np.frombuffer(f.read(), dtype=np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_COLOR)
    return img


def create_output_dirs():
    for d in ["./output", "./output/corners", "./output/undistorted"]:
        os.makedirs(d, exist_ok=True)


def load_image_paths(folder):
    """加载所有标定图片路径（兼容中文）"""
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}
    paths = []
    try:
        for f in os.listdir(folder):
            if os.path.splitext(f)[1].lower() in exts:
                paths.append(os.path.join(folder, f))
    except FileNotFoundError:
        pass
    return sorted(paths)


def get_3d_object_points():
    objp = np.zeros((INNER_CORNERS_X * INNER_CORNERS_Y, 3), dtype=np.float32)
    objp[:, :2] = np.mgrid[0:INNER_CORNERS_X, 0:INNER_CORNERS_Y].T.reshape(-1, 2)
    objp[:, :2] *= SQUARE_SIZE_MM
    return objp


def draw_text(img, text, pos, font_scale=1.2, color=(0, 255, 0)):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(img, text, pos, font, font_scale, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, pos, font, font_scale, color, 2, cv2.LINE_AA)
    return img


def compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist):
    total_err, total_pts = 0.0, 0
    per_view = []
    for i in range(len(objpoints)):
        proj, _ = cv2.projectPoints(objpoints[i], rvecs[i], tvecs[i], K, dist)
        e = cv2.norm(imgpoints[i], proj, cv2.NORM_L2) / np.sqrt(len(proj))
        per_view.append(e)
        total_err += e ** 2 * len(imgpoints[i])
        total_pts += len(imgpoints[i])
    return np.sqrt(total_err / total_pts), per_view


# ============================================================================
# ==================== 主程序 ====================
# ============================================================================

def main():
    print("=" * 70)
    print("  相机标定程序 —— 张正友棋盘格标定法")
    print("=" * 70)

    if not os.path.isdir(IMAGE_FOLDER):
        print(f"\n[错误] 找不到标定图片文件夹: {IMAGE_FOLDER}")
        sys.exit(1)

    image_paths = load_image_paths(IMAGE_FOLDER)
    if len(image_paths) == 0:
        print(f"\n[错误] {IMAGE_FOLDER} 文件夹中没有找到图片文件。")
        sys.exit(1)

    print(f"\n[信息] 共找到 {len(image_paths)} 张图片")
    print(f"  棋盘格规格: {INNER_CORNERS_X} x {INNER_CORNERS_Y} 内角点")
    print(f"  方格边长:   {SQUARE_SIZE_MM} mm\n")

    create_output_dirs()

    # ---- 步骤 1: 定义三维角点 ----
    objp = get_3d_object_points()
    print(f"[步骤 1] 定义三维角点坐标 (z=0, 间距={SQUARE_SIZE_MM}mm)")

    # ---- 步骤 2: 角点检测 ----
    print(f"[步骤 2] 检测棋盘格角点 ...\n")
    objpoints, imgpoints = [], []
    successful, failed = [], []
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    for idx, path in enumerate(image_paths):
        fname = os.path.basename(path)
        print(f"  [{idx+1}/{len(image_paths)}] {fname} ... ", end="", flush=True)

        img = imread_unicode(path)
        if img is None:
            print("读取失败")
            failed.append((path, "读取失败"))
            continue

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(
            gray, (INNER_CORNERS_X, INNER_CORNERS_Y),
            flags=cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE
        )

        if not found:
            print("角点未检测到")
            failed.append((path, "角点检测失败"))
            continue

        corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
        objpoints.append(objp)
        imgpoints.append(corners_sub)
        successful.append(path)

        # 保存角点检测可视化
        vis = cv2.drawChessboardCorners(img.copy(), (INNER_CORNERS_X, INNER_CORNERS_Y), corners_sub, found)
        vis = draw_text(vis, f"Img {idx+1}: {INNER_CORNERS_X}x{INNER_CORNERS_Y} corners", (20, 40))
        cv2.imwrite(f"./output/corners/{os.path.splitext(fname)[0]}_corners.jpg", vis)

        print(f"OK ({len(corners)} 角点)")

    n_valid = len(successful)
    print(f"\n  检测结果: {n_valid}/{len(image_paths)} 成功")

    if failed:
        print(f"  失败图片:")
        for p, r in failed:
            print(f"    - {os.path.basename(p)}: {r}")
        print()

    if n_valid < 5:
        print("[错误] 成功检测的图片不足（至少需要 5 张）")
        sys.exit(1)

    # ---- 步骤 3: 相机标定 ----
    print(f"\n[步骤 3] 相机标定 (cv2.calibrateCamera) ...")
    first_img = imread_unicode(successful[0])
    h, w = first_img.shape[:2]
    print(f"  图片分辨率: {w} x {h}")

    ret_rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(
        objpoints, imgpoints, (w, h), None, None
    )

    print(f"\n  内参矩阵 K:")
    print(f"    fx = {K[0,0]:.4f}    skew = {K[0,1]:.4f}    cx = {K[0,2]:.4f}")
    print(f"    fy = {K[1,1]:.4f}    cy = {K[1,2]:.4f}")
    print(f"    K = np.array([")
    print(f"        [{K[0,0]:.4f}, {K[0,1]:.4f}, {K[0,2]:.4f}],")
    print(f"        [{K[1,0]:.4f}, {K[1,1]:.4f}, {K[1,2]:.4f}],")
    print(f"        [{K[2,0]:.4f}, {K[2,1]:.4f}, {K[2,2]:.4f}] ])")

    print(f"\n  畸变参数 D = [k1, k2, p1, p2, k3]:")
    print(f"    k1 = {dist[0,0]:+.6e}")
    print(f"    k2 = {dist[0,1]:+.6e}")
    print(f"    p1 = {dist[0,2]:+.6e}")
    print(f"    p2 = {dist[0,3]:+.6e}")
    print(f"    k3 = {dist[0,4]:+.6e}")

    # ---- 步骤 4: 重投影误差 ----
    print(f"\n[步骤 4] 重投影误差 ...\n")
    total_rms, per_view = compute_reprojection_error(objpoints, imgpoints, rvecs, tvecs, K, dist)

    print(f"  总体 RMS 重投影误差: {total_rms:.6f} pixels")
    print(f"  {'='*55}")
    print(f"  {'图片':<14} {'RMS (pixels)':<20} {'评价'}")
    print(f"  {'-'*55}")
    for i, (err, path) in enumerate(zip(per_view, successful)):
        if err < 0.5:    g = "优秀"
        elif err < 1.0:  g = "良好"
        elif err < 2.0:  g = "一般"
        else:            g = "较差"
        print(f"  Img {i+1:<10} {err:<20.6f} {g}")
    print(f"  {'='*55}")

    if total_rms < 1.0:
        print(f"  => 重投影误差 < 1 pixel, 标定精度良好!")
    else:
        print(f"  => 重投影误差偏高，建议检查标定图片质量。")

    # ---- 步骤 5: 去畸变 ----
    print(f"\n[步骤 5] 去畸变处理 ...\n")

    demo = imread_unicode(successful[0])
    demo_name = os.path.splitext(os.path.basename(successful[0]))[0]

    # 去畸变
    img_undist = cv2.undistort(demo, K, dist, None, K)
    cv2.imwrite(f"./output/undistorted/{demo_name}_undistorted.jpg", img_undist)

    # optimal 版本
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=1)
    img_undist_opt = cv2.undistort(demo, K, dist, None, new_K)
    x, y, wr, hr = roi
    if wr > 0 and hr > 0:
        img_undist_opt = img_undist_opt[y:y+hr, x:x+wr]
    cv2.imwrite(f"./output/undistorted/{demo_name}_undistorted_optimal.jpg", img_undist_opt)

    # 对比图
    max_w = 900
    if w > max_w:
        scale = max_w / w
        dh = cv2.resize(demo, (max_w, int(h*scale)))
        uh = cv2.resize(img_undist, (max_w, int(h*scale)))
    else:
        dh, uh = demo, img_undist
    hh = dh.shape[0]
    comp = np.vstack([dh, uh])
    cv2.line(comp, (0, hh), (comp.shape[1], hh), (0, 0, 255), 2)
    comp = draw_text(comp, "Original Image",     (20, 40), 1.0, (0, 255, 0))
    comp = draw_text(comp, "Undistorted Image",  (20, hh+40), 1.0, (0, 255, 255))
    cv2.imwrite(f"./output/undistorted/{demo_name}_comparison.jpg", comp)

    # 额外去畸变几张
    for i in range(1, min(3, len(successful))):
        eimg = imread_unicode(successful[i])
        eund = cv2.undistort(eimg, K, dist, None, K)
        ename = os.path.splitext(os.path.basename(successful[i]))[0]
        cv2.imwrite(f"./output/undistorted/{ename}_undistorted.jpg", eund)

    print(f"  去畸变结果已保存至 ./output/undistorted/")

    # ---- 畸变分析 ----
    k1 = dist[0, 0]
    abs_k1 = abs(k1)
    if abs_k1 < 0.05:   severity = "轻微"
    elif abs_k1 < 0.15: severity = "中等"
    else:               severity = "明显"

    cx, cy = K[0, 2], K[1, 2]
    img_cx, img_cy = w / 2, h / 2
    cx_off_pct = (cx - img_cx) / w * 100
    cy_off_pct = (cy - img_cy) / h * 100

    print(f"\n  ┌─────────────────────────────────────────┐")
    print(f"  │  畸变分析                                │")
    print(f"  ├─────────────────────────────────────────┤")
    print(f"  │  k1 = {k1:+.6e} => 畸变程度: {severity}         │")
    print(f"  │  光心偏移: ({cx_off_pct:+.2f}%, {cy_off_pct:+.2f}%)              │")
    print(f"  └─────────────────────────────────────────┘")

    # ---- 汇总 ----
    fx, fy = K[0, 0], K[1, 1]
    print(f"\n{'=' * 70}")
    print(f"  标定结果汇总")
    print(f"{'=' * 70}")
    print(f"  相机: 手机摄像头 (分辨率 {w}x{h})")
    print(f"  棋盘格: {INNER_CORNERS_X}x{INNER_CORNERS_Y} 内角点, 方格 {SQUARE_SIZE_MM}mm")
    print(f"  参与标定: {n_valid}/{len(image_paths)} 张")
    print(f"")
    print(f"  内参矩阵 K:")
    print(f"    fx={fx:.4f}, fy={fy:.4f}, cx={cx:.4f}, cy={cy:.4f}")
    print(f"    fx/fy = {fx/fy:.4f}")
    print(f"")
    print(f"  畸变参数 D:")
    print(f"    [{', '.join(f'{d:+.6e}' for d in dist[0])}]")
    print(f"")
    print(f"  总体 RMS 重投影误差: {total_rms:.6f} pixels")
    print(f"")

    # 简要分析
    print(f"{'=' * 70}")
    print(f"  简要分析（可直接用于实验报告）")
    print(f"{'=' * 70}")
    print(f"""
  1. fx = {fx:.2f}, fy = {fy:.2f}
     比值 = {fx/fy:.4f}，{'fx 和 fy 非常接近，像素近似正方形。' if 0.95<fx/fy<1.05 else '有一定差异。'}

  2. cx = {cx:.2f}, cy = {cy:.2f}
     图像中心为 ({img_cx:.0f}, {img_cy:.0f})，
     光心偏移约 ({cx-img_cx:+.1f}, {cy-img_cy:+.1f}) pixels ({cx_off_pct:+.2f}%, {cy_off_pct:+.2f}%)，
     {'光心非常接近图像中心。' if abs(cx_off_pct)<3 and abs(cy_off_pct)<3 else '光心与图像中心有偏差，属正常范围。'}

  3. 总体 RMS 重投影误差 = {total_rms:.4f} pixels，
     {'低于 1 pixel，标定精度良好。' if total_rms<1.0 else '偏高，建议增加不同姿态图片。'}

  4. 径向畸变 k1 = {k1:+.4e}，
     畸变程度：{severity}。{'镜头畸变在正常范围内。' if abs_k1<0.15 else '畸变较明显，可能是广角镜头。'}

  5. {f'角点检测全部成功（{n_valid}/{len(image_paths)}）。' if len(failed)==0 else f'有 {len(failed)} 张检测失败，可能原因：模糊/遮挡/光照问题。'}
""")
    print(f"{'=' * 70}")
    print(f"  输出文件:")
    print(f"    ./output/corners/       -- 角点检测可视化 ({n_valid} 张)")
    print(f"    ./output/undistorted/   -- 去畸变结果与对比图")
    print(f"{'=' * 70}")
    print(f"  程序执行完毕！\n")


if __name__ == "__main__":
    main()
