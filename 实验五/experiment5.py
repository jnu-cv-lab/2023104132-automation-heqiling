import cv2
import numpy as np

# 路径设置（直接在当前目录运行）
img_original = "test_first.jpg"
img_target = "test_final.jpg"

# 生成测试图像：包含方框、圆形、矩形、网格线
def create_test_image(size=600):
    img = np.ones((size, size, 3), dtype=np.uint8) * 255

    # 外边框
    cv2.line(img, (50, 50), (550, 50), (0, 0, 0), 3)
    cv2.line(img, (50, 550), (550, 550), (0, 0, 0), 3)
    cv2.line(img, (50, 50), (50, 550), (0, 0, 0), 3)
    cv2.line(img, (550, 50), (550, 550), (0, 0, 0), 3)

    # 圆形
    cv2.circle(img, (220, 300), 130, (0, 0, 0), 3)

    # 矩形
    cv2.rectangle(img, (380, 170), (480, 430), (0, 0, 0), 3)

    # 水平与垂直网格线
    for y_pos in [120, 220, 320, 420]:
        cv2.line(img, (80, y_pos), (520, y_pos), (80, 80, 80), 2)
    for x_pos in [120, 220, 320, 420]:
        cv2.line(img, (x_pos, 80), (x_pos, 520), (80, 80, 80), 2)

    return img

# 生成并保存原始图
src_img = create_test_image()
cv2.imwrite(img_original, src_img)
print("已生成测试原图：test_first.jpg")

height, width = src_img.shape[:2]

# ------------------- 三种几何变换 -------------------
# 1. 相似变换（旋转 + 缩放）
M_similarity = cv2.getRotationMatrix2D((width/2, height/2), 15, 0.85)
img_similar = cv2.warpAffine(src_img, M_similarity, (width, height), borderValue=(255,255,255))

# 2. 仿射变换
pts_src_aff = np.float32([[50, 50], [550, 50], [50, 550]])
pts_dst_aff = np.float32([[90, 110], [510, 70], [70, 490]])
M_affine = cv2.getAffineTransform(pts_src_aff, pts_dst_aff)
img_affine = cv2.warpAffine(src_img, M_affine, (width, height), borderValue=(255,255,255))

# 3. 透视变换
pts_src_per = np.float32([[0, 0], [width, 0], [0, height], [width, height]])
pts_dst_per = np.float32([[50, 40], [width-40, 30], [30, height-40], [width-30, height-50]])
M_perspective = cv2.getPerspectiveTransform(pts_src_per, pts_dst_per)
img_perspective = cv2.warpPerspective(src_img, M_perspective, (width, height), borderValue=(255,255,255))

# 拼接四宫格对比图
row1 = np.hstack((src_img, img_similar))
row2 = np.hstack((img_affine, img_perspective))
result_all = np.vstack((row1, row2))

# 保存变换结果
cv2.imwrite("similar_transform.jpg", img_similar)
cv2.imwrite("affine_transform.jpg", img_affine)
cv2.imwrite("perspective_transform.jpg", img_perspective)
cv2.imwrite("compare_4grid.jpg", result_all)

# ------------------- A4 纸透视校正 -------------------
frame = cv2.imread(img_target)
if frame is None:
    print("未找到待校正图片：test_final.jpg")
    exit()

display = frame.copy()
H, W = frame.shape[:2]

# 边缘检测
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
blur = cv2.GaussianBlur(gray, (9, 9), 0)
canny = cv2.Canny(blur, 30, 120)
canny = cv2.morphologyEx(canny, cv2.MORPH_CLOSE, np.ones((7,7), np.uint8))

# 寻找最大四边形轮廓
contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contours = sorted(contours, key=cv2.contourArea, reverse=True)
paper_points = None

for cnt in contours:
    length = cv2.arcLength(cnt, True)
    approx = cv2.approxPolyDP(cnt, 0.02 * length, True)
    if len(approx) == 4 and cv2.contourArea(approx) > 0.2 * W * H:
        paper_points = approx.reshape(4, 2).astype(np.float32)
        break

# 四点排序函数
def sort_points(pts):
    rect = np.zeros((4, 2), dtype=np.float32)
    sum_pts = pts.sum(axis=1)
    rect[0] = pts[np.argmin(sum_pts)]
    rect[2] = pts[np.argmax(sum_pts)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

if paper_points is not None:
    paper_points = sort_points(paper_points)
    cv2.polylines(display, [paper_points.astype(int)], True, (0,0,255), 4)
else:
    paper_points = np.float32([[W*0.05,H*0.05],[W*0.95,H*0.05],[W*0.95,H*0.95],[W*0.05,H*0.95]])

cv2.imwrite("detect_paper.jpg", display)

# 执行透视校正
A4_W = 700
A4_H = int(700 * 1.4142)
dst_points = np.float32([[0,0],[A4_W,0],[A4_W,A4_H],[0,A4_H]])
M_correct = cv2.getPerspectiveTransform(paper_points, dst_points)
corrected_img = cv2.warpPerspective(frame, M_correct, (A4_W, A4_H))

cv2.imwrite("result_corrected.jpg", corrected_img)
print("A4 透视校正完成！")