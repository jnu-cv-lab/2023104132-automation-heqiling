import cv2
import numpy as np
import matplotlib.pyplot as plt

# ===================== 全局配置 =====================
IMG_TEMP = "box.png"          # 模板图
IMG_SCENE = "box_in_scene.png"# 场景图
TOP_MATCH = 50                # 显示前50个匹配
RANSAC_THRESH = 5.0           # 重投影误差阈值

# ===================== 任务1：ORB特征检测 =====================
def orb_detect(nfeatures=1000):
    # 读取图像
    img_temp = cv2.imread(IMG_TEMP, cv2.IMREAD_GRAYSCALE)
    img_scene = cv2.imread(IMG_SCENE, cv2.IMREAD_GRAYSCALE)
    
    # 创建ORB检测器
    orb = cv2.ORB_create(nfeatures=nfeatures)
    
    # 检测关键点+计算描述子
    kp_temp, des_temp = orb.detectAndCompute(img_temp, None)
    kp_scene, des_scene = orb.detectAndCompute(img_scene, None)
    
    # 可视化关键点
    img_temp_kp = cv2.drawKeypoints(img_temp, kp_temp, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img_scene_kp = cv2.drawKeypoints(img_scene, kp_scene, None, color=(0,255,0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    # 保存可视化图
    cv2.imwrite(f"task1_orb_kp_temp_{nfeatures}.jpg", img_temp_kp)
    cv2.imwrite(f"task1_orb_kp_scene_{nfeatures}.jpg", img_scene_kp)
    
    # 输出信息
    print(f"========== ORB特征检测(nfeatures={nfeatures}) ==========")
    print(f"模板图关键点数量：{len(kp_temp)}")
    print(f"场景图关键点数量：{len(kp_scene)}")
    print(f"描述子维度：{des_temp.shape[1]} (256bit/32字节)")
    
    return img_temp, img_scene, kp_temp, kp_scene, des_temp, des_scene

# ===================== 任务2：ORB特征匹配 =====================
def orb_match(img_temp, img_scene, kp_temp, kp_scene, des_temp, des_scene):
    # 创建暴力匹配器（ORB用汉明距离）
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    
    # 匹配描述子
    matches = bf.match(des_temp, des_scene)
    
    # 按距离从小到大排序
    matches = sorted(matches, key=lambda x: x.distance)
    
    # 可视化前50个匹配
    img_match = cv2.drawMatches(img_temp, kp_temp, img_scene, kp_scene, matches[:TOP_MATCH], None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("task2_orb_match.jpg", img_match)
    
    # 输出信息
    print(f"========== ORB特征匹配 ==========")
    print(f"总匹配数量：{len(matches)}")
    
    return matches

# ===================== 任务3：RANSAC剔除错误匹配 =====================
def ransac_filter(img_temp, img_scene, kp_temp, kp_scene, matches):
    # 提取匹配点坐标
    pts_temp = np.float32([kp_temp[m.queryIdx].pt for m in matches]).reshape(-1,1,2)
    pts_scene = np.float32([kp_scene[m.trainIdx].pt for m in matches]).reshape(-1,1,2)
    
    # 计算单应矩阵+RANSAC剔除误匹配
    H, mask = cv2.findHomography(pts_temp, pts_scene, cv2.RANSAC, RANSAC_THRESH)
    mask = mask.ravel().tolist()
    
    # 统计内点
    inlier_num = sum(mask)
    match_num = len(matches)
    inlier_ratio = inlier_num / match_num
    
    # 可视化RANSAC后匹配
    img_ransac = cv2.drawMatches(img_temp, kp_temp, img_scene, kp_scene, matches, None, matchesMask=mask, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    cv2.imwrite("task3_ransac_match.jpg", img_ransac)
    
    # 输出信息
    print(f"========== RANSAC剔除误匹配 ==========")
    print(f"单应矩阵H：\n{H}")
    print(f"总匹配数量：{match_num}")
    print(f"RANSAC内点数量：{inlier_num}")
    print(f"内点比例：{inlier_ratio:.2%}")
    
    return H, mask, inlier_num, match_num, inlier_ratio

# ===================== 任务4：目标定位 =====================
def object_localization(img_scene, img_temp, H):
    # 获取模板图四个角点
    h, w = img_temp.shape
    pts_corner = np.float32([[0,0], [0,h-1], [w-1,h-1], [w-1,0]]).reshape(-1,1,2)
    
    # 投影到场景图
    pts_corner_proj = cv2.perspectiveTransform(pts_corner, H)
    
    # 绘制边框
    img_scene_rgb = cv2.cvtColor(img_scene, cv2.COLOR_GRAY2BGR)
    cv2.polylines(img_scene_rgb, [np.int32(pts_corner_proj)], True, (0,0,255), 3)
    
    # 保存结果
    cv2.imwrite("task4_localization.jpg", img_scene_rgb)
    print("========== 目标定位完成 ==========")
    print("目标边框已绘制，定位成功")
    return img_scene_rgb

# ===================== 任务6：参数对比实验 =====================
def param_compare():
    nfeatures_list = [500, 1000, 2000]
    result_table = []
    
    for n in nfeatures_list:
        img_t, img_s, kp_t, kp_s, des_t, des_s = orb_detect(n)
        matches = orb_match(img_t, img_s, kp_t, kp_s, des_t, des_s)
        H, mask, in_num, match_num, ratio = ransac_filter(img_t, img_s, kp_t, kp_s, matches)
        loc_success = True if ratio > 0.1 else False
        
        result_table.append([n, len(kp_t), len(kp_s), match_num, in_num, f"{ratio:.2%}", loc_success])
    
    # 打印对比表格
    print("\n========== 参数对比实验结果 ==========")
    print(f"{'nfeatures':<10}{'模板关键点数':<12}{'场景关键点数':<12}{'匹配数量':<10}{'内点数':<10}{'内点比例':<10}{'定位成功'}")
    for row in result_table:
        print(f"{row[0]:<10}{row[1]:<12}{row[2]:<12}{row[3]:<10}{row[4]:<10}{row[5]:<10}{row[6]}")
    
    return result_table

# ===================== 选做任务：SIFT特征匹配 =====================
def sift_compare():
    # 读取图像
    img_t = cv2.imread(IMG_TEMP, cv2.IMREAD_GRAYSCALE)
    img_s = cv2.imread(IMG_SCENE, cv2.IMREAD_GRAYSCALE)
    
    # SIFT检测
    sift = cv2.SIFT_create()
    kp_t, des_t = sift.detectAndCompute(img_t, None)
    kp_s, des_s = sift.detectAndCompute(img_s, None)
    
    # KNN匹配+Lowe比值筛选
    bf = cv2.BFMatcher(cv2.NORM_L2)
    matches = bf.knnMatch(des_t, des_s, k=2)
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)
    
    # RANSAC
    pts_t = np.float32([kp_t[m.queryIdx].pt for m in good_matches]).reshape(-1,1,2)
    pts_s = np.float32([kp_s[m.trainIdx].pt for m in good_matches]).reshape(-1,1,2)
    H, mask = cv2.findHomography(pts_t, pts_s, cv2.RANSAC, 5.0)
    in_num = sum(mask.ravel())
    ratio = in_num / len(good_matches)
    loc_success = ratio > 0.1
    
    # 输出结果
    print("\n========== SIFT实验结果 ==========")
    print(f"匹配数量：{len(good_matches)}")
    print(f"内点数量：{in_num}")
    print(f"内点比例：{ratio:.2%}")
    print(f"定位成功：{loc_success}")
    
    return len(good_matches), in_num, ratio, loc_success

# ===================== 主函数 =====================
if __name__ == "__main__":
    # 执行默认实验(nfeatures=1000)
    img_temp, img_scene, kp_temp, kp_scene, des_temp, des_scene = orb_detect(1000)
    matches = orb_match(img_temp, img_scene, kp_temp, kp_scene, des_temp, des_scene)
    H, mask, in_num, match_num, ratio = ransac_filter(img_temp, img_scene, kp_temp, kp_scene, matches)
    object_localization(img_scene, img_temp, H)
    
    # 参数对比实验
    param_compare()
    
    # 选做SIFT
    sift_compare()