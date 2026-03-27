import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def main():
    print("实验一：Python视觉开发环境搭建与图像基本读写")
    print("=" * 60)
    
    # ========== 任务1：使用OpenCV读取测试图片 ==========
    print("\n任务1：读取测试图片")
    # 注意：将 'test_image.jpg' 替换为你的实际图片文件名
    image_path = 'dog.jpg'
    
    if not os.path.exists(image_path):
        print(f"错误：找不到图片文件 '{image_path}'")
        print("请确保图片文件与代码在同一目录")
        # 创建一个示例图片用于测试（如果没有真实图片）
        create_sample_image()
        image_path = 'dog.jpg'
    
    # 读取图片
    img = cv2.imread(image_path)
    
    if img is None:
        print("错误：无法读取图片，请检查文件路径和格式")
        return
    
    print(f"✓ 成功读取图片: {image_path}")
    
    # ========== 任务2：输出图像基本信息 ==========
    print("\n任务2：输出图像基本信息")
    print("-" * 40)
    
    # 获取图像尺寸
    height, width = img.shape[:2]
    print(f"图像尺寸（高度 x 宽度）: {height} x {width}")
    
    # 获取通道数
    if len(img.shape) == 3:
        channels = img.shape[2]
        print(f"图像通道数: {channels}")
    else:
        channels = 1
        print(f"图像通道数: {channels} (灰度图)")
    
    # 获取数据类型
    dtype = img.dtype
    print(f"图像数据类型: {dtype}")
    
    # 获取像素值范围
    min_val = np.min(img)
    max_val = np.max(img)
    print(f"像素值范围: [{min_val}, {max_val}]")
    
    # ========== 任务3：显示原图 ==========
    print("\n任务3：显示原图")
    print("-" * 40)
    
    # 方法1：使用Matplotlib显示（推荐，在WSL中更稳定）
    plt.figure(figsize=(10, 8))
    
    # OpenCV默认是BGR格式，Matplotlib需要RGB格式
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    plt.subplot(2, 2, 1)
    plt.imshow(img_rgb)
    plt.title('原始图像 (RGB)')
    plt.axis('off')
    
    print("✓ 原始图像显示完成")
    print("   - 使用Matplotlib显示，按关闭按钮继续程序")
    
    # ========== 任务4：转换为灰度图并显示 ==========
    print("\n任务4：转换为灰度图")
    print("-" * 40)
    
    # 转换为灰度图
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    print(f"✓ 灰度图转换完成")
    print(f"   灰度图尺寸: {gray_img.shape}")
    print(f"   灰度图数据类型: {gray_img.dtype}")
    
    # 显示灰度图
    plt.subplot(2, 2, 2)
    plt.imshow(gray_img, cmap='gray')
    plt.title('灰度图像')
    plt.axis('off')
    
    # ========== 任务5：保存处理结果 ==========
    print("\n任务5：保存灰度图为新文件")
    print("-" * 40)
    
    # 保存灰度图
    gray_output_path = 'gray_image.jpg'
    cv2.imwrite(gray_output_path, gray_img)
    
    if os.path.exists(gray_output_path):
        print(f"✓ 灰度图已保存为: {gray_output_path}")
        print(f"   文件大小: {os.path.getsize(gray_output_path)} 字节")
    else:
        print("✗ 保存失败")
    
    # ========== 任务6：NumPy简单操作 ==========
    print("\n任务6：NumPy操作")
    print("-" * 40)
    
    # 操作1：输出某个像素值
    print("1. 像素值查看:")
    if len(img.shape) == 3:
        # 彩色图像：获取中心像素的BGR值
        center_y, center_x = height // 2, width // 2
        pixel_value = img[center_y, center_x]
        print(f"   图像中心点 ({center_x}, {center_y}) 的像素值:")
        print(f"   B: {pixel_value[0]}, G: {pixel_value[1]}, R: {pixel_value[2]}")
        
        # 灰度图中心像素值
        gray_pixel = gray_img[center_y, center_x]
        print(f"   对应灰度值: {gray_pixel}")
    else:
        # 灰度图像
        center_y, center_x = height // 2, width // 2
        pixel_value = img[center_y, center_x]
        print(f"   图像中心点 ({center_x}, {center_y}) 的像素值: {pixel_value}")
    
    # 操作2：裁剪左上角区域并保存
    print("\n2. 裁剪左上角区域:")
    
    # 定义裁剪区域大小（例如：100x100像素）
    crop_size = 100
    crop_height = min(crop_size, height)
    crop_width = min(crop_size, width)
    
    # 裁剪左上角区域
    cropped_region = img[0:crop_height, 0:crop_width]
    
    print(f"   裁剪区域尺寸: {crop_height} x {crop_width}")
    print(f"   裁剪后图像形状: {cropped_region.shape}")
    
    # 保存裁剪的图像
    crop_output_path = 'cropped_region.jpg'
    cv2.imwrite(crop_output_path, cropped_region)
    print(f"✓ 裁剪区域已保存为: {crop_output_path}")
    
    # 显示裁剪的区域
    cropped_rgb = cv2.cvtColor(cropped_region, cv2.COLOR_BGR2RGB)
    plt.subplot(2, 2, 3)
    plt.imshow(cropped_rgb)
    plt.title('裁剪区域 (左上角)')
    plt.axis('off')
    
    # 显示裁剪区域在原始图像中的位置
    img_with_rect = img_rgb.copy()
    cv2.rectangle(img_with_rect, (0, 0), (crop_width, crop_height), (255, 0, 0), 3)
    
    plt.subplot(2, 2, 4)
    plt.imshow(img_with_rect)
    plt.title('原始图像 (标记裁剪区域)')
    plt.axis('off')
    
    # 调整布局并显示所有图像
    plt.tight_layout()
    plt.show()
    
    print("\n" + "=" * 60)
    print("所有任务完成！")
    print("=" * 60)
    
    # 显示文件列表
    print("\n生成的文件:")
    files = ['dog.jpg', 'gray_image.jpg', 'cropped_region.jpg']
    for file in files:
        if os.path.exists(file):
            print(f"  ✓ {file} ({os.path.getsize(file)} 字节)")
    
    return True

def create_sample_image():
    """如果没有测试图片，创建一个简单的测试图片"""
    print("创建示例测试图片...")
    
    # 创建一个彩色渐变图像
    height, width = 400, 600
    img = np.zeros((height, width, 3), dtype=np.uint8)
    
    # 创建渐变效果
    for i in range(height):
        for j in range(width):
            # BGR格式
            img[i, j] = [
                int(255 * j / width),        # 蓝色通道：水平渐变
                int(255 * i / height),       # 绿色通道：垂直渐变
                int(255 * (1 - j / width))   # 红色通道：反向水平渐变
            ]
    
    # 添加一些形状
    cv2.rectangle(img, (100, 100), (200, 200), (0, 255, 255), -1)  # 黄色矩形
    cv2.circle(img, (450, 200), 50, (255, 0, 255), -1)  # 紫色圆形
    cv2.putText(img, 'Test Image', (200, 350), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    # 保存图片
    cv2.imwrite('dog.jpg', img)
    print("✓ 已创建示例图片: dog.jpg")
    
if __name__ == "__main__":
    main()