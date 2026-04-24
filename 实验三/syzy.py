import cv2
import numpy as np
import matplotlib.pyplot as plt

# -------------------------- 1. 图像读入与预处理 --------------------------
def load_gray_image(path):
    """读取灰度图像"""
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"无法读取图像文件: {path}")
    return img

# -------------------------- 2. 下采样（两种方式） --------------------------
def downsample_direct(img, scale=0.5):
    """不做预滤波直接下采样"""
    h, w = img.shape
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

def downsample_gaussian_blur(img, scale=0.5, ksize=(5,5), sigmaX=1.0):
    """先高斯平滑再下采样"""
    blurred = cv2.GaussianBlur(img, ksize, sigmaX)
    h, w = blurred.shape
    new_h, new_w = int(h * scale), int(w * scale)
    return cv2.resize(blurred, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

# -------------------------- 3. 图像恢复（三种插值方法） --------------------------
def restore_image(img_small, original_shape):
    h, w = original_shape
    nearest = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_NEAREST)
    bilinear = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_LINEAR)
    bicubic = cv2.resize(img_small, (w, h), interpolation=cv2.INTER_CUBIC)
    return nearest, bilinear, bicubic

# -------------------------- 4. 空间域质量评价（MSE/PSNR） --------------------------
def calculate_mse_psnr(img1, img2):
    mse = np.mean((img1.astype(np.float64) - img2.astype(np.float64)) ** 2)
    if mse == 0:
        return 0, float('inf')
    psnr = 10 * np.log10((255 ** 2) / mse)
    return mse, psnr

# -------------------------- 5. 傅里叶变换分析 --------------------------
def fft_analysis(img):
    f = np.fft.fft2(img)
    f_shift = np.fft.fftshift(f)
    magnitude = 20 * np.log(np.abs(f_shift) + 1)
    return magnitude

# -------------------------- 6. DCT变换分析 --------------------------
def dct_analysis(img):
    img_float = np.float32(img)
    dct = cv2.dct(img_float)
    dct_log = 20 * np.log(np.abs(dct) + 1)
    
    h, w = dct.shape
    low_h, low_w = h // 8, w // 8
    low_freq_energy = np.sum(np.abs(dct[:low_h, :low_w]) ** 2)
    total_energy = np.sum(np.abs(dct) ** 2)
    energy_ratio = low_freq_energy / total_energy if total_energy != 0 else 0
    
    return dct_log, energy_ratio

# -------------------------- 7. 可视化（已修复：保存图片，不弹窗） --------------------------
def visualize_results(original, small_direct, small_blur, 
                     nearest_d, bilinear_d, bicubic_d,
                     nearest_b, bilinear_b, bicubic_b):
    plt.figure(figsize=(18, 12))
    
    plt.subplot(3, 4, 1), plt.imshow(original, cmap='gray'), plt.title('Original'), plt.axis('off')
    plt.subplot(3, 4, 2), plt.imshow(small_direct, cmap='gray'), plt.title('Direct Downsample (1/2)'), plt.axis('off')
    plt.subplot(3, 4, 3), plt.imshow(small_blur, cmap='gray'), plt.title('Gaussian Blur + Downsample (1/2)'), plt.axis('off')
    
    plt.subplot(3, 4, 5), plt.imshow(nearest_d, cmap='gray'), plt.title('Nearest (Direct Down)'), plt.axis('off')
    plt.subplot(3, 4, 6), plt.imshow(bilinear_d, cmap='gray'), plt.title('Bilinear (Direct Down)'), plt.axis('off')
    plt.subplot(3, 4, 7), plt.imshow(bicubic_d, cmap='gray'), plt.title('Bicubic (Direct Down)'), plt.axis('off')
    
    plt.subplot(3, 4, 9), plt.imshow(nearest_b, cmap='gray'), plt.title('Nearest (Blur Down)'), plt.axis('off')
    plt.subplot(3, 4, 10), plt.imshow(bilinear_b, cmap='gray'), plt.title('Bilinear (Blur Down)'), plt.axis('off')
    plt.subplot(3, 4, 11), plt.imshow(bicubic_b, cmap='gray'), plt.title('Bicubic (Blur Down)'), plt.axis('off')
    
    plt.tight_layout()
    plt.savefig("result_images.png", dpi=150, bbox_inches='tight')
    plt.close()

def visualize_fft(fft_original, fft_small_direct, fft_bilinear_d):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1), plt.imshow(fft_original, cmap='gray'), plt.title('Original FFT Spectrum'), plt.axis('off')
    plt.subplot(1, 3, 2), plt.imshow(fft_small_direct, cmap='gray'), plt.title('Direct Downsample FFT Spectrum'), plt.axis('off')
    plt.subplot(1, 3, 3), plt.imshow(fft_bilinear_d, cmap='gray'), plt.title('Bilinear Restored FFT Spectrum'), plt.axis('off')
    plt.tight_layout()
    plt.savefig("result_fft.png", dpi=150, bbox_inches='tight')
    plt.close()

def visualize_dct(dct_original, dct_bilinear_d, dct_bicubic_d, ratios):
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1), plt.imshow(dct_original, cmap='gray'), plt.title(f'Original DCT\nLow Energy Ratio: {ratios[0]:.2%}'), plt.axis('off')
    plt.subplot(1, 3, 2), plt.imshow(dct_bilinear_d, cmap='gray'), plt.title(f'Bilinear Restored DCT\nLow Energy Ratio: {ratios[1]:.2%}'), plt.axis('off')
    plt.subplot(1, 3, 3), plt.imshow(dct_bicubic_d, cmap='gray'), plt.title(f'Bicubic Restored DCT\nLow Energy Ratio: {ratios[2]:.2%}'), plt.axis('off')
    plt.tight_layout()
    plt.savefig("result_dct.png", dpi=150, bbox_inches='tight')
    plt.close()

# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    # 请把这里改成你的图片路径！
    img_path = "/home/chen/cv-course/test_image.jpg"
    
    original = load_gray_image(img_path)
    h, w = original.shape
    print(f"原始图像尺寸: {w}x{h}")

    scale = 0.5
    small_direct = downsample_direct(original, scale)
    small_blur = downsample_gaussian_blur(original, scale)
    print(f"下采样后尺寸: {int(w*scale)}x{int(h*scale)}")

    nearest_d, bilinear_d, bicubic_d = restore_image(small_direct, (h, w))
    nearest_b, bilinear_b, bicubic_b = restore_image(small_blur, (h, w))

    print("\n=== 直接下采样恢复结果（vs 原图） ===")
    mse_n_d, psnr_n_d = calculate_mse_psnr(original, nearest_d)
    mse_bi_d, psnr_bi_d = calculate_mse_psnr(original, bilinear_d)
    mse_bic_d, psnr_bic_d = calculate_mse_psnr(original, bicubic_d)
    print(f"最近邻: MSE={mse_n_d:.2f}, PSNR={psnr_n_d:.2f} dB")
    print(f"双线性: MSE={mse_bi_d:.2f}, PSNR={psnr_bi_d:.2f} dB")
    print(f"双三次: MSE={mse_bic_d:.2f}, PSNR={psnr_bic_d:.2f} dB")

    print("\n=== 高斯下采样恢复结果（vs 原图） ===")
    mse_n_b, psnr_n_b = calculate_mse_psnr(original, nearest_b)
    mse_bi_b, psnr_bi_b = calculate_mse_psnr(original, bilinear_b)
    mse_bic_b, psnr_bic_b = calculate_mse_psnr(original, bicubic_b)
    print(f"最近邻: MSE={mse_n_b:.2f}, PSNR={psnr_n_b:.2f} dB")
    print(f"双线性: MSE={mse_bi_b:.2f}, PSNR={psnr_bi_b:.2f} dB")
    print(f"双三次: MSE={mse_bic_b:.2f}, PSNR={psnr_bic_b:.2f} dB")

    fft_original = fft_analysis(original)
    fft_small_direct = fft_analysis(small_direct)
    fft_bilinear_d = fft_analysis(bilinear_d)

    dct_original, ratio_original = dct_analysis(original)
    dct_bilinear_d, ratio_bilinear = dct_analysis(bilinear_d)
    dct_bicubic_d, ratio_bicubic = dct_analysis(bicubic_d)
    ratios = [ratio_original, ratio_bilinear, ratio_bicubic]

    visualize_results(original, small_direct, small_blur, nearest_d, bilinear_d, bicubic_d, nearest_b, bilinear_b, bicubic_b)
    visualize_fft(fft_original, fft_small_direct, fft_bilinear_d)
    visualize_dct(dct_original, dct_bilinear_d, dct_bicubic_d, ratios)
    
    print("\n✅ 运行完成！已生成三张结果图片：")
    print("1. result_images.png")
    print("2. result_fft.png")
    print("3. result_dct.png")