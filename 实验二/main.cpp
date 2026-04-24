#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>

using namespace cv;
using namespace std;

string matTypeToString(int type) {
    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    string r;
    switch (depth) {
        case CV_8U:  r = "CV_8U"; break;
        case CV_8S:  r = "CV_8S"; break;
        case CV_16U: r = "CV_16U"; break;
        case CV_16S: r = "CV_16S"; break;
        case CV_32S: r = "CV_32S"; break;
        case CV_32F: r = "CV_32F"; break;
        case CV_64F: r = "CV_64F"; break;
        default:     r = "USER"; break;
    }
    r += "C";
    r += (chans+'0');
    return r;
}

int main() {
    // 绝对路径，100% 可读
    string img_path = "/home/hql/cv-course/experiment2/test_image.jpg";
    Mat src_img = imread(img_path, IMREAD_COLOR);

    if (src_img.empty()) {
        cerr << "错误：无法读取图片，请检查路径/文件是否存在！" << endl;
        // 打印当前目录，方便调试
        system("pwd");
        return -1;
    }

    cout << "===== 图像基本信息 =====" << endl;
    cout << "宽度 (cols): " << src_img.cols << endl;
    cout << "高度 (rows): " << src_img.rows << endl;
    cout << "通道数: " << src_img.channels() << endl;
    cout << "像素类型: " << matTypeToString(src_img.type()) << endl;

    namedWindow("Original Image", WINDOW_NORMAL);
    imshow("Original Image", src_img);
    waitKey(0);
    destroyAllWindows();

    Mat gray_img;
    cvtColor(src_img, gray_img, COLOR_BGR2GRAY);

    namedWindow("Grayscale Image", WINDOW_NORMAL);
    imshow("Grayscale Image", gray_img);
    waitKey(0);
    destroyAllWindows();

    string gray_path = "/home/hql/cv-course/experiment2/gray_test.jpg";
    imwrite(gray_path, gray_img);
    cout << "灰度图已保存至: " << gray_path << endl;

    int cx = src_img.cols / 2, cy = src_img.rows / 2;
    Vec3b pixel = src_img.at<Vec3b>(cy, cx);
    cout << "\n===== 中心像素 (BGR) =====" << endl;
    cout << "B: " << (int)pixel[0] << " G: " << (int)pixel[1] << " R: " << (int)pixel[2] << endl;

    Mat crop = src_img(Rect(0, 0, 200, 200));
    string crop_path = "/home/hql/cv-course/experiment2/crop_test.jpg";
    imwrite(crop_path, crop);
    cout << "裁剪图已保存至: " << crop_path << endl;

    namedWindow("Cropped Image", WINDOW_NORMAL);
    imshow("Cropped Image", crop);
    waitKey(0);
    destroyAllWindows();

    cout << "\n所有任务执行完成！" << endl;
    return 0;
}