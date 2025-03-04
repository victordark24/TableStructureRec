import cv2
import numpy as np
import math

def detect_and_correct_skew(image_path, output_path):
    # 1、灰度化读取文件，
    img = cv2.imread(image_path, 0)

    # 2、图像延扩
    h, w = img.shape[:2]
    new_h = cv2.getOptimalDFTSize(h)
    new_w = cv2.getOptimalDFTSize(w)
    right = new_w - w
    bottom = new_h - h
    nimg = cv2.copyMakeBorder(img, 0, bottom, 0, right, borderType=cv2.BORDER_CONSTANT, value=0)

    # 3、执行傅里叶变换，并过得频域图像
    f = np.fft.fft2(nimg)
    fshift = np.fft.fftshift(f)
    magnitude = np.log(np.abs(fshift))

    # 二值化
    magnitude_uint = magnitude.astype(np.uint8)
    ret, thresh = cv2.threshold(magnitude_uint, 11, 255, cv2.THRESH_BINARY)

    # 霍夫直线变换
    lines = cv2.HoughLinesP(thresh, 2, np.pi / 180, 30, minLineLength=40, maxLineGap=100)

    # 创建一个新图像，标注直线
    lineimg = np.ones(nimg.shape, dtype=np.uint8)
    lineimg = lineimg * 255

    piThresh = np.pi / 180
    pi2 = np.pi / 2

    angle_list = []
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(lineimg, (x1, y1), (x2, y2), (0, 255, 0), 2)
        if x2 - x1 == 0:
            continue
        else:
            theta = (y2 - y1) / (x2 - x1)
        if abs(theta) < piThresh or abs(theta - pi2) < piThresh:
            continue
        else:
            angle = math.atan(theta)
            angle = angle * (180 / np.pi)
            if 0 < angle < 90:
                angle_list.append(angle)

    angle_list = np.array(angle_list)

    # 计算均值和标准差
    mean = np.mean(angle_list)
    std_dev = np.std(angle_list)

    # 定义阈值，例如假设是均值加减两倍标准差
    threshold = 2 * std_dev

    # 根据阈值剔除异常值
    angle_list = [x for x in angle_list if abs(x - mean) < threshold]

    avg_angle = np.mean(angle_list)
    # 以45度为界限，小于45度就正常转，大于45度就反转
    if avg_angle > 45:
        avg_angle = avg_angle - 90
    print(f"Detected skew angle: {avg_angle:.2f} degrees")

    image = cv2.imread(image_path)
    # 矫正图像
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, avg_angle, 1.0)
    corrected_image = cv2.warpAffine(image, rotation_matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # 保存结果
    cv2.imwrite(output_path, corrected_image)
    print(f"Corrected image saved to {output_path}")

