# -*- coding: utf-8 -*-
# @File : utils.py 
# @Author : xhj
# @Time : 2021/5/1 17:14


import os
import cv2
import numpy as np
from collections import deque


def find_circles(image, mask=None, threshold=200, nmax=100,
                 rmin=5, rmax=10, rstd=1, rpan=4):
    """
    基于hough梯度变换在图片中找到圆形所在的位置信息
    Args:
        image: 输入图片，BGR格式的彩色图
        mask:
        threshold: int类型，表示筛选circles时score的最小值
        nmax: int类型，表示筛选出权重最高的前nmax个circles
        rmin: int类型，半径的最小值
        rmax: int类型，半径的最大值
        rstd:
        rpan:
    """
    if rmin < rpan + 1:
        raise ValueError

    if mask is not None:
        rmin = max(rmin, int(np.ceil(np.min(mask) - rstd)))
        rmax = min(rmax, int(np.floor(np.max(mask) + rstd)))

    if rmin > rmax:
        return [], []

    # 基于灰度图产生梯度信息
    Dx = cv2.Scharr(image, cv2.CV_32F, 1, 0)
    Dy = cv2.Scharr(image, cv2.CV_32F, 0, 1)
    Da = np.arctan2(Dy, Dx) * 2  # np.arctan2 的取值范围是 -pi ~ pi
    Ds = np.log1p(np.hypot(Dy, Dx))  # = log(1+sqrt(Dy^2+Dx^2))
    Du = np.sum(np.cos(Da) * Ds, axis=-1)  # 将 BGR 三个通道的值进行合并
    Dv = np.sum(np.sin(Da) * Ds, axis=-1)

    # calculate likelihood for each (x, y, r) pair
    # based on: gradient changes across circle
    def iter_scores():
        queue = deque()
        for radius in range(rmin - rpan, rmax + rpan + 1):
            r = int(np.ceil(radius + 6 + rstd * 4))
            Ky, Kx = np.mgrid[-r: r + 1, -r: r + 1]
            Ka = np.arctan2(Ky, Kx) * 2
            Ks = np.exp(np.square(np.hypot(Ky, Kx) - radius) /
                        (-2 * rstd ** 2)) / np.sqrt(radius)
            Ku = np.cos(Ka) * Ks
            Kv = np.sin(Ka) * Ks
            queue.append(cv2.filter2D(Du, cv2.CV_32F, Ku) +
                         cv2.filter2D(Dv, cv2.CV_32F, Kv))
            if len(queue) > rpan * 2:
                yield (radius - rpan, queue[rpan] -
                       (np.fmax(0, queue[0]) + np.fmax(0, queue[rpan * 2])))
                queue.popleft()

    # choose best (x, y, r) for each (x, y)
    radiuses = np.zeros(image.shape[:2], dtype=int)
    scores = np.full(image.shape[:2], -np.inf)
    for radius, score in iter_scores():
        sel = (score > scores)
        if mask is not None:
            sel &= (mask > radius - rstd) & (mask < radius + rstd)
        scores[sel] = score[sel]
        radiuses[sel] = radius

    # choose the top n circles
    circles = []
    weights = []
    for _ in range(nmax):
        # 在每一步循环中找到 scores 矩阵中的最大值的坐标，np.argmax 返回的是
        # 数组拉平之后的最大值下标，需要使用 np.unravel_index 转换为原数组中
        # 的最大值的下标，y是行下标，x是列下标
        y, x = np.unravel_index(np.argmax(scores), scores.shape)
        score = scores[y, x]
        if score < threshold:  # score是当前最大值，如果该值小于阈值，说明没必要继续后面的循环
            break
        r = radiuses[y, x]
        circles.append((x, y, r))
        weights.append(score)
        cv2.circle(scores, (x, y), r, 0, -1)
    return circles, weights


def draw_circles(img, circles):
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    for (x, y, r) in circles:
        cv2.circle(img, (x, y), int(r), (0, 0, 255), 2)
        cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
    return img


def find_central_point(image, reverse_color=True):
    """
    通过质心法计算中心点坐标
    Args:
        image: 输入的原始图像，需要是灰度图，黑色背景，且只包含一个中心
        reverse_color: bool类型，表示是否需要将颜色进行反转（如果是白色背景，则需要反转）
    """
    gray_image = image.copy()
    if gray_image.ndim > 2:
        gray_image = cv2.cvtColor(gray_image, cv2.COLOR_BGR2GRAY)
    if reverse_color:
        gray_image = 255 - gray_image
    x_sum = np.sum(gray_image, axis=0)
    y_sum = np.sum(gray_image, axis=1)
    x_central = np.sum(np.arange(0, len(x_sum), 1) * x_sum) / x_sum.sum()
    y_central = np.sum(np.arange(0, len(y_sum), 1) * y_sum) / y_sum.sum()
    return x_central, y_central


def find_calib_points(image, point_size):
    """
    从图片中找到用于标定的关键点坐标
    Args:
        image: 输入的标定图片
        point_size: 标定点数目, (w_num, h_num)
    """
    pass


def find_contour(image, min_size, max_size):
    """
    从图片中查找轮廓
    Args:
        image: 输入的图片
        min_size: 轮廓的最小尺寸
        max_size: 轮廓的最大尺寸
    """
    pass


if __name__ == "__main__":
    img_file_path = "calib_board_imgs/3.png"
    img = cv2.imread(img_file_path, cv2.IMREAD_COLOR)
    img = cv2.resize(img, (40, 30))
    cv2.imshow("img", img)
    img = cv2.resize(img, (480, 360), cv2.INTER_NEAREST)
    cv2.imshow("src img", img)
    print(img.shape)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1,
                               10, param1=80, param2=22, minRadius=30, maxRadius=60)
    # circles2, _ = find_circles(img, None, 40, 6, 20, 60)
    # print(circles2)
    circles = np.squeeze(circles)
    circle_img = draw_circles(img, circles)

    central_point = find_central_point(circle_img, True)
    print(central_point)

    cv2.imshow("circle img", circle_img)
    cv2.waitKey(0)
