from scipy import integrate
import math


def get_cross_points(a, b, c, image_height):
    """solve the quadratic equation x = ay^2 + by + c"""
    assert a
    d = b**2 - 4*a*c
    if d < 0:
        return (0,None)
    elif d == 0:
        y = -b / (2 * a)
        if (y < image_height) and (y >= 0):
            return (0,y)
        else:
            return (0,None)
    else:
        y1 = (-b + math.sqrt(d)) / (2 * a)
        y2 = (-b - math.sqrt(d)) / (2 * a)
        if (y1 < image_height) and (y1 >= 0) and (y2 < image_height) and (y2 >= 0) :
            return (2,(y1, y2))
        elif (y1 < image_height) and (y1 >= 0):
            return (1, y1)
        elif (y2 < image_height) and (y2 >= 0):
            return (1, y2)
        else:
            return (0, None)


def calculate_cross_area(fit_line, real_line, image_height):
    """
    if the cross area is positive, it's FN area
        if the cross area is negative, it's FP area
    """
    a = real_line[0] - fit_line[0]
    b = real_line[1] - fit_line[1]
    c = real_line[2] - fit_line[2]
    num, cross_points_yvalue = get_cross_points(a, b, c, image_height)
    area_func = lambda x: a*x**2 + b*x + c
    if num == 0:
        area_tmp, error = integrate.quad(area_func, image_height, 0)
        if area_tmp > 0:
            FN_area = area_tmp
            FP_area = 0
        else:
            FN_area = 0
            FP_area = - area_tmp
    elif num == 1:
        area_tmp1, error = integrate.quad(area_func, cross_points_yvalue, image_height)
        area_tmp2, error = integrate.quad(area_func, 0, cross_points_yvalue)
        if area_tmp1 > 0:
            FN_area = area_tmp1
            FP_area = -area_tmp2
        else:
            FN_area = area_tmp2
            FP_area = -area_tmp1
    else:  #num == 2:
        cross_points_max = max(cross_points_yvalue[0], cross_points_yvalue[1])
        cross_points_min = min(cross_points_yvalue[0], cross_points_yvalue[1])
        area_tmp1, error = integrate.quad(area_func, image_height, cross_points_max)
        area_tmp2, error = integrate.quad(area_func, cross_points_max, cross_points_min)
        area_tmp3, error = integrate.quad(area_func, cross_points_min, 0)
        if area_tmp1 > 0:
            FN_area = area_tmp1 + area_tmp3
            FP_area = - area_tmp2
        else:
            FN_area = area_tmp2
            FP_area = -(area_tmp1 + area_tmp3)

    return FN_area, FP_area


def calculate_precision_recall(fit_lines, real_lines, image_height):
    """
    一种车道线识别的评估算法, 输入为根据算法在透视域的拟合车道线和真实车道线
    算法识别的车道和真实车道重合的车道区域为TP_area, 算法识别但错误的车道区域为FN_area,算法没识别到但正确的区域为FP_area
    根据几个区域的面积来计算precision和recall
    具体步骤参看专利
    下列代码只实现了较为理想情况下的评估
    """

    """calculate the cross points of two lines
        calculate the corss area by making integration of two lines
        if the cross area is positive, it's FN area
        if the cross area is negative, it's FP area
    """
    left_fit_line = fit_lines[0]
    left_real_line = real_lines[0]
    left_FN_area, left_FP_area = calculate_cross_area(left_fit_line, left_real_line, image_height)

    right_fit_line = fit_lines[1]
    right_real_line = real_lines[1]
    right_FN_area, right_FP_area = calculate_cross_area(right_fit_line, right_real_line, image_height)

    """the true lane area between left_real_line to right_real_line"""
    _, true_lane_area =  calculate_cross_area(left_real_line, right_real_line, image_height)

    """calculate FN, FP, TP"""
    FN_area = left_FN_area + right_FN_area
    FP_area = left_FP_area + right_FP_area
    TP_area = true_lane_area - FP_area

    """calculate precision and recall"""
    precision = TP_area / (TP_area + FP_area)
    recall = TP_area / (TP_area + FN_area)
    return precision, recall


