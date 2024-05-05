# -*- coding: utf-8 -*-
"""
Created on Sat May  4 16:15:59 2024

@author: 19398
"""


from ultralytics import YOLO
import cv2
import numpy as np
def calculate_iou(box1, box2):
    
    x1_min, y1_min, x1_max, y1_max =[t for d2 in box1.xyxy.tolist() for t in d2]
    x2_min, y2_min, x2_max, y2_max =[t for d2 in box2.xyxy.tolist() for t in d2]
    
    # 计算交集的左上角和右下角坐标
    x_inter_min = max(x1_min, x2_min)
    y_inter_min = max(y1_min, y2_min)
    x_inter_max = min(x1_max, x2_max)
    y_inter_max = min(y1_max, y2_max)
    
    # 计算交集面积
    intersection_area = max(0, x_inter_max - x_inter_min + 1) * max(0, y_inter_max - y_inter_min + 1)
    
    # 计算并集面积
    union_area = (x1_max - x1_min + 1) * (y1_max - y1_min + 1) + (x2_max - x2_min + 1) * (y2_max - y2_min + 1) - intersection_area
    
    # 计算 IoU
    iou = intersection_area / union_area
    return iou

def remove_duplicate_boxes(boxes, iou_threshold):
    filtered_boxes = []
    for i, box1 in enumerate(boxes):
        is_duplicate = False
        for j, box2 in enumerate(filtered_boxes):
            iou = calculate_iou(box1, box2)
            if iou > iou_threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            filtered_boxes.append(box1)
    return filtered_boxes



# 计算红色占比(h值优秀占比)
def calculate_h_ratio(image, mask):
    # 将图像转换为HSV格式
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # 创建苹果部分的掩码
    apple_mask = cv2.bitwise_and(mask, mask, mask=mask)
    # 应用苹果部分的掩码
    apple_hsv = cv2.bitwise_and(img_hsv, img_hsv, mask=apple_mask)
    # 定义红色的HSV范围
    lower_red = np.array([0, 0, 128])
    upper_red = np.array([13, 255, 255])
    # 创建红色的掩码
    red_mask = cv2.inRange(apple_hsv, lower_red, upper_red)
    # 计算红色像素的数量
    red_pixels = cv2.countNonZero(red_mask)
    total_pixels = cv2.countNonZero(mask)
    h_ratio = red_pixels / total_pixels
    return h_ratio

def calculate_b_ratio(image,mask):
    # 将图像转换为HSV格式
    img_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    apple_mask = cv2.bitwise_and(mask, mask, mask=mask)
    # 应用苹果部分的掩码
    apple_hsv = cv2.bitwise_and(img_hsv, img_hsv, mask=apple_mask)
    # 定义棕色的HSV范围
    brown_lower = np.array([0, 0, 50])  # 棕色的下限HSV值
    brown_upper = np.array([13, 250, 150])  # 棕色的上限HSV值
    b_mask = cv2.inRange(apple_hsv, brown_lower, brown_upper)
   
    # 计算棕色像素的数量
    b_pixels = cv2.countNonZero(b_mask)
    total_pixels = cv2.countNonZero(mask)
    b_ratio =b_pixels / total_pixels
    return b_ratio


#计算h值分数
def get_h_score(h_ratio):
    if h_ratio>=0.5:
        return 100
    elif h_ratio>=0.3 and h_ratio<0.5:
        return 60+(h_ratio-0.03)*((100-60)/(0.5-0.03))
    else:
        return 0
    
def CannyThreshold(image, lowThreshold):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    detected_edges = cv2.GaussianBlur(gray, (3, 3), 0)
    detected_edges = cv2.Canny(detected_edges, lowThreshold, lowThreshold * 3, apertureSize=3)

    # 查找轮廓
    contours, _ = cv2.findContours(detected_edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 统计椭圆数量
    ellipse_count = 0
    for contour in contours:
        area = cv2.contourArea(contour)
        if 100 <= area <= 3000:  # 面积在100到3000之间
            perimeter = cv2.arcLength(contour, True)
            circularity = 4 * np.pi * area / (perimeter * perimeter)
            if 0.2 <= circularity <= 2.1:  # 圆度在0.2到2.1之间
                ellipse_count += 1
    return ellipse_count

#转换回原图
def write_attribute(image1,image2,h_score,ellipse_count,rank):
    # 将图片1转换为灰度图
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    # 在图片2中查找图片1的位置
    result = cv2.matchTemplate(cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY), gray1, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    # 获取图片1的宽度和高度
    h, w = gray1.shape
    # 图片1在图片2中的位置
    top_left = max_loc
    bottom_right = (top_left[0] + w, top_left[1] + h)

    # 在图片2中画出图片1的位置
    cv2.rectangle(image2, top_left, bottom_right, (0, 255, 0), 2)

    # 在图片2上添加文字“图片1”
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image2, f"score:{h_score}", (top_left[0], top_left[1] - 10), font, 0.7,(220,0,0), 2, cv2.LINE_AA)
    center_x = (top_left[0] + bottom_right[0]) // 2
    center_y = (top_left[1] + bottom_right[1]) // 2

   # 在图片1对应在图片2中的中央位置添加文字信息
    if h_score!=0:
        cv2.putText(image2, f"rk{rank}", (center_x, center_y), font, 1.0, (220, 0, 0), 2, cv2.LINE_AA)
    else:
        thickness = 15
        color = (0, 0, 0)  # 叉号颜色，这里使用黑色
        cv2.line(image2, (center_x - 20, center_y - 20), (center_x + 20, center_y + 20), color, thickness)
        cv2.line(image2, (center_x - 20, center_y + 20), (center_x + 20, center_y - 20), color, thickness)
    return image2

#计算排名
def get_rank(boxes,ori_img):
    ranks=[]
    h_scores0=[]
    for i in range(len(boxes)):
        xyxy1=boxes[i].xyxy.tolist()
        xyxy1=[t for d2 in xyxy1 for t in d2]
        print(xyxy1)
        apple1=ori_img[int(xyxy1[1]):int(xyxy1[3]),int(xyxy1[0]):int(xyxy1[2])]
        #cv2.imwrite(f'apple{i}.jpg',apple1)
        image=apple1
        # 转换为灰度图像
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 应用阈值处理
        _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

        # 轮廓检测
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # 创建掩码
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

        # 对图像应用掩码
        masked_image = cv2.bitwise_and(image, image, mask=mask)
        h_ratio = calculate_h_ratio(image, mask)
        b_ratio=calculate_b_ratio(image,mask)
        h_score=get_h_score(h_ratio)
        image=apple1
        if CannyThreshold(image, 62)>0:
            h_score=0
        if b_ratio>0.033:
            h_score=0
        ##h_score=round(h_score)
        h_scores0.append(h_score)
   
    sorted_a = sorted(h_scores0, reverse=True)
    ranks = [sorted_a.index(x) + 1 for x in h_scores0]
    return ranks,h_scores0
    
def main1(ori_img_path):
    model=YOLO('yolov8n.pt')
    results=model(ori_img_path)
   # path='Apple1.jpg'
   # image2=cv2.imread(path)
    h_scores=[]
    ellipse_counts=[]
    for result in results:
        ori_img=result.orig_img
        image2=ori_img
        boxes=result.boxes
        boxes=remove_duplicate_boxes(boxes, 0.5)
        ranks,h_scores=get_rank(boxes,ori_img)
        for i in range(len(boxes)):
            xyxy1=boxes[i].xyxy.tolist()
            xyxy1=[t for d2 in xyxy1 for t in d2]
            print(xyxy1)
            apple1=ori_img[int(xyxy1[1]):int(xyxy1[3]),int(xyxy1[0]):int(xyxy1[2])]
            #cv2.imwrite(f'apple{i}.jpg',apple1)
            image=apple1
            # 转换为灰度图像
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # 应用阈值处理
            _, thresh = cv2.threshold(gray, 110, 255, cv2.THRESH_BINARY)

            # 轮廓检测
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # 创建掩码
            mask = np.zeros_like(gray)
            cv2.drawContours(mask, contours, -1, (255), thickness=cv2.FILLED)

            # 对图像应用掩码
            masked_image = cv2.bitwise_and(image, image, mask=mask)
            ''' 
            h_ratio = calculate_h_ratio(image, mask)
            b_ratio=calculate_b_ratio(image,mask)
            h_score=get_h_score(h_ratio)
            image=apple1
            if CannyThreshold(image, 62)>0:
                h_score=0
            if b_ratio>0.033:
                h_score=0
            h_score=round(h_score)
            h_scores.append(h_score)
            '''
            h_scores[i]=round(h_scores[i])
            h_score=h_scores[i]
            image=apple1
            ellipse_count=CannyThreshold(image, 62)
            ellipse_counts.append(ellipse_count)
            image1=apple1
            rank=ranks[i]
            image2=write_attribute(image1,image2,h_score,ellipse_count,rank)
    final=[image2]
    for i in range(len(boxes)):
        final.append([h_scores[i],ellipse_counts[i]])
    return final

