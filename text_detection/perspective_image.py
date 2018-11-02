#coding=utf-8
import numpy as np
import cv2
from matplotlib import pyplot as plt
import os
import cmath
import cfg



def cal_dis(p1,p2,p3,p4):
    return int(cmath.sqrt(pow((p1-p3),2)+pow((p2-p4),2)).real)

def gen_pts2(cut_h,cut_w):
    pts2 = np.float32([[0, 0], [0, cut_h], [cut_w, cut_h], [cut_w, 0]])
    return pts2



def perspective_image(txt_file,image_file):
    file_object = open(txt_file, 'rU')
    ori_img = cv2.imread(image_file)
    for line in file_object:
        line=line.split(',')
        for i, val in enumerate(line):
            line[i]=float(val)
        pts1 = np.float32([[line[0], line[1]], [line[2], line[3]], [line[4], line[5]], [line[6], line[7]]])
        cut_h=  cal_dis(line[0],line[1],line[2],line[3])
        cut_w = cal_dis(line[0], line[1], line[6], line[7])
        pts2=gen_pts2(cut_h,cut_w)
        M_perspective = cv2.getPerspectiveTransform(pts1, pts2)
        img_perspective = cv2.warpPerspective(ori_img, M_perspective, (cut_w, cut_h))
        out_image_file = os.path.join(cfg.save_transform_img_path,os.path.splitext(os.path.basename(image_file))[0]+ "_crop.jpg")
        cv2.imwrite(out_image_file,img_perspective)
    file_object.close()
