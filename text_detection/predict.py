import argparse

import numpy as np
import os
from PIL import Image, ImageDraw
from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input

import cfg
from perspective_image import *
from label import point_inside_of_quad
from network import East
from preprocess import resize_image
from nms import nms


def sigmoid(x):
    """`y = 1 / (1 + exp(-x))`"""
    return 1 / (1 + np.exp(-x))


def cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array, img_path, s):
    geo /= [scale_ratio_w, scale_ratio_h]
    p_min = np.amin(geo, axis=0)
    p_max = np.amax(geo, axis=0)
    min_xy = p_min.astype(int)
    max_xy = p_max.astype(int) + 2
    sub_im_arr = im_array[min_xy[1]:max_xy[1], min_xy[0]:max_xy[0], :].copy()
    for m in range(min_xy[1], max_xy[1]):
        for n in range(min_xy[0], max_xy[0]):
            if not point_inside_of_quad(n, m, geo, p_min, p_max):
                sub_im_arr[m - min_xy[1], n - min_xy[0], :] = 255
    sub_im = image.array_to_img(sub_im_arr, scale=False)
    sub_im.save(img_path + '_subim%d.jpg' % s)


def predict(east_detect, img_path, pixel_threshold, quiet=False):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)#576*576*3
    #print(img.shape)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)
    #print(y.shape)#1*144*144*7
    y = np.squeeze(y, axis=0)#144*144*7
    '''
    # add by lsy
    # 前两维对应原图大小进行放缩后的感受野 最后一个维度是每一个感受野上的结果
    # 输出层分别是1位score map, 是否在文本框内；2位vertex code，是否属于文本框边界像素以及是头还是尾；
    # 4位geo，是边界像素可以预测的2个顶点坐标。
    # 左上x 左上y 左下x 左下y （如果像素在右侧就计算右侧的)
    #https://huoyijie.github.io/zh-Hans/2018/08/24/AdvancedEAST%E6%96%87%E6%9C%AC%E6%A3%80%E6%B5%8B%E5%8E%9F%E7%90%86%E7%AE%80%E4%BB%8B/
    '''
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)#符合条件的坐标 activation_pixels[0]=x activation_pixels[1]=y
    quad_scores, quad_after_nms = nms(y, activation_pixels)
    #quad_scores （3*4）第一个维度是多少个框（最终） 分别是4个端点的平均得分？ 为了验证有没有分数<0?
    # quad_after_nms （3*4*2）第一个维度是多少个框（最终） 后面是4个二维的tupe，表示每个框顶点坐标（x，y）
    #print(quad_scores.shape)
    #print(quad_after_nms.shape)
    with Image.open(img_path) as im:
        im_array = image.img_to_array(im.convert('RGB'))
        d_wight, d_height = resize_image(im, cfg.max_predict_img_size)
        scale_ratio_w = d_wight / im.width
        scale_ratio_h = d_height / im.height
        im = im.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
        quad_im = im.copy()
        if cfg.predict_draw_act == True:
            draw = ImageDraw.Draw(im)
            for i, j in zip(activation_pixels[0], activation_pixels[1]):
                px = (j + 0.5) * cfg.pixel_size
                py = (i + 0.5) * cfg.pixel_size
                line_width, line_color = 1, 'red'
                if y[i, j, 1] >= cfg.side_vertex_pixel_threshold:#是否是头或者尾
                    if y[i, j, 2] < cfg.trunc_threshold:
                        line_width, line_color = 2, 'yellow'#头
                    elif y[i, j, 2] >= 1 - cfg.trunc_threshold:
                        line_width, line_color = 2, 'green'#尾
                draw.line([(px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                           (px + 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size),
                           (px + 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                           (px - 0.5 * cfg.pixel_size, py + 0.5 * cfg.pixel_size),
                           (px - 0.5 * cfg.pixel_size, py - 0.5 * cfg.pixel_size)],
                          width=line_width, fill=line_color)
            im.save(os.path.join(cfg.save_img_path,os.path.splitext(os.path.basename(img_path))[0] + '_act.jpg'))
        if cfg.predict_draw_predict == True:
            quad_draw = ImageDraw.Draw(quad_im)
            txt_items = []
            for score, geo, s in zip(quad_scores, quad_after_nms,
                                     range(len(quad_scores))):


                if np.amin(score) > 0:
                    quad_draw.line([tuple(geo[0]),
                                    tuple(geo[1]),
                                    tuple(geo[2]),
                                    tuple(geo[3]),
                                    tuple(geo[0])], width=2, fill='red')
                    if cfg.predict_cut_text_line:
                        cut_text_line(geo, scale_ratio_w, scale_ratio_h, im_array,
                                      img_path, s)
                    rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
                    rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
                    txt_item = ','.join(map(str, rescaled_geo_list))
                    txt_items.append(txt_item + '\n')
                elif not quiet:
                    print('quad invalid with vertex num less then 4.')
            quad_im.save(os.path.join(cfg.save_img_path,os.path.splitext(os.path.basename(img_path))[0] + '_predict.jpg'))
        if cfg.predict_write2txt and len(txt_items) > 0:
            txt_file=os.path.join(cfg.save_txt_path,os.path.splitext(os.path.basename(img_path))[0] + '.txt')
            with open(txt_file, 'w') as f_txt:
                f_txt.writelines(txt_items)
            f_txt.close()
            if cfg.do_transform == True:
                perspective_image(txt_file,img_path)


def predict_txt(east_detect, img_path, txt_path, pixel_threshold, quiet=False):
    img = image.load_img(img_path)
    d_wight, d_height = resize_image(img, cfg.max_predict_img_size)
    scale_ratio_w = d_wight / img.width
    scale_ratio_h = d_height / img.height
    img = img.resize((d_wight, d_height), Image.NEAREST).convert('RGB')
    img = image.img_to_array(img)
    img = preprocess_input(img, mode='tf')
    x = np.expand_dims(img, axis=0)
    y = east_detect.predict(x)

    y = np.squeeze(y, axis=0)
    y[:, :, :3] = sigmoid(y[:, :, :3])
    cond = np.greater_equal(y[:, :, 0], pixel_threshold)
    activation_pixels = np.where(cond)
    quad_scores, quad_after_nms = nms(y, activation_pixels)

    txt_items = []
    for score, geo in zip(quad_scores, quad_after_nms):
        if np.amin(score) > 0:
            rescaled_geo = geo / [scale_ratio_w, scale_ratio_h]
            rescaled_geo_list = np.reshape(rescaled_geo, (8,)).tolist()
            txt_item = ','.join(map(str, rescaled_geo_list))
            txt_items.append(txt_item + '\n')
        elif not quiet:
            print('quad invalid with vertex num less then 4.')
    if cfg.predict_write2txt and len(txt_items) > 0:
        with open(txt_path, 'w') as f_txt:
            f_txt.writelines(txt_items)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p',
                        default='demo/012.png',
                        help='image path')
    parser.add_argument('--threshold', '-t',
                        default=cfg.pixel_threshold,
                        help='pixel activation threshold')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    img_path = args.path
    threshold = float(args.threshold)
    print('path = {}'.format(img_path))
    img_path='demo/'
    east = East()
    east_detect = east.east_network()
    east_detect.load_weights(cfg.saved_model_weights_file_path)
    if not os.path.isdir(img_path):#如果是单张文件就检测一张
        if not os.path.exists(img_path):
            print('image {} is not found'.format(img_path))
            exit()
        predict(east_detect, img_path, threshold)
        print('done')
    else:#如果输入是一个文件夹，就检测文件夹下所有的图片
        image_list=[]
        g = os.walk(img_path)
        print('process dir = {}'.format(img_path))
        for path, d, filelist in g:
            for filename in filelist:
                if filename.endswith('jpg') or filename.endswith('png') or filename.endswith('JPG'):
                    image_list.append(os.path.join(path,filename))
        sum_image=len(image_list)
        print('total image num = {}' .format(sum_image))
        i=1
        for image_file in image_list:
            if i%1000 ==0:
                print('process {} images, total {} images',i,sum_image)
            predict(east_detect, image_file, threshold)
        print('done')