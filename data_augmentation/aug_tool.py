
import io,os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image,ImageEnhance,ImageChops
import cv2
import random
import aug_cfg
#root_path为图像根目录，img_name为图像名字

def move(img_name,off): #平移，平移尺度为off
    img = Image.open(img_name)
    for i in off:
        region = img.crop((i, 0, img.size[0], img.size[1]))
        offset = region.convert('RGB')
        save_name='{}_off_l{}.jpg'.format(os.path.splitext(os.path.basename(img_name))[0],i)
        offset.save(os.path.join(aug_cfg.out_path, save_name))
    for i in off:
        region = img.crop((0, 0, img.size[0]-i, img.size[1]))
        offset = region.convert('RGB')
        save_name='{}_off_r{}.jpg'.format(os.path.splitext(os.path.basename(img_name))[0],i)
        offset.save(os.path.join(aug_cfg.out_path, save_name))
    for i in off:
        region = img.crop((0, i, img.size[0], img.size[1]))
        offset = region.convert('RGB')
        save_name='{}_off_u{}.jpg'.format(os.path.splitext(os.path.basename(img_name))[0],i)
        offset.save(os.path.join(aug_cfg.out_path, save_name))
    for i in off:
        region = img.crop((0, 0, img.size[0], img.size[1]-i))
        offset = region.convert('RGB')
        save_name='{}_off_d{}.jpg'.format(os.path.splitext(os.path.basename(img_name))[0],i)
        offset.save(os.path.join(aug_cfg.out_path, save_name))


def flip(root_path,img_name):   #翻转图像
    img = Image.open(os.path.join(root_path, img_name))
    filp_img = img.transpose(Image.FLIP_LEFT_RIGHT)
    # filp_img.save(os.path.join(root_path,img_name.split('.')[0] + '_flip.jpg'))
    return filp_img

def aj_contrast(root_path,img_name): #调整对比度 两种方式 gamma/log
    image = skimage.io.imread(os.path.join(root_path, img_name))
    gam= skimage.exposure.adjust_gamma(image, 0.5)
    # skimage.io.imsave(os.path.join(root_path,img_name.split('.')[0] + '_gam.jpg'),gam)
    log= skimage.exposure.adjust_log(image)
    # skimage.io.imsave(os.path.join(root_path,img_name.split('.')[0] + '_log.jpg'),log)
    return gam,log
def rotation(root_path, img_name):
    img = Image.open(os.path.join(root_path, img_name))
    rotation_img = img.rotate(90) #旋转角度
    # rotation_img.save(os.path.join(root_path,img_name.split('.')[0] + '_rotation.jpg'))
    return rotation_img

def randomGaussian(root_path, img_name, mean, sigma):  #高斯噪声
    image = Image.open(os.path.join(root_path, img_name))
    im = np.array(image)
    #设定高斯函数的偏移
    means = 0
    #设定高斯函数的标准差
    sigma = 25
    #r通道
    r = im[:,:,0].flatten()

    #g通道
    g = im[:,:,1].flatten()

    #b通道
    b = im[:,:,2].flatten()

    #计算新的像素值
    for i in range(im.shape[0]*im.shape[1]):

        pr = int(r[i]) + random.gauss(0,sigma)

        pg = int(g[i]) + random.gauss(0,sigma)

        pb = int(b[i]) + random.gauss(0,sigma)

        if(pr < 0):
            pr = 0
        if(pr > 255):
            pr = 255
        if(pg < 0):
            pg = 0
        if(pg > 255):
            pg = 255
        if(pb < 0):
            pb = 0
        if(pb > 255):
            pb = 255
        r[i] = pr
        g[i] = pg
        b[i] = pb
    im[:,:,0] = r.reshape([im.shape[0],im.shape[1]])

    im[:,:,1] = g.reshape([im.shape[0],im.shape[1]])

    im[:,:,2] = b.reshape([im.shape[0],im.shape[1]])
    gaussian_image = gaussian_image = Image.fromarray(np.uint8(im))
    return gaussian_image
def randomColor(img_name,num): #随机颜色
    """
    对图像进行颜色抖动
    :param image: PIL的图像image
    :return: 有颜色色差的图像image
    """
    for i in range(num):
        image = Image.open( img_name)
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        color_image = ImageEnhance.Color(image).enhance(random_factor)  # 调整图像的饱和度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        brightness_image = ImageEnhance.Brightness(color_image).enhance(random_factor)  # 调整图像的亮度
        random_factor = np.random.randint(10, 21) / 10.  # 随机因子
        contrast_image = ImageEnhance.Contrast(brightness_image).enhance(random_factor)  # 调整图像对比度
        random_factor = np.random.randint(0, 31) / 10.  # 随机因子
        res_image=ImageEnhance.Sharpness(contrast_image).enhance(random_factor)  # 调整图像锐度
        res_image = res_image.convert('RGB')
        save_name = '{}_ran_{}.jpg'.format(os.path.splitext(os.path.basename(img_name))[0], i)
        res_image.save(os.path.join(aug_cfg.out_path, save_name))