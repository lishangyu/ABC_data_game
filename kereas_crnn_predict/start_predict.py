from PIL import Image
from model import *
import numpy as np
import argparse
import os
import cfg
import pandas as pd

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', '-p',
                        default=os.path.join(os.getcwd(),"../img_test/test1.png"),
                        help='image path')
    return parser.parse_args()


args = parse_args()
img_path = args.path
if not os.path.isdir(img_path):  # 如果是单张文件就检测一张
    if not os.path.exists(img_path):
        print('image {} is not found'.format(img_path))
        exit()
    im = Image.open(img_path)
    img = np.array(im.convert('RGB'))
    print(predict(im))
    print('done')
else:  # 如果输入是一个文件夹，就检测文件夹下所有的图片
    image_list = []
    g = os.walk(img_path)
    print('process dir = {}'.format(img_path))
    for path, d, filelist in g:
        for filename in filelist:
            if filename.endswith('jpg') or filename.endswith('png') or filename.endswith('JPG'):
                image_list.append(os.path.join(path, filename))
    sum_image = len(image_list)
    print('total image num = {}'.format(sum_image))
    i = 0
    df = pd.DataFrame(columns=('file_name', 'result'))
    for image_file in image_list:
        im = Image.open(image_file)
        img = np.array(im.convert('RGB'))
        result=predict(im)
        if (i+1)%1000==0:
            print("we have processed {} images".format(i+1))
        file_name=os.path.basename(image_file)
        df.loc[i] = [file_name,result]
        i=i+1
    df.to_csv(cfg.res_saved_path, index=False, header=False)
    print('done')
    print('result saved at {}'.format(cfg.res_saved_path))