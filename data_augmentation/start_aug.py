
import os
import aug_tool
import aug_cfg

img_path=aug_cfg.img_path


def do_aug(image):
    aug_tool.move(image,aug_cfg.move_off)


image_list = []
g = os.walk(img_path)
print('process dir = {}'.format(img_path))
for path, d, filelist in g:
    for filename in filelist:
        if filename.endswith('jpg') or filename.endswith('png') or filename.endswith('JPG'):
            image_list.append(os.path.join(path, filename))
sum_image = len(image_list)
print('total image num = {}'.format(sum_image))
i = 1
for image_file in image_list:
    if i % 1000 == 0:
        print('process {} images, total {} images', i, sum_image)
    do_aug(image_file)
    aug_tool.randomColor(image_file,aug_cfg.ran_num)
print('done')