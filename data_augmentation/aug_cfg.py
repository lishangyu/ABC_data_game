
img_path='../img_test'
out_path='aug_images'

move_off=[]#将每一张图像分别像上下左右平移1到10个像素
ran_num=25#随即颜色变化次数
for i in range(1,10):
    move_off.append(i)