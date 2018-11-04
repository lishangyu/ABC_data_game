
img_path='../img_test' #读入路径
out_path='aug_images' #输出路径

move_off=[]#将每一张图像分别像上下左右平移1到10个像素
ran_num=25#随即颜色变化次数
for i in range(1,10):
    move_off.append(i)