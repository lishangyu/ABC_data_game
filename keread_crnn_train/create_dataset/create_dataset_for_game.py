
import glob
import random
import os
import create_dataset

##lmdb 输出目录
create_val=True #是否制作验证集

outputPath_train = '../data/lmdb/train' #训练集输出路径
outputPath_val = '../data/lmdb/val' #验证集输出路径
val_per=0.1#验证集比例
path = '../data/dataline/*.jpg' #读入图像路径
imagePathList = glob.glob(path)
total_image=len(imagePathList)
val_num=int(val_per*total_image)
print('------------', total_image, 'images ------------')

random.shuffle(imagePathList)#打乱顺序

if create_val == False:

    imgLabelLists = []
    for p in imagePathList:
        try:
            imgLabelLists.append((p, create_dataset.read_text(p.replace('.jpg', '.txt'))))
        except:
            continue

    # imgLabelList = [ (p,read_text(p.replace('.jpg','.txt'))) for p in imagePathList]
    ##sort by lebelList
    imgLabelList = sorted(imgLabelLists, key=lambda x: len(x[1]))
    imgPaths = [p[0] for p in imgLabelList]
    txtLists = [p[1] for p in imgLabelList]

    create_dataset.createDataset(outputPath_train, imgPaths, txtLists, lexiconList=None, checkValid=True)

else:
    '''
    create train data
    '''
    val_image_list=imagePathList[:,val_num]
    train_image_list=imagePathList[val_num,:]
    print('val have {} images'.format(len(val_image_list)))
    print('train have {} images'.format(len(train_image_list)))
    imgLabelLists = []
    for p in train_image_list:
        try:
            imgLabelLists.append((p, create_dataset.read_text(p.replace('.jpg', '.txt'))))
        except:
            continue

    # imgLabelList = [ (p,read_text(p.replace('.jpg','.txt'))) for p in imagePathList]
    ##sort by lebelList
    imgLabelList = sorted(imgLabelLists, key=lambda x: len(x[1]))
    imgPaths = [p[0] for p in imgLabelList]
    txtLists = [p[1] for p in imgLabelList]

    create_dataset.createDataset(outputPath_train, imgPaths, txtLists, lexiconList=None, checkValid=True)
    '''
        create val data
    '''
    imgLabelLists = []
    for p in val_image_list:
        try:
            imgLabelLists.append((p, create_dataset.read_text(p.replace('.jpg', '.txt'))))
        except:
            continue

    # imgLabelList = [ (p,read_text(p.replace('.jpg','.txt'))) for p in imagePathList]
    ##sort by lebelList
    imgLabelList = sorted(imgLabelLists, key=lambda x: len(x[1]))
    imgPaths = [p[0] for p in imgLabelList]
    txtLists = [p[1] for p in imgLabelList]

    create_dataset.createDataset(outputPath_val, imgPaths, txtLists, lexiconList=None, checkValid=True)