# ABC_data_game

venv文件夹中是相关的python包，如果不在本地跑的话就不要下载了，很大

OCR分为两个步骤

1.文本检测（输出是文本的包围框）

2.文本识别

由于这次数据集的图像基本上已经是剪裁好的图像（但有些还是有较大的倾斜/旋转）
，所以是否需要进行文本检测**有待实验**

@[TOC]

---

# 1.文本检测

**text_detection**文件夹中提供了文本检测工具，它可以获取图像中文本的包围框坐标
，并根据透视变换，
将文本框位置的图像扣出并转换到一个完全水平的矩形形状。

**crop_img**-----------存储了最终扣出后的图像

**detect_result_img**--存储了文本检测的可视化图像（没用，只是可以看看结果好不好）

**detect_result_txt**--存储了每一个图像的对应的文本框的顶点坐标

使用方法 python -p '*anypath*'

其中如果*anypath*是一个图像，他就会对图像进行文本检测

如果是一个路径，会对该路径下所有的图像进行文本检测

**cfg.py**是配置文件

原理是advanced—EAST  [这里看原理](https://huoyijie.github.io/zh-Hans/2018/08/24/AdvancedEAST%E6%96%87%E6%9C%AC%E6%A3%80%E6%B5%8B%E5%8E%9F%E7%90%86%E7%AE%80%E4%BB%8B/)

---

# 2.文本识别

（原理是CRNN CNN提特征+双向双层lstm（对lstm的每一个输入都是一个（字符集长度+1）的分类问题）+转录）

[论文](https://blog.csdn.net/qq314000558/article/details/83110225)

[讲解](https://blog.csdn.net/jiang_ming_/article/details/82714444)

CRNN的实现大概有基于**tensorflow+kereas**的和基于**pytorch**的

其中基于torch的代码中涉及到**warp——CTC loss**的函数，是基于C++编写的，目前只支持Linux系统，我这里只测试了正向传播，
不知道比赛环境安装这个是不是方便，但是有人说这个版本的更稳定些。

## 2.1使用训练好的模型进行预测

**kereas_crnn_predict**文件夹中提供了使用crnn网络进行正向传播生成文本识别结果

使用方法 python start_predict.py -p *any_path*

其中如果*anypath*是一个图像，他就会对图像进行文本识别

如果是一个路径，会对该路径下所有的图像进行文本识别并输出csv格式的结果

**cfg.py**是配置文件

可以配置 

1.要载入的模型权重位置

2.输出结果位置

**keys.py**是字符集 **比赛时要重新做一份**

## 2.2训练模型

**keread_crnn_train**文件夹

### 2.2.1数据集制作
**create_dataset**文件夹提供了数据制作的方法
运行 create_dataset_for_game.py 就可以制作训练crnn模型需要的lmdb格式的数据集

其中需要配置

**create_val**是否制作验证集

**outputPath_train** 训练集输出路径

**outputPath_val**验证集输出路径

**val_per**验证集比例

**path**  读入图像路径

到时候路径之类的代码可能还要修改

### 2.2.2开始训练

**keras-train**文件夹提供了训练crnn的工具

运行 trainbatch.py即可开始训练 

**keras_train_cfg.py** 是配置文件 

**modelPath**    预训练模型位置

**trainroot**    训练集位置

**valroot**      验证集位置

**workers**      并行核心数

**epoch**        训练轮数

**keys.py**是字符集，也要改成根据比赛数据重新制作的

---

# 3.数据扩增


提供了数据扩增工具，来根据已有图像生成更多的图像。
包括平移，颜色变换（未完成，翻转，旋转可能不适合）

**data_augmention**提供了数据扩增工具

运行start_aug即可

其中 **aug_cfg.py**为数据扩增的参数配置

**img_path**读入路径
**out_path**输出路径

**move_off**将每一张图像分别像上下左右平移1到10个像素
**ran_num**随即颜色变化次数




---

# 4.字符集生成

由于比赛提供的文本真值是经过转码的，所以我们需要重新制作一份字母表，在我找到的CRNN代码中就是一个字母表，
把所有出现的字符按顺序罗列出来即可。

需要提取训练集中所有的文本真值的每一个“字”，并且不重复，即可。

---

# 5.大家补充
