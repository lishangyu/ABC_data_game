# ABC_data_game

venv文件夹中是相关的python包，如果不在本地跑的话就不要下载了，很大

OCR分为两个步骤
1.文本检测（输出是文本的包围框）
2.文本识别

由于这次数据集的图像基本上已经是剪裁好的图像（但有些还是有较大的倾斜/旋转），所以是否需要进行文本检测有待实验
--------------------------------------------------------------------------------------------------------
#1.文本检测
text_detection文件夹中提供了文本检测工具，它可以获取图像中文本的包围框坐标，并根据透视变换，
将文本框位置的图像扣出并转换到一个完全水平的矩形形状。
crop_img-----------存储了最终扣出后的图像
detect_result_img--存储了文本检测的可视化图像（没用，看看结果好不好）
detect_result_txt--存储了每一个图像的对应的文本框的顶点坐标

使用方法 python -p 'anypath'
其中如果anypath是一个图像，他就会对图像进行文本检测
如果是一个路径，会对该路径下所有的图像进行文本检测
cfg.py是配置文件

原理是advanced—EAST 
--------------------------------------------------------------------------------------------------------
#2.文本识别
（原理是CRNN CNN提特征+双向双层lstm（对lstm的每一个输入都是一个（字符集长度+1）的分类问题）+转录）

CRNN的实现大概有基于tensorflow+kereas的和基于pytorch的
其中基于torch的代码中涉及到warp——CTC loss的函数，是基于C++编写的，目前只支持Linux系统，我这里只测试了正向传播，
不知道比赛环境安装这个是不是方便，但是有人说这个版本的更稳定些。

##2.1使用训练好的模型进行预测
##2.2训练模型
--------------------------------------------------------------------------------------------------------
#3.数据扩增
https://blog.csdn.net/qq_36219202/article/details/78339459
提供了数据扩增工具，来根据已有图像生成更多的图像。
包括平移，翻转，旋转，调整对比度，高斯噪声，颜色变换（未完成，翻转，旋转可能不适合）
--------------------------------------------------------------------------------------------------------
#4.字符集生成
由于比赛提供的文本真值是经过转码的，所以我们需要重新制作一份字母表，在我找到的CRNN代码中就是一个字母表，
把所有出现的字符按顺序罗列出来即可。

需要提取训练集中所有的文本真值的每一个“字”，并且不重复，即可。
--------------------------------------------------------------------------------------------------------
#5.大家补充
