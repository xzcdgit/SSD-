import os
import random
 
trainval_percent = 0.9#验证集和训练集占的百分比
train_percent = 0.8#训练集占的百分比
xmlfilepath = r'C:\Code\Python\SSDTest\VOCdevkit\VOC2007\Annotations'#Annotation文件夹所在位置
txtsavepath = r'C:\Code\Python\SSDTest\VOCdevkit\VOC2007\ImageSets\Main'#ImageSets文件下的Main文件夹所在位置
total_xml = os.listdir(xmlfilepath)
 
num = len(total_xml)
list = range(num)
tv = int(num * trainval_percent)
tr = int(tv * train_percent)
trainval = random.sample(list, tv)
train = random.sample(trainval, tr)
#Main文件夹下所对应的四个txt文件夹路径
ftrainval = open(r'C:\Code\Python\SSDTest\VOCdevkit\VOC2007\ImageSets\Main\trainval.txt', 'w')
ftest = open(r'C:\Code\Python\SSDTest\VOCdevkit\VOC2007\ImageSets\Main\test.txt', 'w')
ftrain = open(r'C:\Code\Python\SSDTest\VOCdevkit\VOC2007\ImageSets\Main\train.txt', 'w')
fval = open(r'C:\Code\Python\SSDTest\VOCdevkit\VOC2007\ImageSets\Main\val.txt', 'w')
 
for i in list:
    name = total_xml[i][:-4] + '\n'
    if i in trainval:
        ftrainval.write(name)
        if i in train:
            ftrain.write(name)
        else:
            fval.write(name)
    else:
        ftest.write(name)
 
ftrainval.close()
ftrain.close()
fval.close()
ftest.close()