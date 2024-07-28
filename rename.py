#批量重命名 将文件名改为六位数的序号

import os
import sys
from tqdm import tqdm
directory_path = os.path.dirname(os.path.abspath(__file__))
# 定义一个名字叫做rename的函数
def rename(filePath):
    base = filePath
    files = os.listdir(filePath)
    files.sort()
    for index,file in enumerate(tqdm(files)):
        suffix = file.split('.')[-1]
        new_name = str(index+1)
        new_name = new_name.rjust(6,'0')
        new_file = new_name+'.'+suffix
        os.rename(base+'\\'+file, base + '\\' + new_file)
    

if __name__ == '__main__':
    filePath = directory_path + r'\VOCdevkit\VOC2007\JPEGImages'
    rename(filePath)

    filePath = directory_path + r'\VOCdevkit\VOC2007\Annotations'
    rename(filePath)