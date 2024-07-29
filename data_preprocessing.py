import os
import sys
import shutil
from xml.dom.minidom import Document
import cv2
import numpy as np
from tqdm import tqdm


def makexml(picPath, txtPath, xmlPath):  # txt所在文件夹路径，xml文件保存路径，图片所在文件夹路径
    """此函数用于将yolo格式txt标注文件转换为voc格式xml标注文件
    在自己的标注图片文件夹下建三个子文件夹，分别命名为picture、txt、xml
    """
    dic = {'0': "person",  # 创建字典用来对类型进行转换
           '1': "arclight",  # 此处的字典要与自己的classes.txt文件中的类对应，且顺序要一致
           }
    files = os.listdir(txtPath)
    for i, name in enumerate(tqdm(files)):
        xmlBuilder = Document()
        annotation = xmlBuilder.createElement("annotation")  # 创建annotation标签
        xmlBuilder.appendChild(annotation)
        txtFile = open(txtPath +'\\'+ name)
        txtList = txtFile.readlines()
        for root,dirs,filename in os.walk(picPath):
            img = cv2.imread(root+ '\\'+filename[i])
            Pheight, Pwidth, Pdepth = img.shape

        folder = xmlBuilder.createElement("folder")  # folder标签
        foldercontent = xmlBuilder.createTextNode("driving_annotation_dataset")
        folder.appendChild(foldercontent)
        annotation.appendChild(folder)  # folder标签结束

        filename = xmlBuilder.createElement("filename")  # filename标签
        filenamecontent = xmlBuilder.createTextNode(name[0:-4] + ".jpg")
        filename.appendChild(filenamecontent)
        annotation.appendChild(filename)  # filename标签结束

        size = xmlBuilder.createElement("size")  # size标签
        width = xmlBuilder.createElement("width")  # size子标签width
        widthcontent = xmlBuilder.createTextNode(str(Pwidth))
        width.appendChild(widthcontent)
        size.appendChild(width)  # size子标签width结束

        height = xmlBuilder.createElement("height")  # size子标签height
        heightcontent = xmlBuilder.createTextNode(str(Pheight))
        height.appendChild(heightcontent)
        size.appendChild(height)  # size子标签height结束

        depth = xmlBuilder.createElement("depth")  # size子标签depth
        depthcontent = xmlBuilder.createTextNode(str(Pdepth))
        depth.appendChild(depthcontent)
        size.appendChild(depth)  # size子标签depth结束

        annotation.appendChild(size)  # size标签结束

        for j in txtList:
            oneline = j.strip().split(" ")
            object = xmlBuilder.createElement("object")  # object 标签
            picname = xmlBuilder.createElement("name")  # name标签
            namecontent = xmlBuilder.createTextNode(dic[oneline[0]])
            picname.appendChild(namecontent)
            object.appendChild(picname)  # name标签结束

            pose = xmlBuilder.createElement("pose")  # pose标签
            posecontent = xmlBuilder.createTextNode("Unspecified")
            pose.appendChild(posecontent)
            object.appendChild(pose)  # pose标签结束

            truncated = xmlBuilder.createElement("truncated")  # truncated标签
            truncatedContent = xmlBuilder.createTextNode("0")
            truncated.appendChild(truncatedContent)
            object.appendChild(truncated)  # truncated标签结束

            difficult = xmlBuilder.createElement("difficult")  # difficult标签
            difficultcontent = xmlBuilder.createTextNode("0")
            difficult.appendChild(difficultcontent)
            object.appendChild(difficult)  # difficult标签结束

            bndbox = xmlBuilder.createElement("bndbox")  # bndbox标签
            xmin = xmlBuilder.createElement("xmin")  # xmin标签
            mathData = int(((float(oneline[1])) * Pwidth + 1) - (float(oneline[3])) * 0.5 * Pwidth)
            xminContent = xmlBuilder.createTextNode(str(mathData))
            xmin.appendChild(xminContent)
            bndbox.appendChild(xmin)  # xmin标签结束

            ymin = xmlBuilder.createElement("ymin")  # ymin标签
            mathData = int(((float(oneline[2])) * Pheight + 1) - (float(oneline[4])) * 0.5 * Pheight)
            yminContent = xmlBuilder.createTextNode(str(mathData))
            ymin.appendChild(yminContent)
            bndbox.appendChild(ymin)  # ymin标签结束

            xmax = xmlBuilder.createElement("xmax")  # xmax标签
            mathData = int(((float(oneline[1])) * Pwidth + 1) + (float(oneline[3])) * 0.5 * Pwidth)
            xmaxContent = xmlBuilder.createTextNode(str(mathData))
            xmax.appendChild(xmaxContent)
            bndbox.appendChild(xmax)  # xmax标签结束

            ymax = xmlBuilder.createElement("ymax")  # ymax标签
            mathData = int(((float(oneline[2])) * Pheight + 1) + (float(oneline[4])) * 0.5 * Pheight)
            ymaxContent = xmlBuilder.createTextNode(str(mathData))
            ymax.appendChild(ymaxContent)
            bndbox.appendChild(ymax)  # ymax标签结束

            object.appendChild(bndbox)  # bndbox标签结束

            annotation.appendChild(object)  # object标签结束

        f = open(xmlPath +'\\'+ name[0:-4] + ".xml", 'w')
        xmlBuilder.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')
        f.close()


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


def folder_clean(dir:str):
    if os.path.isdir(dir):
        shutil.rmtree(dir)
        os.mkdir(dir)
        print('清空成功')
        return True
    else:
        print('清空异常')
        return False
    


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
   
    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
 
    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding
 
    dw /= 2  # divide padding into 2 sides
    dh /= 2
    print(dw, dh)
    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, ratio, (dw, dh)

def point_convert(x, ratio):
    y = (1-ratio)*0.5+x*ratio
    return y
def size_convert(x, ratio):
    y = x*ratio
    return y

def resize_and_pad(image, target_width, target_height):
    h, w = image.shape[:2]
    scale = min(target_width / w, target_height / h)
    new_w = int(w * scale)
    new_h = int(h * scale)
    
    resized_image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    
    top = (target_height - new_h) // 2
    bottom = target_height - new_h - top
    left = (target_width - new_w) // 2
    right = target_width - new_w - left
    
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[114, 114, 114])
    
    return padded_image




def main():
    resize_shape = 640
    yolo_all_path = r'C:\Users\24225\Desktop\data\all' # yolo图像和标签路径
    
    yolo_labels_path = r'D:\Code\Python\SSDTest\VOCdevkit\VOC2007\YoloLabels'
    voc_imgs_path = r'VOCdevkit\VOC2007\JPEGImages' #voc图像地址
    voc_labels_path = r'VOCdevkit\VOC2007\Annotations' # voc标签路径
    tmp1_path = r'C:\Users\24225\Desktop\data\tmp1'
    tmp2_path = r'C:\Users\24225\Desktop\data\tmp2'
    folder_clean(voc_imgs_path) #清空voc图像文件夹
    folder_clean(voc_labels_path) #清空voc标签文件夹
    folder_clean(yolo_labels_path)
    
    files = os.listdir(yolo_all_path)
    files.sort() #排序
    count = 0
    for index, file in enumerate(tqdm(files)):
        suffix = file.split('.')[-1]
        name = file[:-len(suffix)-1]
        if suffix == 'jpg':
            if name+'.txt' in files:
                with open(yolo_all_path+'\\'+name+'.txt', 'r') as f:
                    boxes = f.read()
                    boxes = boxes.split('\n')
                    boxes_list = []
                    for box in boxes:
                        if len(box)>0:
                            boxes_list.append(list(map(float,box.split(' '))))
                img = cv2.imread(yolo_all_path+'\\'+file)
                img_height, img_width = img.shape[:2]  # current shape [height, width]
                if img_width > img_height:
                    ratio = img_height/img_width
                    for i, (type, x, y, w, h) in enumerate(boxes_list):
                        new_y = point_convert(y,ratio)
                        new_h = size_convert(h,ratio)
                        boxes_list[i][2] = new_y
                        boxes_list[i][4] = new_h
                elif img_width < img_height:
                    ratio = img_width / img_height
                    for i, (type, x, y, w, h) in enumerate(boxes_list):
                        new_x = point_convert(x,ratio)
                        new_w = size_convert(w,ratio)
                        boxes_list[i][1] = new_x
                        boxes_list[i][3] = new_w
                new_name = str(count+1)
                new_name = new_name.rjust(6,'0')
                padded_image = resize_and_pad(img, resize_shape, resize_shape)
                cv2.imwrite(voc_imgs_path+'\\'+new_name+'.jpg', padded_image)
                out_str = ''
                with open(yolo_labels_path+'\\'+new_name+'.txt', 'w') as f:
                    for line in boxes_list:
                        for i, ele in enumerate(line):
                            if i==0:
                                out_str += str(int(ele)) + ' '
                            else:
                                out_str += str(ele) + ' '
                        out_str = out_str[:-1]
                        out_str += '\n'
                    f.write(out_str)
                count += 1
        else:
            continue
    #yolo格式标签转voc格式
    makexml(voc_imgs_path, yolo_labels_path, voc_labels_path)
    #图像和标签信息汇总
    import voc_annotation
    
if __name__ == '__main__':
    main()
    #res = point_convert(0.5, 2/3)
    #print(res)
    
    
    
    