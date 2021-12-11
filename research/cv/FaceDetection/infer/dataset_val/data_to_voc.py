# Copyright 2021 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""data_to_voc"""
import os
import time
import xml.dom
import xml.dom.minidom
import multiprocessing
from shutil import copy
from PIL import Image

_DIFFICULT = '0'
_TRUNCATED = '0'
_POSE = 'Unspecified'
_ROOT_NODE = 'annotation'
_FOLDER_NODE = 'WIDERFACE_VOC'  # val 1: The folder name of the data under VOCdevkit
_NAME = 'face'  # val 2: Object category name (only one category is supported in this procedure)


# create a node
def createElementNode(doc, tag, attr):
    """
        createElementNode
    """
    element_node = doc.createElement(tag)
    # create a text node
    text_node = doc.createTextNode(attr)
    # treats the text node as a child of the element node
    element_node.appendChild(text_node)
    return element_node


# create childnode
def createChildNode(doc, tag, attr, parent_node):
    """
        createChildNode
    """
    child_node = createElementNode(doc, tag, attr)
    parent_node.appendChild(child_node)


# create object and its chilenodes
def createObjectNode(doc, obj):
    """
        createObjectNode
    """
    obj_info = obj.split(' ')
    bbox = [0, 0, 0, 0]
    bbox[0] = int(obj_info[0])
    bbox[1] = int(obj_info[1])
    bbox[2] = int(obj_info[2])
    bbox[3] = int(obj_info[3])
    bbox = processBbox(bbox)

    object_node = doc.createElement('object')
    createChildNode(doc, 'name', _NAME, object_node)
    createChildNode(doc, 'pose', _POSE, object_node)
    createChildNode(doc, 'truncated', _TRUNCATED, object_node)
    createChildNode(doc, 'difficult', _DIFFICULT, object_node)

    bndbox_node = doc.createElement('bndbox')
    createChildNode(doc, 'xmin', str(bbox[0]), bndbox_node)
    createChildNode(doc, 'ymin', str(bbox[1]), bndbox_node)
    createChildNode(doc, 'xmax', str(bbox[2]), bndbox_node)
    createChildNode(doc, 'ymax', str(bbox[3]), bndbox_node)
    object_node.appendChild(bndbox_node)
    return object_node


def writeXMLFile(doc, filename):
    """
        writeXMLFile
    """
    fout = open(filename, 'w')
    doc.writexml(fout, addindent='' * 4, newl='\n', encoding='utf-8')
    fout.close()


# x,y,w,h -> x1,y1,x2,y2
def processBbox(bbox):
    """
        processBbox
    """
    bbox[2] = bbox[0] + bbox[2]
    bbox[3] = bbox[1] + bbox[3]

    for index, val in enumerate(bbox):
        bbox[index] = round(val)
    return bbox


# class to sore data for each map tag
class ImgAnnotations:
    def __init__(self, s, face, imgPath):
        """
            __init__
        """
        self.faces = face  # the number of faces in the image
        # store object, that is, store all faces and their corresponding labels: [x1, y1, w, h, blur, expression, illumination, invalid, occlusion, pose]
        self.objects = []
        self.size = s  # size of image
        self.imgPath = imgPath


# process each image from coco to VOC, totalImages is a dictionary
def processSingleData(total_images, output_JPEGImages, output_Annotations):
    """
        processSingleData
    """
    for file_name in total_images:
        curImgName = total_images[file_name].imgPath  # original image path
        curOutputImgName = os.path.join(output_JPEGImages, file_name)  # image target path
        copy(curImgName, curOutputImgName)  # copy and paste

        curSize = total_images[file_name].size  # current image size w h
        # print(curSize)

        curObjects = total_images[file_name].objects  # all faces in the current image

        curXml = os.path.join(output_Annotations, file_name.split('.')[0]) + '.xml'  # output XML file
        if os.path.exists(curXml):
            os.remove(curXml)
        # print(curXml)

        my_dom = xml.dom.getDOMImplementation()
        doc = my_dom.createDocument(None, _ROOT_NODE, None)

        # root node
        root_node = doc.documentElement

        # folder node
        createChildNode(doc, 'folder', _FOLDER_NODE, root_node)

        # filename node
        createChildNode(doc, 'filename', file_name, root_node)

        # size node
        size_node = doc.createElement('size')
        createChildNode(doc, 'width', str(curSize[0]), size_node)
        createChildNode(doc, 'height', str(curSize[1]), size_node)
        createChildNode(doc, 'depth', str(3), size_node)
        root_node.appendChild(size_node)

        # object node
        for obj in curObjects:
            object_node = createObjectNode(doc, obj)
            root_node.appendChild(object_node)

        # written to the file
        writeXMLFile(doc, curXml)


if __name__ == '__main__':

    startTime = time.time()

    # imgPath = '/ssd/ssd1/wd/COCO/train2014'  # val 3: source image path
    rootPath = './WIDER_val'

    outputPath = './dataset_val'        # val 4: the output path
    outputAnnotations = os.path.join(outputPath, 'Annotations')     # val 5: the path of mark files, generally need not change
    outputJPEGImages = os.path.join(outputPath, 'JPEGImages')       # val 6: picture output path, generally need not change
    outputImgSet = os.path.join(outputPath, 'ImageSets', 'Main')    # val 7: TXT output path, generally need not change
    trainOrValOrTest = 'val'  # val 8: what the data is used for: train, test, val
    outputTxtFile = os.path.join(outputImgSet, trainOrValOrTest + '.txt')
    if os.path.exists(outputTxtFile):
        os.remove(outputTxtFile)

    multiNum = 48  # val 9: number of multiple processes

    txtFile = './bbx_gt_txt/hard.txt'

    if not os.path.exists(outputPath):
        os.makedirs(outputPath)
    if not os.path.exists(outputAnnotations):
        os.makedirs(outputAnnotations)
    if not os.path.exists(outputJPEGImages):
        os.makedirs(outputJPEGImages)
    if not os.path.exists(outputImgSet):
        os.makedirs(outputImgSet)

    totalImages = {}  # { key: image name, value: label }
    faceNum = 0

    # read WiderFace's TXT file and put the contents into the totalImages dictionary
    with open(txtFile, 'r') as file:
        lines = file.readlines()
        line_length = len(lines)
        i = 0
        while i < line_length:
            if '.jpg' in lines[i]:  # image
                imgName = os.path.basename(lines[i]).replace('\n', '')  # xx.jpg
                imagePath = os.path.join(rootPath, 'images', lines[i]).replace('\n', '')  # xx/xx/xx/xx.jpg
                faces = int(lines[i+1].replace('\n', ''))
                faceNum += faces
                i += 2

                img = Image.open(imagePath)
                size = img.size

                if len(img.getbands()) != 3:  # remove black and white images
                    print(11111111111111, 'get a 8bit image')
                    break

                imgAnnotation = ImgAnnotations(size, faces, imagePath)

                for j in range(faces):
                    curline = i + j
                    info = lines[curline].replace('\n', '').split(' ')
                    w = int(info[2])
                    h = int(info[3])
                    radio = max(float(size[0]) / 1024., float(size[1]) / 576.)
                    new_w = float(w) / radio
                    new_h = float(h) / radio
                    if min(new_w, new_h) < 26.67:
                        # print(new_w, new_h)
                        continue
                    imgAnnotation.objects.append(lines[curline].replace('\n', ''))

                i += faces
                totalImages[imgName] = imgAnnotation

                # there are no faces in some of the pictures
                if faces == 0:
                    i += 1

                continue
            else:
                print(i)
                i += 1

        print('FaceNum:', faceNum)
        print('ImgNum:', len(totalImages))

        # generate ImageSets/Main/train.txt from totalImages
        for fileName in totalImages:
            curTxtName = fileName.split('/')[-1].split('.')[0]  # image names in ImageSets/Main/train.txt
            # print(curTxtName)
            with open(outputTxtFile, 'a') as f:  # write to ImageSets/Main/train.txt
                f.write(curTxtName + '\n')
                f.close()

        # divide totalImages evenly into each process
        totalNum = len(totalImages)
        print('Total imgs: %d' % totalNum)
        partNum = 0
        if totalNum % multiNum != 0:
            partNum = totalNum // multiNum + 1
        else:
            partNum = totalNum // multiNum

        print('There %d imgs to process in each process, except last one.' % partNum)

        targetNum = 1
        count = 1
        totalGroup = []
        curGroup = {}

        for fileName in totalImages:
            curGroup[fileName] = totalImages[fileName]
            if count == partNum and targetNum != multiNum:
                totalGroup.append(curGroup)
                curGroup = {}
                targetNum += 1
                count = 1
                continue
            count += 1

        if curGroup:  # last of all
            totalGroup.append(curGroup)
        print('There are %d processes.' % len(totalGroup))

        processList = []

        for i, imgs in enumerate(totalGroup):
            curP = multiprocessing.Process(target=processSingleData, args=(imgs, outputJPEGImages, outputAnnotations))
            curP.start()
            processList.append(curP)

        for curP in processList:
            curP.join()

    endTime = time.time()
    print('cost time:', endTime - startTime)
