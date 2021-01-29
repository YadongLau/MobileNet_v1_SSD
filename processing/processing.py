from xml.dom.minidom import Document
import json
import numpy as np
import cv2
import math
from PIL import Image
import xml.etree.ElementTree as ET
import sys
from tqdm import tqdm
import os
sys.setrecursionlimit(15000)

class CreateAnno:
    def __init__(self, ):
        self.doc = Document()  # 创建DOM文档对象
        self.anno = self.doc.createElement('annotation')  # 创建根元素
        self.doc.appendChild(self.anno)

        self.add_folder()
        self.add_path()
        self.add_source()
        self.add_segmented()

        # self.add_filename()
        # self.add_pic_size(width_text_str=str(width), height_text_str=str(height), depth_text_str=str(depth))

    def add_folder(self, floder_text_str='JPEGImages'):
        floder = self.doc.createElement('floder')  ##建立自己的开头
        floder_text = self.doc.createTextNode(floder_text_str)  ##建立自己的文本信息
        floder.appendChild(floder_text)  ##自己的内容
        self.anno.appendChild(floder)

    def add_filename(self, filename_text_str='00000.jpg'):
        filename = self.doc.createElement('filename')
        filename_text = self.doc.createTextNode(filename_text_str)
        filename.appendChild(filename_text)
        self.anno.appendChild(filename)

    def add_path(self, path_text_str="None"):
        path = self.doc.createElement('path')
        path_text = self.doc.createTextNode(path_text_str)
        path.appendChild(path_text)
        self.anno.appendChild(path)

    def add_source(self, database_text_str="Unknow"):
        source = self.doc.createElement('source')
        database = self.doc.createElement('database')
        database_text = self.doc.createTextNode(database_text_str)  # 元素内容写入
        database.appendChild(database_text)
        source.appendChild(database)
        self.anno.appendChild(source)

    def add_pic_size(self, width_text_str="0", height_text_str="0", depth_text_str="3"):
        size = self.doc.createElement('size')
        width = self.doc.createElement('width')
        width_text = self.doc.createTextNode(width_text_str)  # 元素内容写入
        width.appendChild(width_text)
        size.appendChild(width)

        height = self.doc.createElement('height')
        height_text = self.doc.createTextNode(height_text_str)
        height.appendChild(height_text)
        size.appendChild(height)

        depth = self.doc.createElement('depth')
        depth_text = self.doc.createTextNode(depth_text_str)
        depth.appendChild(depth_text)
        size.appendChild(depth)

        self.anno.appendChild(size)

    def add_segmented(self, segmented_text_str="0"):
        segmented = self.doc.createElement('segmented')
        segmented_text = self.doc.createTextNode(segmented_text_str)
        segmented.appendChild(segmented_text)
        self.anno.appendChild(segmented)

    def add_object(self,
                   name_text_str="None",
                   xmin_text_str="0",
                   ymin_text_str="0",
                   xmax_text_str="0",
                   ymax_text_str="0",
                   pose_text_str="Unspecified",
                   truncated_text_str="0",
                   difficult_text_str="0"):
        object = self.doc.createElement('object')
        name = self.doc.createElement('name')
        name_text = self.doc.createTextNode(name_text_str)
        name.appendChild(name_text)
        object.appendChild(name)

        pose = self.doc.createElement('pose')
        pose_text = self.doc.createTextNode(pose_text_str)
        pose.appendChild(pose_text)
        object.appendChild(pose)

        truncated = self.doc.createElement('truncated')
        truncated_text = self.doc.createTextNode(truncated_text_str)
        truncated.appendChild(truncated_text)
        object.appendChild(truncated)

        difficult = self.doc.createElement('difficult')
        difficult_text = self.doc.createTextNode(difficult_text_str)
        difficult.appendChild(difficult_text)
        object.appendChild(difficult)

        bndbox = self.doc.createElement('bndbox')
        xmin = self.doc.createElement('xmin')
        xmin_text = self.doc.createTextNode(xmin_text_str)
        xmin.appendChild(xmin_text)
        bndbox.appendChild(xmin)

        ymin = self.doc.createElement('ymin')
        ymin_text = self.doc.createTextNode(ymin_text_str)
        ymin.appendChild(ymin_text)
        bndbox.appendChild(ymin)

        xmax = self.doc.createElement('xmax')
        xmax_text = self.doc.createTextNode(xmax_text_str)
        xmax.appendChild(xmax_text)
        bndbox.appendChild(xmax)

        ymax = self.doc.createElement('ymax')
        ymax_text = self.doc.createTextNode(ymax_text_str)
        ymax.appendChild(ymax_text)
        bndbox.appendChild(ymax)
        object.appendChild(bndbox)

        self.anno.appendChild(object)

    def get_anno(self):
        return self.anno

    def get_doc(self):
        return self.doc

    def save_doc(self, save_path):
        with open(save_path, "w") as f:
            self.doc.writexml(f, indent='\t', newl='\n', addindent='\t', encoding='utf-8')



class ReadAnno:
    def __init__(self, json_path, process_mode="polygon"):
        self.json_data = json.load(open(json_path))
        self.filename = self.json_data['imagePath']
        # self.width = self.json_data['imageWidth']
        # self.height = self.json_data['imageHeight']
        self.img_path = json_path.replace('.json', '.bmp')
        im = Image.open(self.img_path)

        self.width = im.size[0]
        self.height = im.size[1]

        self.coordis = []
        assert process_mode in ["rectangle", "polygon"]
        if process_mode == "rectangle":
            self.process_polygon_shapes()
        elif process_mode == "polygon":
            self.process_polygon_shapes()

    def process_rectangle_shapes(self):
        for single_shape in self.json_data['shapes']:
            bbox_class = single_shape['label']
            xmin = single_shape['points'][0][0]
            ymin = single_shape['points'][0][1]
            xmax = single_shape['points'][1][0]
            ymax = single_shape['points'][1][1]
            self.coordis.append([xmin, ymin, xmax, ymax, bbox_class])

    def process_polygon_shapes(self):
        for single_shape in self.json_data['shapes']:
            bbox_class = single_shape['label']
            temp_points = []
            for couple_point in single_shape['points']:
                x = float(couple_point[0])
                y = float(couple_point[1])
                temp_points.append([x, y])
            temp_points = np.array(temp_points)
            xmin, ymin = temp_points.min(axis=0)
            xmax, ymax = temp_points.max(axis=0)
            self.coordis.append([xmin, ymin, xmax, ymax, bbox_class])

    def get_width_height(self):
        return self.width, self.height

    def get_filename(self):
        return self.filename

    def get_coordis(self):
        return self.coordis


def get_min_max(json_path):
    with open(json_path, 'r',encoding='utf8')as fp:
        json_data = json.load(fp)
        # print(json_data)
        x_y_list = []
        for i in range(len(json_data["shape"])):
            json_dict = json_data["shape"][i]
            x_min = min([json_dict['X1'], json_dict['X2'], json_dict['X3'], json_dict['X4']])
            y_min = min([json_dict['Y1'], json_dict['Y2'], json_dict['Y3'], json_dict['Y4']])
            x_max = max([json_dict['X1'], json_dict['X2'], json_dict['X3'], json_dict['X4']])
            y_max = max([json_dict['Y1'], json_dict['Y2'], json_dict['Y3'], json_dict['Y4']])
            LABEL = json_dict["cracktype"]
            x_y_list.append([x_min, y_min, x_max, y_max, LABEL])
    return x_y_list


def json_transform_xml(json_path, xml_path, process_mode="polygon"):
    json_path = json_path
    with open(json_path, 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
        num = sum(1 for line in json_data)
        # 此处将标准的json文件转为xml
        if num >= 4:
            json_anno = ReadAnno(json_path, process_mode=process_mode)
            width, height = json_anno.get_width_height()
            filename = json_anno.get_filename()
            coordis = json_anno.get_coordis()
            xml_anno = CreateAnno()
            xml_anno.add_filename(filename)
            xml_anno.add_pic_size(width_text_str=str(width), height_text_str=str(height), depth_text_str=str(3))
            for xmin, ymin, xmax, ymax, label in coordis:
                xml_anno.add_object(name_text_str=str(label),
                                    xmin_text_str=str(int(xmin)),
                                    ymin_text_str=str(int(ymin)),
                                    xmax_text_str=str(int(xmax)),
                                    ymax_text_str=str(int(ymax)))
            xml_anno.save_doc(xml_path)
        # 此处将单行的json文件转为xml
        else:
            x_y_label = get_min_max(json_path)
            bmp_path = json_path.replace('.json', '.bmp')
            bmp_name = bmp_path.split('\\')[-1]
            xml_anno = CreateAnno()
            xml_anno.add_filename(bmp_name)
            bmp_size = Image.open(bmp_path)
            width = bmp_size.size[0]
            height = bmp_size.size[1]
            xml_anno.add_pic_size(width_text_str=str(width), height_text_str=str(height), depth_text_str=str(3))
            for i in range(len(x_y_label)):
                xmin, ymin, xmax, ymax, label = x_y_label[i]
                # print(xmin, ymin, xmax, ymax, label)
            # for xmin, ymin, xmax, ymax, label in x_y_label:
                xml_anno.add_object(name_text_str=str(label),
                                    xmin_text_str=str(int(xmin)),
                                    ymin_text_str=str(int(ymin)),
                                    xmax_text_str=str(int(xmax)),
                                    ymax_text_str=str(int(ymax)))
            xml_anno.save_doc(xml_path)


class ImgAugemention():
    def __init__(self):
        self.angle = 90

    # rotate_img
    def rotate_image(self, src, angle, scale=1.):
        w = src.shape[1]
        h = src.shape[0]
        # convet angle into rad
        rangle = np.deg2rad(angle)  # angle in radians
        # calculate new image width and height
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # map
        return cv2.warpAffine(
            src, rot_mat, (int(math.ceil(nw)), int(math.ceil(nh))),
            flags=cv2.INTER_LANCZOS4)

    def rotate_xml(self, src, xmin, ymin, xmax, ymax, angle, scale=1.):
        w = src.shape[1]
        h = src.shape[0]
        rangle = np.deg2rad(angle)  # angle in radians
        # now calculate new image width and height
        # get width and heigh of changed image
        nw = (abs(np.sin(rangle)*h) + abs(np.cos(rangle)*w))*scale
        nh = (abs(np.cos(rangle)*h) + abs(np.sin(rangle)*w))*scale
        # ask OpenCV for the rotation matrix
        rot_mat = cv2.getRotationMatrix2D((nw*0.5, nh*0.5), angle, scale)
        # calculate the move from the old center to the new center combined
        # with the rotation
        rot_move = np.dot(rot_mat, np.array([(nw-w)*0.5, (nh-h)*0.5, 0]))
        # the move only affects the translation, so update the translation
        # part of the transform
        rot_mat[0, 2] += rot_move[0]
        rot_mat[1, 2] += rot_move[1]
        # rot_mat: the final rot matrix
        # get the four center of edges in the initial martix，and convert the coord
        point1 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymin, 1]))
        point2 = np.dot(rot_mat, np.array([xmax, (ymin+ymax)/2, 1]))
        point3 = np.dot(rot_mat, np.array([(xmin+xmax)/2, ymax, 1]))
        point4 = np.dot(rot_mat, np.array([xmin, (ymin+ymax)/2, 1]))
        # concat np.array
        concat = np.vstack((point1, point2, point3, point4))
        # change type
        concat = concat.astype(np.int32)
        # print(concat)
        rx, ry, rw, rh = cv2.boundingRect(concat)
        return rx, ry, rw, rh

    def process_img(self, imgs_path, xmls_path, img_save_path, xml_save_path, angle_list):
        # assign the rot angles
        for angle in angle_list:
            for img_name in tqdm(os.listdir(imgs_path)):
                # split filename and suffix
                n, s = os.path.splitext(img_name)
                # for the sake of use yol model, only process '.jpg'
                if s == ".bmp":
                    img_path = os.path.join(imgs_path, img_name)
                    img = cv2.imread(img_path)
                    rotated_img = self.rotate_image(img, angle)
                    # 写入图像
                    cv2.imwrite(img_save_path + n + "_" + str(angle) + "d.bmp", rotated_img)
                    # print("log: [%sd] %s is processed." % (angle, img))
                    xml_url = img_name.split('.')[0] + '.xml'
                    xml_path = os.path.join(xmls_path, xml_url)
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    for box in root.iter('bndbox'):
                        xmin = float(box.find('xmin').text)
                        ymin = float(box.find('ymin').text)
                        xmax = float(box.find('xmax').text)
                        ymax = float(box.find('ymax').text)
                        x, y, w, h = self.rotate_xml(img, xmin, ymin, xmax, ymax, angle)
                        # change the coord
                        box.find('xmin').text = str(x)
                        box.find('ymin').text = str(y)
                        box.find('xmax').text = str(x+w)
                        box.find('ymax').text = str(y+h)
                        box.set('updated', 'yes')
                    # write into new xml
                    tree.write(xml_save_path + n + "_" + str(angle) + "d.xml")
                # print("[%s] %s is processed." % (angle, img_name))


class FolderProcess:

    def arrange_img(self,file_path):
        imgDir = []
        jsonDir = []
        dir = os.listdir(file_path)
        for i in dir:
            if i.endswith('bmp'):
                imgDir.append(i)
            if i.endswith('.json'):
                jsonDir.append(i)
        for i in imgDir:
            new_i = i.replace('.bmp', '.json')
            if os.path.exists(file_path + '//' + new_i):
                pass
            else:
                os.remove(file_path + '//' + i)
        for j in jsonDir:
            new_j = j.replace('.json', '.bmp')
            if os.path.exists(file_path + '//' + new_j):
                pass
            else:
                os.remove(file_path + '//' + j)
        print('数据集整理完成！')

    def create_folder(self, path_dir):
        os.mkdir(path_dir + '/' + 'JPEGImages')
        os.mkdir(path_dir + '/' + 'Annotations')
        os.mkdir(path_dir + '/' + 'ImageSets')
        os.mkdir(path_dir + '/' + 'ImageSets' + '/' + 'Main')
        os.mkdir(path_dir + '/' + 'train_val_test')
        os.mkdir(path_dir + '/' + 'logs')
        os.mkdir(path_dir + '/' + 'Temp')
        os.mkdir(path_dir + '/' + 'Temp' + '/' + 'Anno')
        os.mkdir(path_dir + '/' + 'Temp' + '/' + 'JPEG')

