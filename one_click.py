from multiprocessing import freeze_support
freeze_support()
import warnings
warnings.filterwarnings("ignore")
import shutil
import cv2
import random
import json
import xml.etree.ElementTree as ET
import sys
from processing.processing import json_transform_xml, ImgAugemention, FolderProcess
from nets.ssd import get_ssd
from nets.ssd_training import Generator, MultiBoxLoss
from torch.utils.data import DataLoader
from utils.dataloader import ssd_dataset_collate, SSDDataset
from utils.config import Config
from torch.autograd import Variable
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import time
import torch
import numpy as np
import torch.optim as optim
import os
from datetime import datetime

os.environ["MKL_NUM_THREADS"] = '4'
os.environ["NUMEXPR_NUM_THREADS"] = '4'
os.environ["OMP_NUM_THREADS"] = '4'
dt = datetime.now()
sys.setrecursionlimit(15000)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def convert_annotation(image_id, list_file):
    in_file = open(path_dir + '/' + 'Annotations' + '/%s.xml'%(image_id))
    tree = ET.parse(in_file)
    root = tree.getroot()
    for obj in root.iter('object'):
        xmlbox = obj.find('bndbox')
        b = (int(xmlbox.find('xmin').text), int(xmlbox.find('ymin').text), int(xmlbox.find('xmax').text), int(xmlbox.find('ymax').text))
        list_file.write(" " + ",".join([str(a) for a in b]) + ',' + str('1'))

if __name__ == "__main__":
    print('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=')
    print('                  西瞳智能AI训练软件                    ')
    print('=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=*=')
    print(" ")
    print(" ")
    path_dir=''
    with open('cfg.json', encoding='utf-8') as F:
        json_file = json.load(F)
        panduan = json_file["是否进行数据处理部分（是：1  否：0）"]
        root_bmp_path = json_file["json和bmp文件路径"]
        path_dir = json_file["训练文件的路径"]
        train_fir = json_file["训练文件的路径"]
        trainval_percent = json_file["训练样本占的比例"]
        aug_1 = json_file["是否进行数据扩充，降低细节层次？检测小瑕疵或缺陷特征不明显时不建议使用。（是：1 / 否：0）"]
        FREEZE_LEARNING_RATE = json_file["冻结层的学习率（0.0001-0.0005之间）"]
        LEARNING_RATE = json_file["全部训练学习率（0.0001-0.0005之间）"]
        FREEZE_EPOCH = json_file["冻结多少世代"]
        TOTAL_EPOCH = json_file["共学习多少世代"]
        BATCH_SIZE = json_file["每批次放入多少张图像"]
        DATALOADER = json_file["是否使用网络内置的数据扩充进行训练（是：1  否：0）"]
        NUM_CLASSES = json_file["num_classes"]
        FEATURE_MAPS = json_file["训练图像设置为多大(300或512)"]
        PERTRAINED_PTH = json_file["预训练权重路径"]
        F.close()
    if int(panduan) == 1:
        name_list_1 = []
        print('=========================')
        print('第一部分：数据处理并生成文件夹')
        print('=========================')
        print(' ')


        # 删除未标注的bmp:
        print("整理文件夹图像数据：")
        a1 = FolderProcess()
        a1.arrange_img(root_bmp_path)
        a1.create_folder(path_dir)
        print("整理文件夹图像数据：")
        save_bmp_path = path_dir + '/' + 'JPEGImages'
        save_xml_path = path_dir + '/' + 'Annotations'
        save_main_path = path_dir + '/' + 'ImageSets' + '/' + 'Main' + '/'
        save_split_path = path_dir + '/' + 'train_val_test'
        name_list_2 = []
        root_json_dir = root_bmp_path
        for file in os.listdir(root_json_dir):
            if file.endswith('.json'):
                name_list_2.append(file)
        root_save_xml_dir = save_xml_path
        for i in os.listdir(root_bmp_path):
            if i.endswith(".bmp"):
                name_list_1.append(i)
        for j in tqdm(name_list_1):
            shutil.copy(root_bmp_path + '\\' + j, save_bmp_path)
        for json_filename in tqdm(name_list_2):
            time.sleep(0.1)
            json_path = os.path.join(root_json_dir, json_filename)
            save_xml_path = os.path.join(root_save_xml_dir, json_filename.replace(".json", ".xml"))
            json_transform_xml(json_path, save_xml_path, process_mode="polygon")

        # ==================================================
        #                生成main中的txt文档，对原图进行分类
        # ==================================================

        xmlfilepath = path_dir + '/' + 'Annotations' + '/'

        train_percent = 0.9
        temp_xml = os.listdir(xmlfilepath)
        total_xml = []
        for xml in temp_xml:
            if xml.endswith(".xml"):
                total_xml.append(xml)

        num = len(total_xml)
        list = range(num)
        tv = int(num * float(trainval_percent))
        tr = int(tv * train_percent)
        trainval = random.sample(list, tv)
        train = random.sample(trainval, tr)

        print("训练和测试图像数量：", tv)
        print("测试图像数量：", tr)
        ftrainval = open(os.path.join(save_main_path, 'trainval.txt'), 'w')
        ftest = open(os.path.join(save_main_path, 'test.txt'), 'w')
        ftrain = open(os.path.join(save_main_path, 'train.txt'), 'w')
        fval = open(os.path.join(save_main_path, 'val.txt'), 'w')

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

        # ================================================
        #                根据名字进行数据增强
        # ================================================
        print('======================')
        print('      开始数据扩充      ')
        print('======================')
        print(' ')
        train_txt_path = path_dir + '/' + 'ImageSets' + '/' + 'Main' + '/'+'train.txt'

        img_name_list = []
        read_train_txt = open(train_txt_path, 'a')
        with open(train_txt_path, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                img_name = line+'.bmp'
                img_name_list.append(img_name)  # 获取train列表下的图像名

        if aug_1 == "1":
            print("开始进行数据扩充（高斯模糊）:")
            for i in tqdm(img_name_list):
                img = cv2.imread(path_dir+'/'+'JPEGImages'+'/'+i)
                img_aug = cv2.GaussianBlur(img, (9, 9), 1.5)
                cv2.imwrite(path_dir + '/'+'JPEGImages'+'/'+'Gauss_blur_'+i, img_aug)
                xml_file_name = i.replace('.bmp', '.xml')
                shutil.copyfile(path_dir+'/'+'Annotations'+'/'+xml_file_name, path_dir+'/'+'Annotations'+'/' + 'Gauss_blur_' + xml_file_name)

                # 写入增强的图像名称到train.txt:
                new_i = i.replace('.bmp', '')
                read_train_txt.write('Gauss_blur_'+new_i+'\n')
            print('高斯模糊完成')

        # -------------
        #     旋转
        # -------------
        print('开始使用数据扩充(旋转):')
        with open(path_dir + '/' + 'ImageSets' + '/' + 'Main' + '/'+'train.txt', 'r') as f:
            for na in f.readlines():
                line = na.strip('\n')
                shutil.copyfile(path_dir+'/'+'JPEGImages'+'/' + line + '.bmp', path_dir + '/' + 'Temp' + '/' + 'JPEG'+'/' + line + '.bmp' )
                shutil.copyfile(path_dir+'/'+'Annotations'+'/' + line + '.xml', path_dir + '/' + 'Temp' + '/' + 'Anno'+'/' + line + '.xml' )
            f.close()
        imgs_aug = ImgAugemention()
        imgs_path = path_dir+'/'+'Temp'+'/'+'JPEG'+'/'
        xmls_path = path_dir+'/'+'Temp'+'/'+'Anno'+'/'
        img_save_path = path_dir+'/'+'JPEGImages'+'/'
        xml_save_path = path_dir+'/'+'Annotations'+'/'
        angle_list = [90, 210, 270]
        imgs_aug.process_img(imgs_path, xmls_path, img_save_path, xml_save_path, angle_list)
        # 写入增强的图像名称到train.txt
        for nl in os.listdir(path_dir+'/'+'JPEGImages'+'/'):
            if nl.endswith('d.bmp'):
                nl_2 = nl.replace('.bmp', '')
                read_train_txt.write(nl_2+'\n')
        read_train_txt.close()

        # ---------------------------------
        #     生成带标注信息的训练和测试txt
        # ---------------------------------
        sets = ['train', 'val', 'test']

        # classes = ["1"]
        for image_set in sets:
            image_ids = open(path_dir + '/' + 'ImageSets' + '/' + 'Main' + '/%s.txt' %(image_set)).read().strip().split()
            list_file = open(path_dir + '/' + 'train_val_test' + '/' + '%s.txt' %(image_set), 'w')
            for image_id in image_ids:
                list_file.write(path_dir + '/' + 'JPEGImages' + '/%s.bmp' %(image_id))
                convert_annotation(image_id, list_file)
                list_file.write('\n')
            list_file.close()
        shutil.rmtree(path_dir + '/' + 'Temp')
        print('数据扩充完成')

    # ------------------------------------#
    #                train
    # ------------------------------------#

    # ------------------------------------#
    #   先冻结一部分权重训练
    #   后解冻全部权重训练
    #   先大学习率
    #   后小学习率
    # ------------------------------------#
    print("=========================")
    print("        程序训练部分       ")
    print("=========================")

    root_dir = path_dir
    if os.path.exists(root_dir + '/logs'):
        pass
    else:
        os.mkdir(root_dir + '/logs')

    freeze_lr = FREEZE_LEARNING_RATE
    lr = LEARNING_RATE
    Cuda = True

    Start_iter = 0
    Freeze_epoch = FREEZE_EPOCH
    Epoch = TOTAL_EPOCH
    Batch_size = BATCH_SIZE
    data_loader = DATALOADER
    if int(data_loader) == 1:
        Use_Data_Loader = True
    else:
        Use_Data_Loader = False

    num_classes = len(NUM_CLASSES)
    feature_map_size = FEATURE_MAPS


    model = get_ssd("train", num_classes, feature_map_size)

    print('Loading weights into state dict...')
    # 指定gpu进行训练
    os.environ["CUDA_VISIBLE_DEVICES"] = '0'
    device = torch.device('cuda:0')
    model_dict = model.state_dict()
    pretrained_pth = PERTRAINED_PTH
    # pretrained_dict = torch.load(pretrained_pth, map_location=device)
    # pretrained_dict = {k: v for k, v in pretrained_dict.items() if np.shape(model_dict[k]) == np.shape(v)}
    # model_dict.update(pretrained_dict)
    # model.load_state_dict(model_dict)
    print('Finished!')
    net = model.train()
    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()
    print('if cuda...')
    annotation_path = root_dir + '/train_val_test/train.txt'

    with open(annotation_path) as f:
        lines = f.readlines()
    np.random.seed(10101)
    np.random.shuffle(lines)
    np.random.seed(None)
    num_train = len(lines)

    print('open annotation xml...')
    if Use_Data_Loader:
        print('use dataloader...')
        train_dataset = SSDDataset(lines[:num_train], (FEATURE_MAPS, FEATURE_MAPS))
        gen = DataLoader(train_dataset, shuffle=True, batch_size=Batch_size, num_workers=0, pin_memory=True,
                         drop_last=True, collate_fn=ssd_dataset_collate)
    else:
        gen = Generator(Batch_size, lines,
                        (Config["min_dim"], Config["min_dim"]), num_classes).generate()

    criterion = MultiBoxLoss(num_classes, 0.5, True, 0, True, 3, 0.5, False, Cuda)
    # criterion = MultiBoxLoss(Config['num_classes'], 0.5, True, 0, True, 3, 0.5, False, Cuda)
    epoch_size = num_train // Batch_size
    pth_save_path = root_dir + '/logs'


    if True:
        # ------------------------------------#
        #   冻结一定部分训练
        # ------------------------------------#
        for param in model.vgg.parameters():
            param.requires_grad = False
        for param in model.extras.parameters():
            param.requires_grad = False
        print('进入冻结层...')
        optimizer = optim.Adam(net.parameters(), lr=float(freeze_lr))
        # optimizer = optim.SGD(net.parameters(), lr=lr,momentum=0.95,weight_decay=0.0005)
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        for epoch in range(Start_iter, Freeze_epoch):
            # print('进入冻结层的训练层...')
            with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict,
                      mininterval=0.3) as pbar:
                # print('显示进度条...')
                loc_loss = 0
                conf_loss = 0
                for iteration, batch in enumerate(gen):
                    if iteration >= epoch_size:
                        break
                    images, targets = batch[0], batch[1]
                    with torch.no_grad():
                        if Cuda:
                            images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                            targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in
                                       targets]
                        else:
                            images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                            targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                    # 前向传播
                    out = net(images)
                    # 清零梯度
                    optimizer.zero_grad()
                    # 计算loss
                    loss_l, loss_c = criterion(out, targets)
                    loss = loss_l + loss_c
                    # 反向传播
                    loss.backward()
                    optimizer.step()

                    loc_loss += loss_l.item()
                    conf_loss += loss_c.item()

                    pbar.set_postfix(**{'loc_loss': loc_loss / (iteration + 1),
                                        'conf_loss': conf_loss / (iteration + 1),
                                        'lr': get_lr(optimizer)})
                    pbar.update(1)

            lr_scheduler.step()
            print('Saving state, iter:', str(epoch + 1))
            torch.save(model.state_dict(), pth_save_path + '/' + '%d-%d-%d-Epoch%d-Loc%.4f-Conf%.4f.pth' % (
            dt.day, dt.hour, dt.minute, epoch, loc_loss / (iteration + 1), conf_loss / (iteration + 1)))
            print('pth has saved...')

    if True:
        # ------------------------------------#
        #   全部解冻训练
        # ------------------------------------#
        for param in model.vgg.parameters():
            param.requires_grad = True
        for param in model.extras.parameters():
            param.requires_grad = False

        # optimizer = optim.SGD(net.parameters(), lr=lr, momentum=0.95, weight_decay=0.0005)
        # # lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.95)
        # lr_scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=70)

        optimizer = optim.Adam(net.parameters(), lr=float(lr))
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.95)
        for epoch in range(Freeze_epoch, Epoch):
            with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict,
                      mininterval=0.3) as pbar:
                loc_loss = 0
                conf_loss = 0
                for iteration, batch in enumerate(gen):
                    if iteration >= epoch_size:
                        break
                    images, targets = batch[0], batch[1]
                    with torch.no_grad():
                        if Cuda:
                            images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                            targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)).cuda() for ann in
                                       targets]
                        else:
                            images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                            targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                    # 前向传播
                    out = net(images)
                    # 清零梯度
                    optimizer.zero_grad()
                    # 计算loss
                    loss_l, loss_c = criterion(out, targets)
                    loss = loss_l + loss_c
                    # 反向传播
                    loss.backward()
                    optimizer.step()
                    # 加上
                    loc_loss += loss_l.item()
                    conf_loss += loss_c.item()

                    pbar.set_postfix(**{'loc_loss': loc_loss / (iteration + 1),
                                        'conf_loss': conf_loss / (iteration + 1),
                                        'lr': get_lr(optimizer)})
                    pbar.update(1)

            lr_scheduler.step()
            print('Saving state, iter:', str(epoch + 1))

            torch.save(model.state_dict(),
                       pth_save_path + '/' + '%d_%d_%d_Epoch%d_Loc%.4f-Conf%.4f.pth'
                       % (dt.day, dt.hour, dt.minute, epoch, loc_loss / (iteration + 1),
                          conf_loss / (iteration + 1)))
