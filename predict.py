from ssd import SSD
from PIL import Image
import time
import cv2
import os

ssd = SSD()

# name = input("请输入要预测的图片路径（文件夹）：")
# file_list = os.listdir(name)
# save_path = str(input('请输入保存路径：'))
# for i in file_list:
#     r_image = Image.open(name+ '/' + i,'r')
#     det = ssd.detect_image(r_image)
#     print(det)
#     r_image.save(save_path+str(i) + ".bmp")

name_path = 'C:/Users/520/Desktop/2/train_val_test/test.txt'
with open(name_path, 'r') as f:
    j =0
    for i in f:

        t1 = i.split(' ')[0]
        # img = t1.strip('\n')
        try:
            image_1 = Image.open(t1)
        except:
            print('Open Error! Try again!')
            continue
        else:
            start_time = time.time()
            r_image = ssd.detect_image(image_1)
            end = time.time() - start_time
            print("gigigigi------",end)
            new_path = r"C:/Users/520/Desktop/2/2/" + str(j) + ".jpg"
            #cv2.imwrite(new_path,r_image)
            r_image.save(new_path)
            #plt.savefig(new_path)
            j += 1

