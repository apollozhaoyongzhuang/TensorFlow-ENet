# 把path文件夹下以及其子文件下的所有png图片移动（不是复制）到new_path

import os
import shutil

path = '/home/yong/dataset/Cityscapes/gtFine_trainvaltest/gtFine/train'
# path = '/home/yong/dataset/Cityscapes/leftImg8bit_trainvaltest/leftImg8bit/train'
new_path = '/home/yong/code/TensorFlow-ENet/dataset/Cityscapes_trainannot'

#####image_GT
for root, dirs, files in os.walk(path):
    if len(dirs) == 0:
        for i in range(len(files)):
            if files[i][-12:-4] == 'labelIds':
                file_path = root + '/' + files[i]
                new_file_path = new_path + '/' + files[i][:-19] + 'leftImg8bit.png'
                shutil.copy(file_path, new_file_path)

# ####image
# for root, dirs, files in os.walk(path):
#     if len(dirs) == 0:
#         for i in range(len(files)):
#             # if files[i][-12:-4] == 'labelIds':
#                 file_path = root + '/' + files[i]
#                 new_file_path = new_path + '/' + files[i]
#                 shutil.copy(file_path, new_file_path)

#
# import os
# import shutil
#
# print('输入格式：E:\myprojectnew\jupyter\整理文件夹\示例')
# path = input('请键入需要整理的文件夹地址：')
# new_path = input('请键入要复制到的文件夹地址：')
#
# for root, dirs, files in os.walk(path):
#     for i in range(len(files)):
#         # print(files[i])
#         if (files[i][-3:] == 'jpg') or (files[i][-3:] == 'png') or (files[i][-3:] == 'JPG'):
#             file_path = root + '/' + files[i]
#             new_file_path = new_path + '/' + files[i]
#             shutil.move(file_path, new_file_path)
#
#             # yn_close = input('是否退出？')
