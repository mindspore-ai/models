"""pre_process_sysu.py"""
# from
# github.com/mangye16/Cross-Modal-Re-ID-baseline/blob/master/pre_process_sysu.py
import os
import numpy as np
from PIL import Image

# todo_change your own path
data_path = '/disk1/zzw/dataset/sysu'

rgb_cameras = ['cam1', 'cam2', 'cam4', 'cam5']
ir_cameras = ['cam3', 'cam6']

# load id info
file_path_train = os.path.join(data_path, 'exp/train_id.txt')
file_path_val = os.path.join(data_path, 'exp/val_id.txt')
with open(file_path_train, 'r', encoding='utf-8') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_train = ["%04d" % x for x in ids]

with open(file_path_val, 'r', encoding='utf-8') as file:
    ids = file.read().splitlines()
    ids = [int(y) for y in ids[0].split(',')]
    id_val = ["%04d" % x for x in ids]

# combine train and val split
id_train.extend(id_val)

files_rgb = []
files_ir = []
for id_num in sorted(id_train):
    for cam in rgb_cameras:
        img_dir = os.path.join(data_path, cam, id_num)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_rgb.extend(new_files)

    for cam in ir_cameras:
        img_dir = os.path.join(data_path, cam, id_num)
        if os.path.isdir(img_dir):
            new_files = sorted([img_dir+'/'+i for i in os.listdir(img_dir)])
            files_ir.extend(new_files)

# relabel
pid_container = set()
for image_path in files_ir:
    pid_num = int(image_path[-13:-9])
    pid_container.add(pid_num)
pid2label = {pid_num: label for label, pid_num in enumerate(pid_container)}
fix_image_width = 144
fix_image_height = 288


def read_imgs(train_image):
    """
    function of reading images
    """
    train_img = []
    train_label = []
    for img_path in train_image:
        # img
        img = Image.open(img_path)
        img = img.resize((fix_image_width, fix_image_height), Image.ANTIALIAS)
        pix_array = np.array(img)

        train_img.append(pix_array)

        # label
        pid = int(img_path[-13:-9])
        pid = pid2label[pid]
        train_label.append(pid)
    return np.array(train_img), np.array(train_label)


# rgb imges
train_imgs, train_labels = read_imgs(files_rgb)
np.save(os.path.join(data_path, 'train_rgb_resized_img.npy'), train_imgs)
np.save(os.path.join(data_path, 'train_rgb_resized_label.npy'), train_labels)

# ir imges
train_imgs, train_labels = read_imgs(files_ir)
np.save(os.path.join(data_path, 'train_ir_resized_img.npy'), train_imgs)
np.save(os.path.join(data_path, 'train_ir_resized_label.npy'), train_labels)
