import os
import random
import shutil

def CopyFile(imageDir, te_rate, save_test_dir, save_train_dir):  # 三个参数，第一个为每个类别的所有图像在计算机中的位置
    # 第二个为copy的图片数目所占总的比例，最后一个为移动的图片保存的位置，
    image_number = len(imageDir)  # 图片总数目
    test_number = int(image_number * te_rate)  # 要移动的图片数目
    print("要移动到%s目录下的图片数目为:%d" % (save_test_dir, test_number))
    test_samples = random.sample(imageDir, test_number)  # 随机截取列表imageDir中数目为test_number的元素
    # copy图像到目标文件夹
    if not os.path.exists(save_test_dir):
        os.makedirs(save_test_dir)
        print("save_test_dir has been created successfully!")
    else:
        print("save_test_dir already exited!")
    if not os.path.exists(save_train_dir):
        os.makedirs(save_train_dir)
        print("save_train_dir has been created successfully!")
    else:
        print("save_train_dir already exited!")
    for j in test_samples:
        shutil.copy(test_samples[j], save_test_dir + test_samples[j].split("/")[-1])
    print("test移动完成！")
    for train_imgs in imageDir:
        if train_imgs not in test_samples:
            shutil.copy(train_imgs, save_train_dir + train_imgs.split("/")[-1])
    print("train移动完成")


if __name__ == '__main__':
    # 只需给定file_path、test_rate即可完成整个任务
    # 原始路径+分割比例
    ################################
    file_path = "/dataset/wikiart"
    test_rate = 0.1
    ################################
    file_dirs = os.listdir(file_path)
    origion_paths = []
    save_test_dirs = []
    save_train_dirs = []
    for path in file_dirs:
        origion_paths.append(file_path + "/" + path + "/")
        save_train_dirs.append(file_path + "/train/" + path + "/")
        save_test_dirs.append(file_path + "/test/" + path + "/")
    for i, origion_path in enumerate(origion_paths):
        image_list = os.listdir(origion_path)  # 获得原始路径下的所有图片的name（默认路径下都是图片）
        image_Dir = []
        for x, y in enumerate(image_list):
            image_Dir.append(os.path.join(origion_path, y))
        print("%s目录下共有%d张图片！" % (origion_path, len(image_Dir)))
        CopyFile(image_Dir, test_rate, save_test_dirs[i], save_train_dirs[i])
    print("all data has been moved successfully!")
