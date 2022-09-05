import argparse
import os
import numpy as np
from PIL import Image

val_origin_size = True
val_save_result = True
_STATUS_CHECK_ITERATIONS = 10

class config:
    image_scales = [0.25, 0.3536, 0.5, 0.7071, 1.0, 1.4142, 2.0]
    max_feature_num = 1000
    score_threshold = 100.0

def parse_args():
    parser = argparse.ArgumentParser(description='DELF prepocess')
    # Datasets
    parser.add_argument('--images_path', default='../data/ox/', type=str,
                        help='data path')
    parser.add_argument('--use_list_txt', type=str, default="False", choices=['True', 'False'])
    parser.add_argument('--list_images_path', type=str, default="list_images.txt")
    parser.add_argument('--pre_path', type=str, default="./Preprocess_result")
    parser.add_argument('--resultnpz_path', type=str, default="./results/eval_features")
    parser.add_argument('--batchsize', type=int, default=10)
    arg = parser.parse_args()

    return arg

def ReadFromFile(file_path):
    data = np.load(file_path)
    return data

def safe_makedirs(path_dir):
    if not os.path.exists(path_dir):
        os.makedirs(path_dir)

def ReadImageList(list_path):
    f = open(list_path, "r")
    image_paths = f.readlines()
    image_paths = [entry.rstrip() for entry in image_paths]
    return image_paths

def preprocess(i, num_images, image_paths, image_scales_tensor, counter):
    """preprocess images"""
    args = parse_args()
    a = 1
    if a > 0:
        # If descriptor already exists, skip its computation.
        out_desc_filename = os.path.splitext(os.path.basename(
            image_paths[i]))[0]
        out_desc_fullpath = os.path.join(args.pre_path, 'images_batch', out_desc_filename)
        out_npz_fullpath = os.path.join(args.resultnpz_path, out_desc_filename)
        size_list_fullpath = os.path.join(args.pre_path, 'size_list', out_desc_filename)
        print(out_npz_fullpath)
        if os.path.exists(out_npz_fullpath+'.feature'+'.npz'):
            print(f'Skipping {image_paths[i]}')
            return 0
        safe_makedirs(os.path.join(args.pre_path, 'images_batch'))
        safe_makedirs(os.path.join(args.pre_path, 'size_list'))
        img = Image.open(os.path.join(args.images_path, image_paths[i]) + '.jpg')

        im = np.array(img, np.float32)
        original_image_shape = np.array([im.shape[0], im.shape[1]])
        original_image_shape_float = original_image_shape.astype(np.float32)
        new_image_size = np.array([2048, 2048])

        images_batch = np.zeros((image_scales_tensor.shape[0], 3, 2048, 2048), np.float32)
        size_list = []

        for j in range(image_scales_tensor.shape[0]):
            scale_size = np.round(original_image_shape_float * image_scales_tensor[j]).astype(int)
            size_list.append(scale_size)
            img_pil = img.resize((scale_size[1], scale_size[0]))
            scale_image = np.array(img_pil, np.float32)

            H_pad = new_image_size[0] - scale_size[0]
            W_pad = new_image_size[1] - scale_size[1]
            new_image = np.pad(scale_image, ((0, H_pad), (0, W_pad), (0, 0)))

            new_image = (new_image-128.0) / 128.0

            perm = (2, 0, 1)
            new_image = np.transpose(new_image, perm)
            new_image = np.expand_dims(new_image, 0)
            images_batch[j] = new_image
        size_list = np.array(size_list)
        images_batch.tofile(out_desc_fullpath)
        size_list.tofile(size_list_fullpath)
        return 1
    return 1

def main(args):
    safe_makedirs(args.pre_path)
    # create testing dataset
    if args.use_list_txt == "True":
        image_paths = ReadImageList(args.list_images_path)
    else:
        names = os.listdir(args.images_path)
        image_paths = []
        for name in names:
            if '.txt' in name:
                continue
            image_name = name.replace('.jpg', '')
            image_paths.append(image_name)

    num_images = len(image_paths)
    print(f'done! Found {num_images} images')

    image_scales_tensor = np.array(config.image_scales, np.float32)
    counter = 0
    for i in range(num_images):
        img = preprocess(i, num_images, image_paths, image_scales_tensor, counter)
        counter = counter + img
        print(counter)
        print("all:", args.batchsize, ",finished:", i+1)
        if counter == args.batchsize:
            return
    return
if __name__ == '__main__':
    arg_ = parse_args()
    main(arg_)
