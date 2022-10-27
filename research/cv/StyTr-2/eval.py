import argparse
import os
from os.path import splitext, basename
import random
import time
from pathlib import Path
from PIL import Image
import mindspore
from mindspore import context, ops, load_checkpoint, load_param_into_net
from mindspore.dataset.vision import c_transforms, py_transforms
from mindspore.dataset.transforms.c_transforms import Compose
from src.models import StyTR, transformer
from src.utils.function import Msave_image

context.set_context(mode=context.GRAPH_MODE, device_target='GPU', device_id=0)


def test_transform(shape, crops):
    transforms_l = []
    if shape != 0:
        transforms_l.append(c_transforms.Resize(shape))
    if crops:
        transforms_l.append(c_transforms.CenterCrop(shape))
    transforms_l.append(py_transforms.ToTensor())
    transform = Compose(transforms_l)
    return transform


def content_transform():
    transforms_l = []
    transforms_l.append(py_transforms.ToTensor())
    transform = Compose(transforms_l)
    return transform


def style_transform(h, w):
    transform_list = []
    transform_list.append(c_transforms.CenterCrop((h, w)))
    transform_list.append(py_transforms.ToTensor())
    transform = Compose(transform_list)
    return transform


def get_arg():
    parser = argparse.ArgumentParser()
    # Basic options
    parser.add_argument('--content', type=str, default='',
                        help='the content image file path')
    parser.add_argument('--content_dir', type=str, default='COCO2014/val2014',
                        help='content images directory')
    parser.add_argument('--style', type=str, default='',
                        help='the style image file path')
    parser.add_argument('--style_dir', type=str, default='wikiart/test',
                        help='style images directory')
    parser.add_argument('--output', type=str, default='output',
                        help='the path to save the output image(s)')
    parser.add_argument('--decoder_path', type=str, default='decoder.ckpt')
    parser.add_argument('--trans_path', type=str, default='transformer.ckpt')
    parser.add_argument('--embedding_path', type=str, default='embedding.ckpt')
    args = parser.parse_args()
    return args


def main():
    args = get_arg()
    content_size = 512
    style_size = 512
    crop = 'store_true'
    save_ext = '.jpg'
    output_path = args.output

    if args.content:
        content_paths = [Path(args.content_dir)]
    else:
        content_dir = Path(args.content_dir)
        content_paths = [f for f in content_dir.glob('*')]

    # Either --style or --style_dir should be given.
    style_paths = []
    if args.style:
        style_paths = [Path(args.style)]
    else:
        path = os.listdir(args.style_dir)
        if os.path.isdir(os.path.join(args.style_dir, path[0])):

            for file_name in os.listdir(args.style_dir):
                for file_name1 in os.listdir(os.path.join(args.style_dir, file_name)):
                    style_paths.append(args.style_dir + "/" + file_name + "/" + file_name1)
        else:
            style_paths = list(Path(args.style_dir).glob('*'))

    #当数据量很大时，这里只随机选取了一定数量的图片进行测试，若想测试全部图片，可以删除以下两行代码
    content_paths = random.sample(content_paths, 200)
    style_paths = random.sample(style_paths, 50)

    if not os.path.exists(output_path):
        os.mkdir(output_path)

    decoder = StyTR.Decoder(False)
    decoder_parm_dict = load_checkpoint(args.decoder_path)
    load_param_into_net(decoder, decoder_parm_dict)
    Trans = transformer.Transformer()
    Trans_parm_dict = load_checkpoint(args.trans_path)
    load_param_into_net(Trans, Trans_parm_dict)
    embedding = StyTR.PatchEmbed()
    embedding_parm_dict = load_checkpoint(args.embedding_path)
    load_param_into_net(embedding, embedding_parm_dict)

    network = StyTR.StyTrans(decoder, embedding, Trans)
    network.set_train(False)
    network.set_grad(False)

    content_tf = test_transform(content_size, crop)
    style_tf = test_transform(style_size, crop)
    expand_dims = ops.ExpandDims()
    for content_path in content_paths:
        for style_path in style_paths:
            print(content_path)
            print(style_path)
            content = content_tf(Image.open(content_path).convert("RGB"))
            style = style_tf(Image.open(style_path).convert("RGB"))
            content = mindspore.Tensor(content, mindspore.float32)
            style = mindspore.Tensor(style, mindspore.float32)
            content = expand_dims(content, 0)
            style = expand_dims(style, 0)

            t = time.time()
            output = network(content, style)[0]
            print('eval step time is : ', time.time()-t)
            output_name = '{:s}/{:s}_stylized_{:s}{:s}'.format(
                output_path, splitext(basename(content_path))[0],
                splitext(basename(style_path))[0], save_ext
            )
            cat_op = ops.Concat()
            pic = cat_op((style, content))
            pic2 = cat_op((pic, output))
            Msave_image(pic2, output_name)


if __name__ == '__main__':
    main()
    print('completed!!')
