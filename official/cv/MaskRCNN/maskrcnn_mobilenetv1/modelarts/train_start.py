import argparse
import os
import subprocess
import moxing as mox

_CACHE_DATA_URL = "./cache/data"
_CACHE_TRAIN_URL = "./cache/train"


def _parse_args():
    parser = argparse.ArgumentParser('mindspore maskrcnn_mobilenetv1 training')
    parser.add_argument('--train_url', type=str, default='obs://neu-base/maskrcnn_1/modelarts/cache/train',
                        help='where training log and ckpts saved')
    parser.add_argument('--coco_root', type=str, default="./cache/data/cocodataset",
                        help='read coco root')
    parser.add_argument('--data_path', type=str, default="/cache/data/cocodataset",
                        help='read data ')
    # dataset
    parser.add_argument('--data_url', type=str, default='obs://neu-base/maskrcnn_1/modelarts/cache/data',
                        help='path of dataset')
    parser.add_argument('--batch_size', type=int, default=8,
                        help='batch size')
    parser.add_argument('--num_classes', type=int, default=81,
                        help='number of classes')
    parser.add_argument('--file_format', type=str, default="AIR",
                        choices=['AIR', 'MINDIR'],
                        help='output model formats')
    parser.add_argument('--save_checkpoint_path', type=str, default='./cache/train',
                        help='save ckpt file address')
    parser.add_argument('--instance_set', type=str, default='annotations/instances_{}.json',
                        help='instances address')
    # model

    parser.add_argument('--pre_trained', type=str, default='',
                        help='pretrained model')

    # train
    parser.add_argument('--epoch_size', type=int, default=1,
                        help='epoch num')
    parser.add_argument('--save_checkpoint_epochs', type=int, default=12,
                        help='how many epochs to save ckpt once')
    parser.add_argument('--device_target', type=str, default='Ascend',
                        choices=['Ascend', 'CPU', 'GPU'],
                        help='device where the code will be implemented. '
                             '(Default: Ascend)')
    parser.add_argument('--mindrecord_dir', type=str, default='./MindRecord_COCO',
                        help='set up MindRecord build path')
    parser.add_argument('--train_data_type', type=str, default='train2017',
                        help='storing train data sets')
    parser.add_argument('--val_data_type', type=str, default='val2017',
                        help='storing val data sets')
    parser.add_argument('--coco_classes', type=tuple, default=('background', 'person',
                                                               'bicycle', 'car', 'motorcycle', 'airplane',
                                                               'bus',
                                                               'train', 'truck', 'boat', 'traffic light',
                                                               'fire hydrant',
                                                               'stop sign', 'parking meter', 'bench', 'bird',
                                                               'cat', 'dog',
                                                               'horse', 'sheep', 'cow', 'elephant', 'bear',
                                                               'zebra',
                                                               'giraffe', 'backpack', 'umbrella', 'handbag',
                                                               'tie',
                                                               'suitcase', 'frisbee', 'skis', 'snowboard',
                                                               'sports ball',
                                                               'kite', 'baseball bat', 'baseball glove',
                                                               'skateboard',
                                                               'surfboard', 'tennis racket', 'bottle',
                                                               'wine glass', 'cup',
                                                               'fork', 'knife', 'spoon', 'bowl', 'banana',
                                                               'apple',
                                                               'sandwich', 'orange', 'broccoli', 'carrot',
                                                               'hot dog', 'pizza',
                                                               'donut', 'cake', 'chair', 'couch',
                                                               'potted plant', 'bed',
                                                               'dining table', 'toilet', 'tv', 'laptop',
                                                               'mouse', 'remote',
                                                               'keyboard', 'cell phone', 'microwave',
                                                               'oven', 'toaster', 'sink',
                                                               'refrigerator', 'book', 'clock', 'vase',
                                                               'scissors',
                                                               'teddy bear', 'hair drier', 'toothbrush'))
    parser.add_argument('--file_name', type=str, default="maskrcnn_mobilenetv1", help='output  file name')
    args, _ = parser.parse_known_args()
    return args


def _train(args, train_url, data_url, pretrained_checkpoint):
    train_file = os.path.join(os.path.dirname(os.path.dirname(os.path.realpath(__file__))),
                              "train.py")
    cmd = ["python", train_file,
           f"--train_url={os.path.abspath(train_url)}",
           f"--coco_root={args.coco_root}",
           f"--data_path={args.data_path}",
           f"--data_url={os.path.abspath(data_url)}",
           f"--num_classes={args.num_classes}",
           f"--pre_trained={pretrained_checkpoint}",
           f"--epoch_size={args.epoch_size}",
           f"--save_checkpoint_epochs={args.save_checkpoint_epochs}",
           f"--device_target={args.device_target}",
           f"--save_checkpoint_path={args.save_checkpoint_path}",
           f"--instance_set={args.instance_set}",
           f"--coco_classes={args.coco_classes}",
           f"--val_data_type={args.val_data_type}",
           f"--train_data_type={args.train_data_type}",
           f"--mindrecord_dir={args.mindrecord_dir}"]
    print(' '.join(cmd))
    process = subprocess.Popen(cmd, shell=False)
    return process.wait()


def _get_last_ckpt(ckpt_dir):
    ckpt_dir = os.path.join(ckpt_dir, 'ckpt_0')
    ckpt_files = [ckpt_file for ckpt_file in os.listdir(ckpt_dir)
                  if ckpt_file.endswith('.ckpt')]
    if not ckpt_files:
        print("No ckpt file found.")
        return None

    return os.path.join(ckpt_dir, sorted(ckpt_files)[-1])


def _export_air(args, ckpt_dir):
    ckpt_file = _get_last_ckpt(ckpt_dir)
    if not ckpt_file:
        return

    export_file = os.path.join(
        os.path.dirname(os.path.dirname(os.path.realpath(__file__))), "export.py")
    cmd = ["python", export_file,
           f"--file_format={args.file_format}",
           f"--ckpt_file={ckpt_file}",
           f"--num_classes={args.num_classes}",
           f"--file_name={os.path.join(_CACHE_TRAIN_URL,args.file_name)}"]
    print(f"Start exporting AIR, cmd = {' '.join(cmd)}.")
    process = subprocess.Popen(cmd, shell=False)
    process.wait()


def main():
    args = _parse_args()
    try:
        os.makedirs(_CACHE_TRAIN_URL, exist_ok=True)
        os.makedirs(_CACHE_DATA_URL, exist_ok=True)
        mox.file.copy_parallel(args.data_url, _CACHE_DATA_URL)
        train_url = _CACHE_TRAIN_URL
        data_url = _CACHE_DATA_URL
        pretrained_checkpoint = os.path.join(_CACHE_DATA_URL,
                                             args.pre_trained) if args.pre_trained else ""
        ret = _train(args, train_url, data_url, pretrained_checkpoint)
        _export_air(args, train_url)
        mox.file.copy_parallel(_CACHE_TRAIN_URL, args.train_url)
    except ModuleNotFoundError:
        train_url = _CACHE_TRAIN_URL
        data_url = _CACHE_DATA_URL
        pretrained_checkpoint = args.pre_trained
        ret = _train(args, train_url, data_url, pretrained_checkpoint)
        _export_air(args, train_url)
        mox.file.copy_parallel(_CACHE_TRAIN_URL, args.train_url)
    if ret != 0:
        exit(1)


if __name__ == '__main__':
    main()
