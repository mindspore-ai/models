import os
import argparse
import subprocess
import time
import shutil
from mindspore import context
import onnxruntime
from detect import detect_dataset_onnx

parser = argparse.ArgumentParser('mindspore icdar eval')

# device related
parser.add_argument(
    '--device_target',
    type=str,
    default='GPU',
    help='device where the code will be implemented. (Default: GPU)')
parser.add_argument(
    '--device_num',
    type=int,
    default=5,
    help='device where the code will be implemented. (Default: Ascend)')

parser.add_argument(
    '--test_img_path',
    default='./data/icdar2015/Test/image/',
    type=str,
    help='Train dataset directory.')

parser.add_argument('--onnx_path', default='east_gpu.onnx', type=str,
                    help='The onnx file of EAST. Default: "".')
args, _ = parser.parse_known_args()

context.set_context(
    mode=context.GRAPH_MODE,
    device_target=args.device_target,
    save_graphs=False,
    device_id=args.device_num)


def eval_model(session, test_img_path, submit, save_flag=True):
    start_time = time.time()
    if os.path.exists(submit):
        shutil.rmtree(submit)
    os.mkdir(submit)
    detect_dataset_onnx(session, test_img_path, submit)
    os.chdir(submit)

    res = subprocess.getoutput('zip -q submit.zip *.txt')
    res = subprocess.getoutput('mv submit.zip ../')
    os.chdir('../')
    res = subprocess.getoutput(
        'python ./evaluate/script.py -g=./evaluate/gt.zip -s=./submit.zip')
    print(res)
    os.remove('./submit.zip')
    print('eval time is {}'.format(time.time() - start_time))

    if not save_flag:
        shutil.rmtree(submit)


if __name__ == '__main__':
    dataset_path = args.test_img_path
    onnx_path = args.onnx_path
    print("onnx_path = ", onnx_path)
    submit_path = './submit'
    print(args.device_target)
    if args.device_target == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif args.device_target == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {args.device_target}, '
            f'Expected one of: "CPU", "GPU"'
        )
    Session = onnxruntime.InferenceSession(onnx_path, provider_options=providers)
    eval_model(Session, dataset_path, submit_path)
