# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import os
import onnxruntime as ort
import mindspore.context as context
from src.utils.inference import get_img, onnx_inference, MPIIEval, parse_args
args = parse_args()

def create_session(onnx_path, target_device):
    """
        Create onnx session.
    """
    if target_device == 'GPU':
        providers = ['CUDAExecutionProvider']
    elif target_device == 'CPU':
        providers = ['CPUExecutionProvider']
    else:
        raise ValueError(
            f'Unsupported target device {target_device}, '
            f'Expected one of: "CPU", "GPU"'
        )
    session = ort.InferenceSession(onnx_path, providers=providers)
    input_name = [x.name for x in session.get_inputs()]

    return session, input_name

def hourglass_onnx_inference():
    """
        Onnx inference
    """
    session, input_name = create_session(args.onnx_file, args.device_target)
    gts = []
    preds = []
    normalizing = []

    num_eval = args.num_eval
    num_train = args.train_num_eval
    for anns, img, c, s, n in get_img(num_eval, num_train):
        gts.append(anns)
        ans = onnx_inference(img, session, input_name, c, s)
        if ans.size > 0:
            ans = ans[:, :, :3]
        pred = []
        for i in range(ans.shape[0]):
            pred.append({"keypoints": ans[i, :, :]})
        preds.append(pred)
        normalizing.append(n)

    mpii_eval = MPIIEval()
    mpii_eval.eval(preds, gts, normalizing, num_train)


if __name__ == "__main__":

    if not os.path.exists(args.onnx_file):
        print("onnx file not valid")
        exit()

    if not os.path.exists(args.img_dir) or not os.path.exists(args.annot_dir):
        print("Dataset not found.")
        exit()
    # Set context mode
    if args.context_mode == "GRAPH":
        context.set_context(mode=context.GRAPH_MODE, device_target=args.device_target, save_graphs=False)
    else:
        context.set_context(mode=context.PYNATIVE_MODE, device_target=args.device_target)
    hourglass_onnx_inference()
