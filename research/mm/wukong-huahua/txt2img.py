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

# This file refers to https://github.com/CompVis/stable-diffusion/blob/main/scripts/txt2img.py
import os
import argparse
from itertools import islice
import cv2
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
import mindspore as ms

from ldm.util import instantiate_from_config
from ldm.models.diffusion.plms import PLMSSampler


def seed_everything(seed):
    if seed:
        ms.set_seed(seed)
        np.random.seed(seed)


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def numpy_to_pil(images):
    """
    Convert a numpy image or a batch of images to a PIL image.
    """
    if images.ndim == 3:
        images = images[None, ...]
    images = (images * 255).round().astype("uint8")
    pil_images = [Image.fromarray(image) for image in images]

    return pil_images


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    model = instantiate_from_config(config.model)
    if os.path.exists(ckpt):
        param_dict = ms.load_checkpoint(ckpt)
        if param_dict:
            param_not_load, _ = ms.load_param_into_net(model, param_dict)
            print("param not load:", param_not_load)
    else:
        print(f"{ckpt} not exist:")

    return model

def put_watermark(img, wm_encoder=None):
    if wm_encoder is not None:
        img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        img = wm_encoder.encode(img, 'dwtDct')
        img = Image.fromarray(img[:, :, ::-1])
    return img


def load_replacement(x):
    try:
        hwc = x.shape
        y = Image.open("assets/rick.jpeg").convert("RGB").resize((hwc[1], hwc[0]))
        y = (np.array(y)/255.0).astype(x.dtype)
        assert y.shape == x.shape
        return y
    except (Exception,): # pylint: disable=broad-except
        return x


def get_user_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--prompt", type=str, nargs="?", default="狗 绘画 写实风格", help="the prompt to render")
    parser.add_argument("--outdir", type=str, nargs="?",
                        help="dir to write results to", default="outputs/txt2img-samples")
    parser.add_argument("--skip_grid", action='store_true',
                        help="do not save a grid, only individual samples. Helpful when evaluating lots of samples")
    parser.add_argument("--skip_save", action='store_true',
                        help="do not save individual samples. For speed measurements.")
    parser.add_argument("--ddim_steps", type=int, default=50, help="number of ddim sampling steps")
    parser.add_argument("--plms", action='store_true', help="use plms sampling")
    parser.add_argument("--fixed_code", action='store_true',
                        help="if enabled, uses the same starting code across samples")
    parser.add_argument("--ddim_eta", type=float, default=0.0,
                        help="ddim eta (eta=0.0 corresponds to deterministic sampling")
    parser.add_argument("--n_iter", type=int, default=2, help="sample this often")
    parser.add_argument("--H", type=int, default=512, help="image height, in pixel space")
    parser.add_argument("--W", type=int, default=512, help="image width, in pixel space")
    parser.add_argument(
        "--n_samples",
        type=int,
        default=4,
        help="how many samples to produce for each given prompt. A.k.a. batch size",
    )
    parser.add_argument(
        "--n_rows",
        type=int,
        default=0,
        help="rows in the grid (default: n_samples)",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=7.5,
        help="unconditional guidance scale: eps = eps(x, empty) + scale * (eps(x, cond) - eps(x, empty))",
    )
    parser.add_argument(
        "--from-file",
        type=str,
        help="if specified, load prompts from this file",
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/v1-inference-chinese.yaml",
        help="path to config which constructs model",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/wukong-huahua-ms.ckpt",
        help="path to checkpoint of model",
    )
    parser.add_argument(
        "--seed",
        type=int,
        # default=42,
        default=None,
        help="the seed (for reproducible sampling)",
    )
    parser.add_argument(
        "--precision",
        type=str,
        help="evaluate at this precision",
        choices=["full", "autocast"],
        default="autocast"
    )
    opt = parser.parse_args()
    return opt


def main():
    opt = get_user_args()
    device_id = int(os.getenv("DEVICE_ID", '0'))
    ms.context.set_context(mode=ms.context.GRAPH_MODE,
                           device_target="Ascend",
                           device_id=device_id,
                           max_device_memory="30GB",
                           save_graphs=False,
                           save_graphs_path="graph/")

    seed_everything(opt.seed)

    config = OmegaConf.load(f"{opt.config}")
    model = load_model_from_config(config, f"{opt.ckpt}")

    sampler = PLMSSampler(model)
    os.makedirs(opt.outdir, exist_ok=True)
    outpath = opt.outdir

    batch_size = opt.n_samples

    if not opt.from_file:
        prompt = opt.prompt
        assert prompt is not None
        data = [batch_size * [prompt]]
    else:
        print(f"reading prompts from {opt.from_file}")
        with open(opt.from_file, "r") as f:
            data = f.read().splitlines()
            data = list(chunk(data, batch_size))

    sample_path = os.path.join(outpath, "samples")
    os.makedirs(sample_path, exist_ok=True)
    base_count = len(os.listdir(sample_path))

    start_code = None
    if opt.fixed_code:
        stdnormal = ms.ops.StandardNormal()
        start_code = stdnormal((opt.n_samples, 4, opt.H // 8, opt.W // 8))

    all_samples = list()
    from time import time
    last = time()
    for _ in range(opt.n_iter):
        for prompts in data:
            uc = None
            if opt.scale != 1.0:
                uc = model.get_learned_conditioning(batch_size * [""])
            if isinstance(prompts, tuple):
                prompts = list(prompts)
            c = model.get_learned_conditioning(prompts)
            shape = [4, opt.H // 8, opt.W // 8]
            samples_ddim, _ = sampler.sample(S=opt.ddim_steps,
                                             conditioning=c,
                                             batch_size=opt.n_samples,
                                             shape=shape,
                                             verbose=False,
                                             unconditional_guidance_scale=opt.scale,
                                             unconditional_conditioning=uc,
                                             eta=opt.ddim_eta,
                                             x_T=start_code)
            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = ms.ops.clip_by_value((x_samples_ddim + 1.0) / 2.0,
                                                  clip_value_min=0.0, clip_value_max=1.0)
            x_samples_ddim_numpy = x_samples_ddim.asnumpy()

            if not opt.skip_save:
                for x_sample in x_samples_ddim_numpy:
                    x_sample = 255. * x_sample.transpose(1, 2, 0)
                    img = Image.fromarray(x_sample.astype(np.uint8))
                    img.save(os.path.join(sample_path, f"{base_count:05}.png"))
                    base_count += 1

            if not opt.skip_grid:
                all_samples.append(x_samples_ddim_numpy)

            cur = time()
            print(cur - last)
            last = cur
        print(f"Your samples are ready and waiting for you here: \n{outpath} \n"
              f" \nEnjoy.")


if __name__ == "__main__":
    main()
