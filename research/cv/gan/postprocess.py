# Copyright 2021 Huawei Technologies Co., Ltd
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
""" postprocess """
import os
import time
import gc
import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
from src.param_parse import parameter_parser
from src.dataset import load_test_data

opt = parameter_parser()
print(opt)

def log_mean_exp(a):
    max_ = a.max(axis=1)
    max2 = max_.reshape(max_.shape[0], 1)
    return max_ + np.log(np.exp(a - max2).mean(1))

def mind_parzen(x, mu, sigma):
    ''' mind parzen '''
    a = (x.reshape(x.shape[0], 1, x.shape[-1]) - mu.reshape(1, mu.shape[0], mu.shape[-1])) / sigma
    a5 = -0.5 * (a ** 2).sum(2)
    E = log_mean_exp(a5)
    t4 = sigma * np.sqrt(np.pi * 2)
    t5 = np.log(t4)
    Z = mu.shape[1] * t5
    return E - Z

def get_nll(x, samples, sigma, batch_size=10):
    '''get_nll'''
    inds = range(x.shape[0])
    inds = list(inds)
    n_batches = int(np.ceil(float(len(inds)) / batch_size))
    times = []
    nlls = np.array([]).astype(np.float32)
    for l in range(n_batches):
        begin = time.time()
        nll = mind_parzen(x[inds[l::n_batches]], samples, sigma)
        end = time.time()
        times.append(end - begin)
        nlls = np.concatenate((nlls, nll))
        if l % 10 == 0:
            print(l, np.mean(times), nlls.mean())

    return nlls

def cross_validate_sigma(samples, data, sigmas, batch_size):
    '''cross_validate_sigma'''
    lls = np.array([]).astype(np.float32)
    for sigma in sigmas:
        print(sigma)
        tmp = get_nll(data, samples, sigma, batch_size=batch_size)
        tmp = tmp.mean()
        tmp = tmp.reshape(1, 1)
        tmp = tmp.squeeze()
        lls = np.concatenate((lls, tmp))
        gc.collect()

    ind = lls.argmax()
    return sigmas[ind]

def get_test():
    dataset = load_test_data().astype("float32")
    image = dataset
    image = image.reshape(image.shape[0], 784)
    return image

def parzen(samples):
    '''parzen'''
    shape = samples.shape
    samples = samples.reshape(shape[0], -1)
    sigma = 0.16681005372000587
    print("Using Sigma: {}".format(sigma))
    gc.collect()
    test_data = get_test()
    test_data = test_data / 255
    ll = get_nll(test_data, samples, sigma, batch_size=opt.batch_size_t)
    se = ll.std() / np.sqrt(test_data.shape[0])
    print("Log-Likelihood of test set = {}, se: {}".format(ll.mean(), se))
    return ll.mean(), se

test_latent_code_parzen = np.random.normal(size=(10000, opt.latent_dim)).astype(np.float32)

def save_imgs2(gen_imgs, idx):
    '''save images'''
    index = gen_imgs < 1
    gen_imgs[index] = 0
    index = gen_imgs > 250
    gen_imgs[index] = 255
    for k in range(gen_imgs.shape[0]):
        plt.subplot(5, 20, k + 1)
        plt.imshow(gen_imgs[k, 0, :, :], cmap="gray")
        plt.axis("off")

    plt.savefig("./images2/{}.png".format(idx))

if __name__ == "__main__":
    args_opt = parameter_parser()
    imageSize = args_opt.img_size
    nc = 1
    f_name = os.path.join("ascend310_infer/result_files", 'gan_bs_0.bin')
    fake = np.fromfile(f_name, np.float32).reshape(10000, nc, imageSize, imageSize)
    fake = np.multiply(fake, 0.5*255)
    fake = np.add(fake, 0.5*255)
    for j in range(10000):
        img_pil = fake[j, ...].reshape(1, nc, imageSize, imageSize)
        img_pil = img_pil[0].astype(np.uint8)
        img_pil = img_pil[0].astype(np.uint8)
        img_pil = Image.fromarray(img_pil)
        img_pil.save(os.path.join("images", "generated_%02d.png" % j))

    print("Generate images success!")
    size = 28
    imag = []
    for i in range(10000):
        img_temp = mpimg.imread(os.path.join("images", "generated_%02d.png" % i))
        img_temp = img_temp.reshape(1, size, size)
        imag.append(img_temp)

    imag = np.array(imag)
    samples1 = imag
    mean_ll, se_ll = parzen(samples1)
    print("Log-Likelihood of test set = {}, se: {}".format(mean_ll, se_ll))
