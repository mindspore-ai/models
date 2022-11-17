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
"""quick start"""
# ## This paper mainly visualizes the prediction data, uses the model to predict, and visualizes the prediction results.
# In[1]:
import os
import matplotlib.pyplot as plt
import numpy as np

from mindspore import Tensor
from mindspore.train import Model
from mindspore import load_checkpoint, load_param_into_net
from mindspore.communication.management import init

from model_utils.config import config
from src.dataset import create_dataset_imagenet as create_dataset
from src.squeezenet import SqueezeNet as squeezenet


if config.run_distribute:
    if config.device_target == "Ascend":
        # Ascend target
        device_id = int(os.getenv('DEVICE_ID'))
        init()
    elif config.device_target == "GPU":
        # GPU target
        init()
    else:
        raise TypeError(
            f"target type wrong, when run_distribute is Ture, expect [CPU/GPU], but got {config.device_target}")

# load inference dataset
test_path = os.path.join(config.train_url, 'test')
val_dataset = create_dataset(dataset_path=test_path,
                             do_train=False,
                             batch_size=64,
                             run_distribute=config.run_distribute)

# class_name corresponds to label,and labels are marked in the order of folders
class_name = {0: "daisy", 1: "dandelion", 2: "roses", 3: "sunflowers", 4: "tulips"}


# define visual prediction data functions：
def visual_input_data(dataset):
    data = next(dataset.create_dict_iterator())
    images = data["image"]
    labels = data["label"]
    print("Tensor of image", images.shape)
    print("Labels:", labels)
    plt.figure(figsize=(15, 7))
    for i in range(len(labels)):
        # get the image and its corresponding label
        data_image = images[i].asnumpy()
        # data_label = labels[i]
        # process images for display
        data_image = np.transpose(data_image, (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        data_image = std * data_image + mean
        data_image = np.clip(data_image, 0, 1)
        # display image
        plt.subplot(8, 8, i+1)
        plt.imshow(data_image)
        plt.title(class_name[int(labels[i].asnumpy())], fontsize=10)
        plt.axis("off")

    plt.show()

visual_input_data(val_dataset)


# define visualize_model()，visualize model prediction
def visualize_model(best_ckpt_path, val_ds):
    network = squeezenet(num_classes=config.class_num)
    # load model parameters
    param_dict = load_checkpoint(best_ckpt_path)
    load_param_into_net(network, param_dict)
    model = Model(network)
    # load the data of the validation set for validation
    data = next(val_ds.create_dict_iterator())
    images = data["image"].asnumpy()
    labels = data["label"].asnumpy()
    flower_class_name = {0: "daisy", 1: "dandelion", 2: "roses", 3: "sunflowers", 4: "tulips"}
    # prediction image category
    output = model.predict(Tensor(data['image']))
    pred = np.argmax(output.asnumpy(), axis=1)

    # display the image and the predicted value of the image
    plt.figure(figsize=(15, 7))
    for i in range(len(labels)):
        plt.subplot(8, 8, i + 1)
        # if the prediction is correct, it is displayed in blue; if the prediction is wrong, it is displayed in red
        color = 'blue' if pred[i] == labels[i] else 'red'
        plt.title('predict:{}'.format(flower_class_name[pred[i]]), color=color)
        picture_show = np.transpose(images[i], (1, 2, 0))
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        picture_show = std * picture_show + mean
        picture_show = np.clip(picture_show, 0, 1)
        plt.imshow(picture_show)
        plt.axis('off')

    plt.show()


# the best ckpt file obtained by model tuning is used to predict the images of the validation set
# (need to go to cpu_default_config.yaml set ckpt_path as the best ckpt file path）
visualize_model(config.pre_trained, val_dataset)
