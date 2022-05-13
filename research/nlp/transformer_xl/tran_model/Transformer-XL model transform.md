## Transformer-XL model transform

作者在GitHub提供的源代码一共提供了PyTorch和TensorFlow两种版本的代码，这里提供了将PyTorch和TensorFlow训练好的pt模型转为MindSpore的ckpt模型的方案和具体操作

由于Transformer-XL源代码所需要的环境版本较低，并且高版本的环境会出现代码无法正常运行等问题，因此强烈建议先配置好Transformer-XL作者提供的源代码所需要的环境。模型转化的思路是，先在作者源代码所需的环境下，通过训练/下载对应模型的方式，将模型转化为numpy格式下的.pkl参数文件，再切换到MindSpore环境下将.pkl的参数文件传入MindSpore模型并保存为.ckpt文件。为了保证模型的正常运行，在保存模型后，加入了对test数据集的推理。

论文官方源代码（包含PyTorch版本与TensorFlow版本）：[点此](https://github.com/kimiyoung/transformer-xl)

作者提供的enwik8_large与text8_large模型链接：[enwik8_large](http://curtis.ml.cmu.edu/datasets/pretrained_xl/tf_enwiki8/) ; [text8_large](http://curtis.ml.cmu.edu/datasets/pretrained_xl/tf_text8/)

### PyTorch2MindSpore

所需环境：

- Python：3.7.5

- PyTorch：0.4.0

```shell
# Step1：将/tran_model/torch_get_param.py和/tran_model/torch_get_param.sh拷贝到源代码的/pytorch/目录下
cp "/home/transformer-xl/tran_model/torch_get_param.py" "/home/txl_author/pytorch/torch_get_param.py" 
cp "/home/transformer-xl/tran_model/torch_get_param.sh" "/home/txl_author/pytorch/torch_get_param.sh" 
# Step2：在PyTorch0.4.0环境下运行torch_get_param.sh，将模型参数取出转为numpy格式，并存为.pkl文件，其中[DATA_SET]为数据集名称，例如enwik8/text8，[WORK_DIR]为模型所在路径，因为PyTorch训练得到的模型默认名称为model.pt
cd /home/txl_author/pytorch/
bash torch_get_param.sh [DATA_SET] [WORK_DIR]
# bash torch_get_param.sh "enwik8" "/home/ganweikang/project/txl_torch/pytorch/LM-TFM-enwik8/20220322-202922/"
# Step3：切换到高版本PyTorch下，将model.state_dict中的参数转为numpy，[WORK_DIR]为Step2中保存的enwik8_base.pkl所在的路径
cd /home/transformer-xl/tran_model/torch2msp
bash torch2numpy.sh [DATA_SET] [WORK_DIR]
# bash torch2numpy.sh "enwik8" "/home/ganweikang/project/txl_torch/pytorch/"
# Step4：切换到MindSpore环境下，执行torch2msp.sh，将numpy格式的.pkl文件传入MindSpore模型并保存为.ckpt文件并执行一次test数据集的推理
cd /home/transformer-xl/tran_model/
bash torch2msp.sh [DATA_DIR] [DATA_NAME] [TORCH_PT_PATH]
     [CONFIG_PATH] [DEVICE_ID(optional)]
# bash torch2msp.sh "/home/ganweikang/project/transformer-xl/data/enwik8/" "enwik8" "/home/ganweikang/project/txl_0512/tran_model/torch2msp/enwik8_base.pkl" "/home/ganweikang/project/txl_0512/yaml/enwik8_base_eval.yaml"

```



### TensorFlow2MindSpore

所需环境：

- Python：2.7

- TensorFlow：1.12.0

```shell
# Step1：将/tran_model/tf_get_param.py和/tran_model/tf_get_param.sh拷贝到源代码的/tf/目录下
cp "/home/transformer-xl/tran_model/tf_get_param.py" "/home/txl_author/tf/torch_get_param.py" 
cp "/home/transformer-xl/tran_model/tf_get_param.sh" "/home/txl_author/tf/torch_get_param.sh" 
# Step2：在TensorFlow环境下运行tf_get_param.sh，将模型参数取出转为numpy格式，并存为.pkl文件，其中[DATA_SET]为数据集名称，例如enwik8/text8。
cd /home/txl_author/tf/
bash tf_get_param.sh [DATA_SET]
# Step3：切换到MindSpore环境下，执行tf2msp.sh，将.pkl文件传入MindSpore模型并保存为.ckpt文件并执行一次test数据集的推理
bash tf2msp.sh "/home/transformer-xl/data/text8" "text8" "/home/txl_author/tf/text8_large.pkl" "/home/transformer-xl/yaml/text8_large_eval.yaml"
```
