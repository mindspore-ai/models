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
"""infer_by_sdk"""
import argparse
import json
import os
from StreamManagerApi import MxDataInput
from StreamManagerApi import StreamManagerApi

# 支持的图片后缀，分别是这四个后缀名
SUPPORT_IMG_SUFFIX = (".jpg", ".JPG", ".jpeg", ".JPEG")

# os.path.dirname(__file__)获取当前脚本的完整路径，os.path.abspath()获取当前脚本的完整路径
current_path = os.path.abspath(os.path.dirname(__file__))

# argparse是个解析器，argparse块可以让人轻松编写用户友好的命令行接口,使用argparse首先要创建ArgumentParser对象，
parser = argparse.ArgumentParser(
    description="SSD ResNet50 infer " "example.",
    fromfile_prefix_chars="@",
)

# name or flags，一个命令或一个选项字符串的列表
# str将数据强制转换为字符串。每种数据类型都可以强制转换为字符串
# help 一个此选项作用的简单描述
# default 当参数未在命令行中出现时使用的值。
parser.add_argument(
    "--pipeline_path",
    type=str,
    help="mxManufacture pipeline file path",
    default=os.path.join(current_path, "../conf/ssd_resnet50.pipeline"),
)
parser.add_argument(
    "--stream_name",
    type=str,
    help="Infer stream name in the pipeline config file",
    default="detection",
)
parser.add_argument(
    "--img_path",
    type=str,
    help="Image pathname, can be a image file or image directory",
    default=os.path.join(current_path, "../coco/val2017"),
)
# 目的用于存放推理后的结果
parser.add_argument(
    "--res_path",
    type=str,
    help="Directory to store the inferred result",
    default=None,
    required=False,
)

# 赋值然后解析参数
args = parser.parse_args()

# 推理图像
def infer():
    """Infer images by DVPP + OM.    """
    pipeline_path = args.pipeline_path
    # 将stream_name编码为utf-8的格式
    stream_name = args.stream_name.encode()
    img_path = os.path.abspath(args.img_path)
    res_dir_name = args.res_path

    # StreamManagerApi()用于对流程的基本管理：加载流程配置、创建流程、向流程上发送数据、获得执行结果
    stream_manager_api = StreamManagerApi()
    # InitManager初始化一个StreamManagerApi
    ret = stream_manager_api.InitManager()
    if ret != 0:
        print("Failed to init Stream manager, ret=%s" % str(ret))
        exit()

    # create streams by pipeline config file
    # 读取pipeline文件
    with open(pipeline_path, "rb") as f:
        pipeline_str = f.read()

    # CreateMultipleStreams，根据指定的pipeline配置创建Stream
    ret = stream_manager_api.CreateMultipleStreams(pipeline_str)
    if ret != 0:
        print("Failed to create Stream, ret=%s" % str(ret))
        exit()

    # 插件的id
    in_plugin_id = 0
    # Construct the input of the stream
    # 构造stream的输入，MxDataInput用于Stream接收的数据结构定义。
    data_input = MxDataInput()

    # os.path.isfile()用于判断某一对象(需提供绝对路径)是否为文件
    # endswith用于判断是否为指定的图片的字符串结尾
    if os.path.isfile(img_path) and img_path.endswith(SUPPORT_IMG_SUFFIX):
        file_list = [os.path.abspath(img_path)]
    else:
        # os.path.isdir()用于判断对象是否为一个目录
        file_list = os.listdir(img_path)
        file_list = [
            # 将图片路径和图片连接，for in if 过滤掉那些不符合照片后缀的图片
            os.path.join(img_path, img)
            for img in file_list
            if img.endswith(SUPPORT_IMG_SUFFIX)
        ]

    if not res_dir_name:
        res_dir_name = os.path.join(".", "infer_res")
    print(f"res_dir_name={res_dir_name}")
    # 创建目录，e目标目录已存在的情况下不会触发FileExistsError异常。
    os.makedirs(res_dir_name, exist_ok=True)
    pic_infer_dict_list = []
    # 开始对file_list进行遍历
    for file_name in file_list:
        # 依次读出每张照片
        with open(file_name, "rb") as f:
            img_data = f.read()
            if not img_data:
                print(f"read empty data from img:{file_name}")
                continue
            # data_input这个对象的data元素值为img_data
            data_input.data = img_data
            # SendDataWithUniqueId向指定的元件发送数据，输入in_plugin_id目标输入插件id，data_input ，根据官方的API,stream_name应该是可以不作为输入的
            unique_id = stream_manager_api.SendDataWithUniqueId(
                stream_name, in_plugin_id, data_input
            )
            if unique_id < 0:
                print("Failed to send data to stream.")
                exit()
            # 获得Stream上的输出元件的结果(appsink), 延时3000ms
            infer_result = stream_manager_api.GetResultWithUniqueId(
                stream_name, unique_id, 3000
            )
            if infer_result.errorCode != 0:
                print(
                    "GetResultWithUniqueId error. errorCode=%d, errorMsg=%s"
                    % (infer_result.errorCode, infer_result.data.decode())
                )
                exit()
            # 将推理的结果parse_img_infer_result追加到pic_infer_dict_list数组中
            pic_infer_dict_list.extend(
                parse_img_infer_result(file_name, infer_result)
            )

        print(f"Inferred image:{file_name} success!")

    with open(os.path.join(res_dir_name, "det_result.json"), "w") as fw:
        # 将Python格式转为json格式并且写入
        fw.write(json.dumps(pic_infer_dict_list))

    stream_manager_api.DestroyAllStreams()


def parse_img_infer_result(file_name, infer_result):
    """parse_img_infer_result"""
    # 将infer_result.data即元器件返回的结果转为dict格式，用get("MxpiObject", [])新建一个MxpiObject的Key且复制为[]
    obj_list = json.loads(infer_result.data.decode()).get("MxpiObject", [])
    det_obj_list = []
    for o in obj_list:
        # 把图片框一个框，一个正方形，四个角的位置，坐标位置
        x0, y0, x1, y1 = (
            # round()函数，四舍五入到第四位
            round(o.get("x0"), 4),
            round(o.get("y0"), 4),
            round(o.get("x1"), 4),
            round(o.get("y1"), 4),
        )
        bbox_for_map = [x0, y0, x1 - x0, y1 - y0]
        score = o.get("classVec")[0].get("confidence")
        category_id = o.get("classVec")[0].get("classId")
        # basename()用于选取最后的文件名，即image的name,.split(".")用于把后缀给分割掉
        img_fname_without_suffix = os.path.basename(file_name).split(".")[0]
        image_id = img_fname_without_suffix
        det_obj_list.append(
            dict(
                image_id=image_id,
                bbox=bbox_for_map,
                # 目录id，把图片归类
                category_id=category_id,
                # 置信度的问题，过滤掉比较小的，意思就是说假如我这边猜个猫的机率为0.1,那就是大概率不是猫，那这个数据就可以筛掉了
                # ssd_mobilenet_v1_fpn_ms_on_coco_postprocess.cfg文件里面的 SCORE_THRESH=0.6设置
                score=score,
            )
        )
    return det_obj_list
if __name__ == "__main__":
    infer()
