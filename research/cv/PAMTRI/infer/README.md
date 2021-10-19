# 推理

## 模型转换

1. 准备air模型文件。

   air文件路径：infer/models

2. 模型转换。

    ```
    model_path=$1
    output_model_name=$2
    /usr/local/Ascend/atc/bin/atc \
      --model=$model_path \                     # 待转换的air模型，模型可以通过训练生成或通过“下载模型”获得。
      --framework=1 \                           # 1代表MindSpore。
      --output=$output_model_name \             # 转换后输出的om模型。
      --log=error \                             # 日志级别。
      --soc_version=Ascend310 \                 # 模型转换时指定芯片版本。
     exit 0
    ```

    转换命令如下。

    **bash air2om.sh** ../models/XXX.air ../models/XXX

    **表 1**  参数说明

    <a name="table15982121511203"></a>
    <table><thead align="left"><tr id="row1598241522017"><th class="cellrowborder" valign="top" width="40%" id="mcps1.2.3.1.1"><p id="p189821115192014"><a name="p189821115192014"></a><a name="p189821115192014"></a>参数</p>
    </th>
    <th class="cellrowborder" valign="top" width="60%" id="mcps1.2.3.1.2"><p id="p1982161512206"><a name="p1982161512206"></a><a name="p1982161512206"></a>说明</p>
    </th>
    </tr>
    </thead>
    <tbody><tr id="row0982101592015"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p1598231542020"><a name="p1598231542020"></a><a name="p1598231542020"></a>model_path</p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p598231511200"><a name="p598231511200"></a><a name="p598231511200"></a>AIR文件路径。</p>
    </td>
    </tr>
    <tr id="row109831315132011"><td class="cellrowborder" valign="top" width="40%" headers="mcps1.2.3.1.1 "><p id="p598319158204"><a name="p598319158204"></a><a name="p598319158204"></a>output_model_name</p>
    </td>
    <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.3.1.2 "><p id="p1898316155207"><a name="p1898316155207"></a><a name="p1898316155207"></a>生成的OM文件名，转换脚本会在此基础上添加.om后缀。</p>
    </td>
    </tr>
    </tbody>
    </table>

### mxBase推理

1. 修改配置

    配置项位于"infer/mxbase/main_opencv.cpp"中，根据训练时配置修改是否有segment与heatmap输入

    ```cpp
    InitParam initParam = {};
    initParam.deviceId = 0;
    initParam.PoseEstNetPath = "../models/PoseEstNet.om";
    initParam.MultiTaskNetPath = "../models/MultiTask.om";
    initParam.resultPath = argv[2];
    initParam.segmentAware = true;
    initParam.heatmapAware = false;
    initParam.batchSize = 1;
    ```

2. "infer/mxbase"目录下编译工程。

    ```
    bash build.sh
    ```

3. 在相同目录下运行推理服务。

    ```python
    # ./pamtri arg0:待推理图片所在路径 arg1:结果存放文件路径
    ./pamtri ../data/ result.txt
    ```

4. 打开生成文件(路径为步骤3中的arg1)查看结果。

    ```python
    cat result.txt
    # img1 color:xx type:xx
    # img2 ...
    # ...
    ```

### MindX SDK推理

1. 修改配置文件。

    打开"infer/sdk/pipkine/pamtri.pipline" 修改pipeline配置(如om模型路径)。
    将modelPath设为自己的om模型路径

    ```
    "modelPath": "../models/PoseEstNet.om"
    ...
    "modelPath": "../models/MultiTaskNet.om"
    ```

2. 运行推理服务。

    1. 若要观测推理性能，需要打开性能统计开关。如下将“enable_ps”参数设置为true，“ps_interval_time”参数设置为6。

        vim /usr/local/sdk_home/mxManufacture/config/sdk.conf

        ```
        # MindX SDK configuration file

        # whether to enable performance statistics, default is false [dynamic config]
        enable_ps=true
        ...
        ps_interval_time=6
        ...
        ```

    2. 进入infer/sdk路径下并执行推理。

        ```python
        # python3.7 main.py arg0:图片路径 arg1:结果文件路径 arg2:pipline路径
        python3.7 main.py --img_path=XXX --result_path=result.txt --pipline_path=XXX
        ```

    3. 查看推理结果。

        ```python
        cat result.txt
        # img1 color:xx type:xx
        # img2 ...
        # ...
        ```

    4. 查看推理性能。

        请确保性能开关已打开，在日志目录“/usr/local/sdk_home/mxManufacture/logs”查看性能统计结果。

        ```
        performance—statistics.log.e2e.xxx
        performance—statistics.log.plugin.xxx
        performance—statistics.log.tpr.xxx
        ```

3. 运行精度测试。

    1. 根据实际修改精度脚本的参数（数据、标签、pipline路径）

        eval_pn.py

        ```python
        parser = argparse.ArgumentParser(description='Eval PoseEstNet network')
        # default='../data/PoseEstNet/veri/images/image_test'
        parser.add_argument('--img_path', type=str, required=True)
        # default='../data/PoseEstNet/veri/annot/label_test.csv'
        parser.add_argument('--label_path', type=str, required=True)
        # default='../pipline/pamtri.pipline'
        parser.add_argument('--pipline_path', type=str, required=True)
        args = parser.parse_args()
        if __name__ == '__main__':
            eval_posenet(args.label_path, args.img_path, args.pipline_path)
        ```

        eval_mt.py：根据训练时配置修改是否有segment与heatmap输入

        ```python
        parser = argparse.ArgumentParser(description='Eval MultiTaskNet network')
        # default='../data/MultiTaskNet/veri'
        parser.add_argument('--img_path', type=str, required=True)
        # default='../data/MultiTaskNet/veri'
        parser.add_argument('--label_path', type=str, required=True)
        # default='../pipline/pamtri.pipline'
        parser.add_argument('--pipline_path', type=str, required=True)
        parser.add_argument('--segmentaware', type=ast.literal_eval, default=False)
        parser.add_argument('--heatmapaware', type=ast.literal_eval, default=True)
        args = parser.parse_args()
        if __name__ == '__main__':
            eval_multitasknet(args.img_path, args.label_path, args.pipline_path,
                            segmentaware=args.segmentaware, heatmapaware=args.heatmapaware)
        ```

    2. 运行精度测试

        ```
        python3.7 eval_pn.py --img_path=XXX --label_path=XXX --pipline_path=XXX
        python3.7 eval_mt.py --img_path=XXX --label_path=XXX --pipline_path=XXX
        ```

    3. 精度将输出在终端

