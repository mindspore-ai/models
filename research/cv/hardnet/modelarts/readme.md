# 在ModelArts上应用

## 创建OBS桶

1.登录[OBS管理控制台](https://storage.huaweicloud.com/obs)，创建OBS桶。具体请参见[创建桶](https://support.huaweicloud.com/usermanual-obs/obs_03_0306.html)章节。

创建桶的区域需要与ModelArts所在的区域一致。例如：当前ModelArts在华北-北京四区域，在对象存储服务创建桶时，请选择华北-北京四。

创建用于存放数据的文件夹，具体请参见[新建文件夹](https://support.huaweicloud.com/usermanual-obs/obs_03_0316.html)章节。例如，在已创建的OBS桶中创建code、data、log、output目录。

目录结构说明：

- code：存放训练脚本目录
- data：存放训练数据集目录
- log：存放训练日志目录
- output：存放训练ckpt文件（output中result文件夹中）

数据集imagenet传至“data”目录。

### 创建算法

1.使用华为云帐号登录[ModelArts管理控制台](https://console.huaweicloud.com/modelarts)，在左侧导航栏中选择“算法管理”。
2.在“我的算法管理”界面，单击左上角“创建”，进入“创建算法”页面。
3.在“创建算法”页面，填写相关参数，然后单击“提交”。
    1.设置算法基本信息。
    2.设置“创建方式”为“自定义脚本”。
        用户需根据实际算法代码情况设置“AI引擎”、“代码目录”和“启动文件”。选择的AI引擎和编写算法代码时选择的框架必须一致。例如编写算法代码使用的是MindSpore，则在创建算法时也要选择MindSpore。
        _示例：_
        **表 1** _参数说明_
        <a name="table09972489125"></a>
        <table><thead align="left"><tr id="row139978484125"><th class="cellrowborder" valign="top" width="29.470000000000002%" id="mcps1.2.3.1.1"><p id="p16997114831219"><a name="p16997114831219"></a><a name="p16997114831219"></a><em id="i1199720484127"><a name="i1199720484127"></a><a name="i1199720484127"></a>参数名称</em></p>
        </th>
        <th class="cellrowborder" valign="top" width="70.53%" id="mcps1.2.3.1.2"><p id="p199976489122"><a name="p199976489122"></a><a name="p199976489122"></a><em id="i9997154816124"><a name="i9997154816124"></a><a name="i9997154816124"></a>说明</em></p>
        </th>
        </tr>
        </thead>
        <tbody><tr id="row11997124871210"><td class="cellrowborder" valign="top" width="29.470000000000002%" headers="mcps1.2.3.1.1 "><p id="p1299734820121"><a name="p1299734820121"></a><a name="p1299734820121"></a><em id="i199764819121"><a name="i199764819121"></a><a name="i199764819121"></a>AI引擎</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="70.53%" headers="mcps1.2.3.1.2 "><p id="p1899720481122"><a name="p1899720481122"></a><a name="p1899720481122"></a><em id="i9997848191217"><a name="i9997848191217"></a><a name="i9997848191217"></a>Ascend-Powered-Engine，mindspore_1.3.0-cann_5.0.2</em></p>
        </td>
        </tr>
        <tr id="row5997348121218"><td class="cellrowborder" valign="top" width="29.470000000000002%" headers="mcps1.2.3.1.1 "><p id="p139971748141218"><a name="p139971748141218"></a><a name="p139971748141218"></a><em id="i1199784811220"><a name="i1199784811220"></a><a name="i1199784811220"></a>代码目录</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="70.53%" headers="mcps1.2.3.1.2 "><p id="p2099724810127"><a name="p2099724810127"></a><a name="p2099724810127"></a><em id="i17997144871212"><a name="i17997144871212"></a><a name="i17997144871212"></a>算法代码存储的OBS路径。上传训练脚本，如：/obs桶/chengyj/code/hardnet/</em></p>
        </td>
        </tr>
        <tr id="row899794811124"><td class="cellrowborder" valign="top" width="29.470000000000002%" headers="mcps1.2.3.1.1 "><p id="p799714482129"><a name="p799714482129"></a><a name="p799714482129"></a><em id="i399704871210"><a name="i399704871210"></a><a name="i399704871210"></a>启动文件</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="70.53%" headers="mcps1.2.3.1.2 "><p id="p13997154831215"><a name="p13997154831215"></a><a name="p13997154831215"></a><em id="i11997648161214"><a name="i11997648161214"></a><a name="i11997648161214"></a>启动文件：启动训练的python脚本，如：/obs桶/chengyj/hardnet/code/modelarts/train_start.py</em></p>
        </div>
        </td>
        </tr>
        <tr id="row59981448101210"><td class="cellrowborder" valign="top" width="29.470000000000002%" headers="mcps1.2.3.1.1 "><p id="p19998124812123"><a name="p19998124812123"></a><a name="p19998124812123"></a><em id="i1399864831211"><a name="i1399864831211"></a><a name="i1399864831211"></a>输入数据配置</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="70.53%" headers="mcps1.2.3.1.2 "><p id="p139982484129"><a name="p139982484129"></a><a name="p139982484129"></a><em id="i299816484122"><a name="i299816484122"></a><a name="i299816484122"></a>代码路径参数：data_dir</em></p>
        </td>
        </tr>
        <tr id="row179981948151214"><td class="cellrowborder" valign="top" width="29.470000000000002%" headers="mcps1.2.3.1.1 "><p id="p89981948191220"><a name="p89981948191220"></a><a name="p89981948191220"></a><em id="i599844831217"><a name="i599844831217"></a><a name="i599844831217"></a>输出数据配置</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="70.53%" headers="mcps1.2.3.1.2 "><p id="p599814485120"><a name="p599814485120"></a><a name="p599814485120"></a><em id="i189981748171218"><a name="i189981748171218"></a><a name="i189981748171218"></a>代码路径参数：train_url</em></p>
        </td>
        </tr>
        </tbody>
        </table>
    3.填写超参数。
        单击“添加超参”，手动添加超参。配置代码中的命令行参数值，请根据您编写的算法代码逻辑进行填写，确保参数名称和代码的参数名称保持一致，可填写多个参数。
        _示例：_
        **表 2** _超参说明_
        <a name="table29981482127"></a>
        <table><thead align="left"><tr id="row1599894881216"><th class="cellrowborder" valign="top" width="25%" id="mcps1.2.6.1.1"><p id="p89988484121"><a name="p89988484121"></a><a name="p89988484121"></a><em id="i89985485123"><a name="i89985485123"></a><a name="i89985485123"></a>参数名称</em></p>
        </th>
        <th class="cellrowborder" valign="top" width="15%" id="mcps1.2.6.1.2"><p id="p1999114814121"><a name="p1999114814121"></a><a name="p1999114814121"></a><em id="i7999448181212"><a name="i7999448181212"></a><a name="i7999448181212"></a>类型</em></p>
        </th>
        <th class="cellrowborder" valign="top" width="17%" id="mcps1.2.6.1.3"><p id="p6999124810126"><a name="p6999124810126"></a><a name="p6999124810126"></a><em id="i17999144818126"><a name="i17999144818126"></a><a name="i17999144818126"></a>默认值</em></p>
        </th>
        <th class="cellrowborder" valign="top" width="18%" id="mcps1.2.6.1.4"><p id="p69992486123"><a name="p69992486123"></a><a name="p69992486123"></a><em id="i1599916488127"><a name="i1599916488127"></a><a name="i1599916488127"></a>是否必填</em></p>
        </th>
        <th class="cellrowborder" valign="top" width="25%" id="mcps1.2.6.1.5"><p id="p1999248121214"><a name="p1999248121214"></a><a name="p1999248121214"></a><em id="i299915481121"><a name="i299915481121"></a><a name="i299915481121"></a>描述</em></p>
        </th>
        </tr>
        </thead>
        <tbody><tr id="row9999134818128"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.1 "><p id="p14999124811212"><a name="p14999124811212"></a><a name="p14999124811212"></a><em id="i39991748101218"><a name="i39991748101218"></a><a name="i39991748101218"></a>pre_trained</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.6.1.2 "><p id="p599924815129"><a name="p599924815129"></a><a name="p599924815129"></a><em id="i8999184811212"><a name="i8999184811212"></a><a name="i8999184811212"></a>bollean</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="17%" headers="mcps1.2.6.1.3 "><p id="p179992484129"><a name="p179992484129"></a><a name="p179992484129"></a><em id="i1799913488128"><a name="i1799913488128"></a><a name="i1799913488128"></a>True</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="18%" headers="mcps1.2.6.1.4 "><p id="p179991348181213"><a name="p179991348181213"></a><a name="p179991348181213"></a><em id="i20999134812126"><a name="i20999134812126"></a><a name="i20999134812126"></a>是</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.5 "><p id="p899916487125"><a name="p899916487125"></a><a name="p899916487125"></a><em id="i99999482127"><a name="i99999482127"></a><a name="i99999482127"></a>是否需要预训练模型</em></p>
        </td>
        </tr>
        <tr id="row14999148161210"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.1 "><p id="p199915488129"><a name="p199915488129"></a><a name="p199915488129"></a><em id="i11999448141216"><a name="i11999448141216"></a><a name="i11999448141216"></a>pre_ckpt_path</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.6.1.2 "><p id="p7999124813124"><a name="p7999124813124"></a><a name="p7999124813124"></a><em id="i7999748151214"><a name="i7999748151214"></a><a name="i7999748151214"></a>string</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="17%" headers="mcps1.2.6.1.3 "><p id="p902049121213"><a name="p902049121213"></a><a name="p902049121213"></a><em id="i100124914123"><a name="i100124914123"></a><a name="i100124914123"></a>-</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="18%" headers="mcps1.2.6.1.4 "><p id="p19004917125"><a name="p19004917125"></a><a name="p19004917125"></a><em id="i208494126"><a name="i208494126"></a><a name="i208494126"></a>是</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.5 "><p id="p10134915129"><a name="p10134915129"></a><a name="p10134915129"></a><em id="i101949121214"><a name="i101949121214"></a><a name="i101949121214"></a>预训练模型的ckpt文件路径</em></p>
        </td>
        </tr>
        <tr id="row100124911121"><td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.1 "><p id="p150849131211"><a name="p150849131211"></a><a name="p150849131211"></a><em id="i1101549151218"><a name="i1101549151218"></a><a name="i1101549151218"></a>ckpt_file</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="15%" headers="mcps1.2.6.1.2 "><p id="p19054914124"><a name="p19054914124"></a><a name="p19054914124"></a><em id="i10144919126"><a name="i10144919126"></a><a name="i10144919126"></a>string</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="17%" headers="mcps1.2.6.1.3 "><p id="p6011490123"><a name="p6011490123"></a><a name="p6011490123"></a><em id="i00144917122"><a name="i00144917122"></a><a name="i00144917122"></a>-</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="18%" headers="mcps1.2.6.1.4 "><p id="p301449191215"><a name="p301449191215"></a><a name="p301449191215"></a><em id="i180104910126"><a name="i180104910126"></a><a name="i180104910126"></a>是</em></p>
        </td>
        <td class="cellrowborder" valign="top" width="25%" headers="mcps1.2.6.1.5 "><p id="p1702495127"><a name="p1702495127"></a><a name="p1702495127"></a><em id="i170249181214"><a name="i170249181214"></a><a name="i170249181214"></a>ckpt保存路径。</em></p>
        </td>
        </tr>
        </tbody>
        </table>

### 创建训练作业

1.使用华为云帐号登录[ModelArts管理控制台](https://console.huaweicloud.com/modelarts)，在左侧导航栏中选择“训练管理 \> 训练作业（New）”，默认进入“训练作业”列表。
2.单击“创建训练作业”，进入“创建训练作业”页面，在该页面填写训练作业相关参数。
    1.填写基本信息。
        基本信息包含“名称”和“描述”。
    2.填写作业参数。
        包含数据来源、算法来源等关键信息。本步骤只提供训练任务部分参数配置说明，其他参数配置详情请参见[《ModelArts AI 工程师用户指南](https://support.huaweicloud.com/modelarts/index.html)》中“训练模型（new）”。
        **表 1**  参数说明
        <a name="table96111035134613"></a>
        <table><thead align="left"><tr id="zh-cn_topic_0000001178072725_row1727593212228"><th class="cellrowborder" valign="top" width="20%" id="mcps1.2.4.1.1"><p id="zh-cn_topic_0000001178072725_p102751332172212"><a name="zh-cn_topic_0000001178072725_p102751332172212"></a><a name="zh-cn_topic_0000001178072725_p102751332172212"></a>参数名称</p>
        </th>
        <th class="cellrowborder" valign="top" width="20%" id="mcps1.2.4.1.2"><p id="zh-cn_topic_0000001178072725_p186943411156"><a name="zh-cn_topic_0000001178072725_p186943411156"></a><a name="zh-cn_topic_0000001178072725_p186943411156"></a>子参数</p>
        </th>
        <th class="cellrowborder" valign="top" width="60%" id="mcps1.2.4.1.3"><p id="zh-cn_topic_0000001178072725_p1827543282216"><a name="zh-cn_topic_0000001178072725_p1827543282216"></a><a name="zh-cn_topic_0000001178072725_p1827543282216"></a>说明</p>
        </th>
        </tr>
        </thead>
        <tbody><tr id="zh-cn_topic_0000001178072725_row780219161358"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001178072725_p0803121617510"><a name="zh-cn_topic_0000001178072725_p0803121617510"></a><a name="zh-cn_topic_0000001178072725_p0803121617510"></a>算法</p>
        </td>
        <td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001178072725_p186947411520"><a name="zh-cn_topic_0000001178072725_p186947411520"></a><a name="zh-cn_topic_0000001178072725_p186947411520"></a>我的算法</p>
        </td>
        <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001178072725_p20803141614514"><a name="zh-cn_topic_0000001178072725_p20803141614514"></a><a name="zh-cn_topic_0000001178072725_p20803141614514"></a>选择“我的算法”页签，勾选上文中创建的算法。</p>
        <p id="zh-cn_topic_0000001178072725_p24290418284"><a name="zh-cn_topic_0000001178072725_p24290418284"></a><a name="zh-cn_topic_0000001178072725_p24290418284"></a>如果没有创建算法，请单击“创建”进入创建算法页面，详细操作指导参见“创建算法”。</p>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000001178072725_row1927503211228"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001178072725_p327583216224"><a name="zh-cn_topic_0000001178072725_p327583216224"></a><a name="zh-cn_topic_0000001178072725_p327583216224"></a>训练输入</p>
        </td>
        <td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001178072725_p1069419416510"><a name="zh-cn_topic_0000001178072725_p1069419416510"></a><a name="zh-cn_topic_0000001178072725_p1069419416510"></a>数据来源</p>
        </td>
        <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001178072725_p142750323227"><a name="zh-cn_topic_0000001178072725_p142750323227"></a><a name="zh-cn_topic_0000001178072725_p142750323227"></a>选择OBS上数据集存放的目录。</p>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000001178072725_row127593211227"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001178072725_p9744151562"><a name="zh-cn_topic_0000001178072725_p9744151562"></a><a name="zh-cn_topic_0000001178072725_p9744151562"></a>训练输出</p>
        </td>
        <td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001178072725_p1027563212210"><a name="zh-cn_topic_0000001178072725_p1027563212210"></a><a name="zh-cn_topic_0000001178072725_p1027563212210"></a>模型输出</p>
        </td>
        <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001178072725_p13275113252214"><a name="zh-cn_topic_0000001178072725_p13275113252214"></a><a name="zh-cn_topic_0000001178072725_p13275113252214"></a>选择训练结果的存储位置（OBS路径），请尽量选择空目录来作为训练输出路径。</p>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000001178072725_row18750142834916"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001178072725_p5751172811492"><a name="zh-cn_topic_0000001178072725_p5751172811492"></a><a name="zh-cn_topic_0000001178072725_p5751172811492"></a>规格</p>
        </td>
        <td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001178072725_p107514288495"><a name="zh-cn_topic_0000001178072725_p107514288495"></a><a name="zh-cn_topic_0000001178072725_p107514288495"></a>-</p>
        </td>
        <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001178072725_p3751142811495"><a name="zh-cn_topic_0000001178072725_p3751142811495"></a><a name="zh-cn_topic_0000001178072725_p3751142811495"></a>Ascend: 1*Ascend 910(32GB) | ARM: 24 核 96GB</p>
        </td>
        </tr>
        <tr id="zh-cn_topic_0000001178072725_row16275103282219"><td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.1 "><p id="zh-cn_topic_0000001178072725_p15275132192213"><a name="zh-cn_topic_0000001178072725_p15275132192213"></a><a name="zh-cn_topic_0000001178072725_p15275132192213"></a>作业日志路径</p>
        </td>
        <td class="cellrowborder" valign="top" width="20%" headers="mcps1.2.4.1.2 "><p id="zh-cn_topic_0000001178072725_p1369484117516"><a name="zh-cn_topic_0000001178072725_p1369484117516"></a><a name="zh-cn_topic_0000001178072725_p1369484117516"></a>-</p>
        </td>
        <td class="cellrowborder" valign="top" width="60%" headers="mcps1.2.4.1.3 "><p id="zh-cn_topic_0000001178072725_p227563218228"><a name="zh-cn_topic_0000001178072725_p227563218228"></a><a name="zh-cn_topic_0000001178072725_p227563218228"></a>设置训练日志存放的目录。请注意选择的OBS目录有读写权限。</p>
        </td>
        </tr>
        </tbody>
        </table>

>![输入图片说明](https://images.gitee.com/uploads/images/2021/0926/181247_df155dc2_8725359.gif "icon-note.gif")**说明：**
></span><div class="notebody"><p id="p118851046192714"><a name="p118851046192714"></a><a name="p118851046192714"></a>超参：创建训练作业时可不传入非必填的超参，需用户手动删除。 <strong id="b1720214810184">

3.单击“提交”，完成训练作业的创建。
    训练作业一般需要运行一段时间，根据您选择的数据量和资源不同，训练时间将耗时几分钟到几十分钟不等。

### 查看训练任务日志

1.在ModelArts管理控制台，在左侧导航栏中选择“训练管理 \> 训练作业（New）”，默认进入“训练作业”列表。
2.在训练作业列表中，您可以单击作业名称，查看该作业的详情。
    详情中包含作业的基本信息、训练参数、日志详情和资源占用情况。
