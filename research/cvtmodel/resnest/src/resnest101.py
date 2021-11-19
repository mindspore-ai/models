import mindspore.ops as P
from mindspore import nn


class Module5(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_padding,
                 conv2d_0_pad_mode, conv2d_0_group):
        super(Module5, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=conv2d_0_kernel_size,
                                  stride=(1, 1),
                                  padding=conv2d_0_padding,
                                  pad_mode=conv2d_0_pad_mode,
                                  dilation=(1, 1),
                                  group=conv2d_0_group,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        return opt_relu_1


class Module22(nn.Cell):
    def __init__(self, module5_0_conv2d_0_in_channels, module5_0_conv2d_0_out_channels, module5_0_conv2d_0_kernel_size,
                 module5_0_conv2d_0_padding, module5_0_conv2d_0_pad_mode, module5_0_conv2d_0_group,
                 module5_1_conv2d_0_in_channels, module5_1_conv2d_0_out_channels, module5_1_conv2d_0_kernel_size,
                 module5_1_conv2d_0_padding, module5_1_conv2d_0_pad_mode, module5_1_conv2d_0_group):
        super(Module22, self).__init__()
        self.module5_0 = Module5(conv2d_0_in_channels=module5_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module5_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module5_0_conv2d_0_kernel_size,
                                 conv2d_0_padding=module5_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module5_0_conv2d_0_pad_mode,
                                 conv2d_0_group=module5_0_conv2d_0_group)
        self.module5_1 = Module5(conv2d_0_in_channels=module5_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module5_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module5_1_conv2d_0_kernel_size,
                                 conv2d_0_padding=module5_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module5_1_conv2d_0_pad_mode,
                                 conv2d_0_group=module5_1_conv2d_0_group)

    def construct(self, x):
        module5_0_opt = self.module5_0(x)
        module5_1_opt = self.module5_1(module5_0_opt)
        return module5_1_opt


class Module0(nn.Cell):
    def __init__(self, avgpool2d_0_kernel_size, conv2d_1_in_channels, conv2d_1_out_channels, conv2d_3_in_channels,
                 conv2d_3_out_channels, reshape_4_shape):
        super(Module0, self).__init__()
        self.avgpool2d_0 = nn.AvgPool2d(kernel_size=avgpool2d_0_kernel_size)
        self.conv2d_1 = nn.Conv2d(in_channels=conv2d_1_in_channels,
                                  out_channels=conv2d_1_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_2 = nn.ReLU()
        self.conv2d_3 = nn.Conv2d(in_channels=conv2d_3_in_channels,
                                  out_channels=conv2d_3_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.reshape_4 = P.Reshape()
        self.reshape_4_shape = tuple(reshape_4_shape)
        self.transpose_5 = P.Transpose()

    def construct(self, x):
        opt_avgpool2d_0 = self.avgpool2d_0(x)
        opt_conv2d_1 = self.conv2d_1(opt_avgpool2d_0)
        opt_relu_2 = self.relu_2(opt_conv2d_1)
        opt_conv2d_3 = self.conv2d_3(opt_relu_2)
        opt_reshape_4 = self.reshape_4(opt_conv2d_3, self.reshape_4_shape)
        opt_transpose_5 = self.transpose_5(opt_reshape_4, (0, 3, 1, 2))
        return opt_transpose_5


class Module20(nn.Cell):
    def __init__(self, reshape_2_shape, module0_0_avgpool2d_0_kernel_size, module0_0_conv2d_1_in_channels,
                 module0_0_conv2d_1_out_channels, module0_0_conv2d_3_in_channels, module0_0_conv2d_3_out_channels,
                 module0_0_reshape_4_shape):
        super(Module20, self).__init__()
        self.module0_0 = Module0(avgpool2d_0_kernel_size=module0_0_avgpool2d_0_kernel_size,
                                 conv2d_1_in_channels=module0_0_conv2d_1_in_channels,
                                 conv2d_1_out_channels=module0_0_conv2d_1_out_channels,
                                 conv2d_3_in_channels=module0_0_conv2d_3_in_channels,
                                 conv2d_3_out_channels=module0_0_conv2d_3_out_channels,
                                 reshape_4_shape=module0_0_reshape_4_shape)
        self.softmax_0 = nn.Softmax(axis=3)
        self.transpose_1 = P.Transpose()
        self.reshape_2 = P.Reshape()
        self.reshape_2_shape = tuple(reshape_2_shape)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_softmax_0 = self.softmax_0(module0_0_opt)
        opt_transpose_1 = self.transpose_1(opt_softmax_0, (0, 3, 2, 1))
        opt_reshape_2 = self.reshape_2(opt_transpose_1, self.reshape_2_shape)
        return opt_reshape_2


class MainModel(nn.Cell):
    def __init__(self):
        super(MainModel, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=3,
                                  out_channels=64,
                                  kernel_size=(3, 3),
                                  stride=(2, 2),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.module22_0 = Module22(module5_0_conv2d_0_in_channels=64,
                                   module5_0_conv2d_0_out_channels=64,
                                   module5_0_conv2d_0_kernel_size=(3, 3),
                                   module5_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_0_conv2d_0_pad_mode="pad",
                                   module5_0_conv2d_0_group=1,
                                   module5_1_conv2d_0_in_channels=64,
                                   module5_1_conv2d_0_out_channels=128,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad",
                                   module5_1_conv2d_0_group=1)
        self.pad_maxpool2d_6 = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (1, 0)))
        self.maxpool2d_6 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.module22_1 = Module22(module5_0_conv2d_0_in_channels=128,
                                   module5_0_conv2d_0_out_channels=64,
                                   module5_0_conv2d_0_kernel_size=(1, 1),
                                   module5_0_conv2d_0_padding=0,
                                   module5_0_conv2d_0_pad_mode="valid",
                                   module5_0_conv2d_0_group=1,
                                   module5_1_conv2d_0_in_channels=64,
                                   module5_1_conv2d_0_out_channels=128,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad",
                                   module5_1_conv2d_0_group=2)
        self.split_13 = P.Split(axis=1, output_num=2)
        self.add_14_bias = 0.0
        self.module20_0 = Module20(reshape_2_shape=[1, 128],
                                   module0_0_avgpool2d_0_kernel_size=(64, 64),
                                   module0_0_conv2d_1_in_channels=64,
                                   module0_0_conv2d_1_out_channels=32,
                                   module0_0_conv2d_3_in_channels=32,
                                   module0_0_conv2d_3_out_channels=128,
                                   module0_0_reshape_4_shape=[1, 1, 2, 64])
        self.reshape_25 = P.Reshape()
        self.reshape_25_shape = tuple([1, 128, 1, 1])
        self.split_26 = P.Split(axis=1, output_num=2)
        self.add_29_bias = 0.0
        self.conv2d_31 = nn.Conv2d(in_channels=64,
                                   out_channels=256,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.pad_avgpool2d_8 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_8 = nn.AvgPool2d(kernel_size=(1, 1), stride=(1, 1))
        self.conv2d_10 = nn.Conv2d(in_channels=128,
                                   out_channels=256,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_33 = nn.ReLU()
        self.module22_2 = Module22(module5_0_conv2d_0_in_channels=256,
                                   module5_0_conv2d_0_out_channels=64,
                                   module5_0_conv2d_0_kernel_size=(1, 1),
                                   module5_0_conv2d_0_padding=0,
                                   module5_0_conv2d_0_pad_mode="valid",
                                   module5_0_conv2d_0_group=1,
                                   module5_1_conv2d_0_in_channels=64,
                                   module5_1_conv2d_0_out_channels=128,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad",
                                   module5_1_conv2d_0_group=2)
        self.split_38 = P.Split(axis=1, output_num=2)
        self.add_39_bias = 0.0
        self.module20_1 = Module20(reshape_2_shape=[1, 128],
                                   module0_0_avgpool2d_0_kernel_size=(64, 64),
                                   module0_0_conv2d_1_in_channels=64,
                                   module0_0_conv2d_1_out_channels=32,
                                   module0_0_conv2d_3_in_channels=32,
                                   module0_0_conv2d_3_out_channels=128,
                                   module0_0_reshape_4_shape=[1, 1, 2, 64])
        self.reshape_50 = P.Reshape()
        self.reshape_50_shape = tuple([1, 128, 1, 1])
        self.split_51 = P.Split(axis=1, output_num=2)
        self.add_54_bias = 0.0
        self.conv2d_56 = nn.Conv2d(in_channels=64,
                                   out_channels=256,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_58 = nn.ReLU()
        self.module22_3 = Module22(module5_0_conv2d_0_in_channels=256,
                                   module5_0_conv2d_0_out_channels=64,
                                   module5_0_conv2d_0_kernel_size=(1, 1),
                                   module5_0_conv2d_0_padding=0,
                                   module5_0_conv2d_0_pad_mode="valid",
                                   module5_0_conv2d_0_group=1,
                                   module5_1_conv2d_0_in_channels=64,
                                   module5_1_conv2d_0_out_channels=128,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad",
                                   module5_1_conv2d_0_group=2)
        self.split_63 = P.Split(axis=1, output_num=2)
        self.add_64_bias = 0.0
        self.module20_2 = Module20(reshape_2_shape=[1, 128],
                                   module0_0_avgpool2d_0_kernel_size=(64, 64),
                                   module0_0_conv2d_1_in_channels=64,
                                   module0_0_conv2d_1_out_channels=32,
                                   module0_0_conv2d_3_in_channels=32,
                                   module0_0_conv2d_3_out_channels=128,
                                   module0_0_reshape_4_shape=[1, 1, 2, 64])
        self.reshape_75 = P.Reshape()
        self.reshape_75_shape = tuple([1, 128, 1, 1])
        self.split_76 = P.Split(axis=1, output_num=2)
        self.add_79_bias = 0.0
        self.conv2d_81 = nn.Conv2d(in_channels=64,
                                   out_channels=256,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_83 = nn.ReLU()
        self.module22_4 = Module22(module5_0_conv2d_0_in_channels=256,
                                   module5_0_conv2d_0_out_channels=128,
                                   module5_0_conv2d_0_kernel_size=(1, 1),
                                   module5_0_conv2d_0_padding=0,
                                   module5_0_conv2d_0_pad_mode="valid",
                                   module5_0_conv2d_0_group=1,
                                   module5_1_conv2d_0_in_channels=128,
                                   module5_1_conv2d_0_out_channels=256,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad",
                                   module5_1_conv2d_0_group=2)
        self.split_90 = P.Split(axis=1, output_num=2)
        self.add_91_bias = 0.0
        self.module20_3 = Module20(reshape_2_shape=[1, 256],
                                   module0_0_avgpool2d_0_kernel_size=(64, 64),
                                   module0_0_conv2d_1_in_channels=128,
                                   module0_0_conv2d_1_out_channels=64,
                                   module0_0_conv2d_3_in_channels=64,
                                   module0_0_conv2d_3_out_channels=256,
                                   module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_102 = P.Reshape()
        self.reshape_102_shape = tuple([1, 256, 1, 1])
        self.split_103 = P.Split(axis=1, output_num=2)
        self.add_106_bias = 0.0
        self.pad_108 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.pad_avgpool2d_109 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_109 = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2d_110 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_85 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_85 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2d_87 = nn.Conv2d(in_channels=256,
                                   out_channels=512,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_112 = nn.ReLU()
        self.module22_5 = Module22(module5_0_conv2d_0_in_channels=512,
                                   module5_0_conv2d_0_out_channels=128,
                                   module5_0_conv2d_0_kernel_size=(1, 1),
                                   module5_0_conv2d_0_padding=0,
                                   module5_0_conv2d_0_pad_mode="valid",
                                   module5_0_conv2d_0_group=1,
                                   module5_1_conv2d_0_in_channels=128,
                                   module5_1_conv2d_0_out_channels=256,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad",
                                   module5_1_conv2d_0_group=2)
        self.split_117 = P.Split(axis=1, output_num=2)
        self.add_118_bias = 0.0
        self.module20_4 = Module20(reshape_2_shape=[1, 256],
                                   module0_0_avgpool2d_0_kernel_size=(32, 32),
                                   module0_0_conv2d_1_in_channels=128,
                                   module0_0_conv2d_1_out_channels=64,
                                   module0_0_conv2d_3_in_channels=64,
                                   module0_0_conv2d_3_out_channels=256,
                                   module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_129 = P.Reshape()
        self.reshape_129_shape = tuple([1, 256, 1, 1])
        self.split_130 = P.Split(axis=1, output_num=2)
        self.add_133_bias = 0.0
        self.conv2d_135 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_137 = nn.ReLU()
        self.module22_6 = Module22(module5_0_conv2d_0_in_channels=512,
                                   module5_0_conv2d_0_out_channels=128,
                                   module5_0_conv2d_0_kernel_size=(1, 1),
                                   module5_0_conv2d_0_padding=0,
                                   module5_0_conv2d_0_pad_mode="valid",
                                   module5_0_conv2d_0_group=1,
                                   module5_1_conv2d_0_in_channels=128,
                                   module5_1_conv2d_0_out_channels=256,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad",
                                   module5_1_conv2d_0_group=2)
        self.split_142 = P.Split(axis=1, output_num=2)
        self.add_143_bias = 0.0
        self.module20_5 = Module20(reshape_2_shape=[1, 256],
                                   module0_0_avgpool2d_0_kernel_size=(32, 32),
                                   module0_0_conv2d_1_in_channels=128,
                                   module0_0_conv2d_1_out_channels=64,
                                   module0_0_conv2d_3_in_channels=64,
                                   module0_0_conv2d_3_out_channels=256,
                                   module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_154 = P.Reshape()
        self.reshape_154_shape = tuple([1, 256, 1, 1])
        self.split_155 = P.Split(axis=1, output_num=2)
        self.add_158_bias = 0.0
        self.conv2d_160 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_162 = nn.ReLU()
        self.module22_7 = Module22(module5_0_conv2d_0_in_channels=512,
                                   module5_0_conv2d_0_out_channels=128,
                                   module5_0_conv2d_0_kernel_size=(1, 1),
                                   module5_0_conv2d_0_padding=0,
                                   module5_0_conv2d_0_pad_mode="valid",
                                   module5_0_conv2d_0_group=1,
                                   module5_1_conv2d_0_in_channels=128,
                                   module5_1_conv2d_0_out_channels=256,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad",
                                   module5_1_conv2d_0_group=2)
        self.split_167 = P.Split(axis=1, output_num=2)
        self.add_168_bias = 0.0
        self.module20_6 = Module20(reshape_2_shape=[1, 256],
                                   module0_0_avgpool2d_0_kernel_size=(32, 32),
                                   module0_0_conv2d_1_in_channels=128,
                                   module0_0_conv2d_1_out_channels=64,
                                   module0_0_conv2d_3_in_channels=64,
                                   module0_0_conv2d_3_out_channels=256,
                                   module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_179 = P.Reshape()
        self.reshape_179_shape = tuple([1, 256, 1, 1])
        self.split_180 = P.Split(axis=1, output_num=2)
        self.add_183_bias = 0.0
        self.conv2d_185 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_187 = nn.ReLU()
        self.module22_8 = Module22(module5_0_conv2d_0_in_channels=512,
                                   module5_0_conv2d_0_out_channels=256,
                                   module5_0_conv2d_0_kernel_size=(1, 1),
                                   module5_0_conv2d_0_padding=0,
                                   module5_0_conv2d_0_pad_mode="valid",
                                   module5_0_conv2d_0_group=1,
                                   module5_1_conv2d_0_in_channels=256,
                                   module5_1_conv2d_0_out_channels=512,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad",
                                   module5_1_conv2d_0_group=2)
        self.split_194 = P.Split(axis=1, output_num=2)
        self.add_195_bias = 0.0
        self.module20_7 = Module20(reshape_2_shape=[1, 512],
                                   module0_0_avgpool2d_0_kernel_size=(32, 32),
                                   module0_0_conv2d_1_in_channels=256,
                                   module0_0_conv2d_1_out_channels=128,
                                   module0_0_conv2d_3_in_channels=128,
                                   module0_0_conv2d_3_out_channels=512,
                                   module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_206 = P.Reshape()
        self.reshape_206_shape = tuple([1, 512, 1, 1])
        self.split_207 = P.Split(axis=1, output_num=2)
        self.add_210_bias = 0.0
        self.pad_212 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.pad_avgpool2d_213 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_213 = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2d_214 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_189 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_189 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2d_191 = nn.Conv2d(in_channels=512,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_216 = nn.ReLU()
        self.module22_9 = Module22(module5_0_conv2d_0_in_channels=1024,
                                   module5_0_conv2d_0_out_channels=256,
                                   module5_0_conv2d_0_kernel_size=(1, 1),
                                   module5_0_conv2d_0_padding=0,
                                   module5_0_conv2d_0_pad_mode="valid",
                                   module5_0_conv2d_0_group=1,
                                   module5_1_conv2d_0_in_channels=256,
                                   module5_1_conv2d_0_out_channels=512,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad",
                                   module5_1_conv2d_0_group=2)
        self.split_221 = P.Split(axis=1, output_num=2)
        self.add_222_bias = 0.0
        self.module20_8 = Module20(reshape_2_shape=[1, 512],
                                   module0_0_avgpool2d_0_kernel_size=(16, 16),
                                   module0_0_conv2d_1_in_channels=256,
                                   module0_0_conv2d_1_out_channels=128,
                                   module0_0_conv2d_3_in_channels=128,
                                   module0_0_conv2d_3_out_channels=512,
                                   module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_233 = P.Reshape()
        self.reshape_233_shape = tuple([1, 512, 1, 1])
        self.split_234 = P.Split(axis=1, output_num=2)
        self.add_237_bias = 0.0
        self.conv2d_239 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_241 = nn.ReLU()
        self.module22_10 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_246 = P.Split(axis=1, output_num=2)
        self.add_247_bias = 0.0
        self.module20_9 = Module20(reshape_2_shape=[1, 512],
                                   module0_0_avgpool2d_0_kernel_size=(16, 16),
                                   module0_0_conv2d_1_in_channels=256,
                                   module0_0_conv2d_1_out_channels=128,
                                   module0_0_conv2d_3_in_channels=128,
                                   module0_0_conv2d_3_out_channels=512,
                                   module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_258 = P.Reshape()
        self.reshape_258_shape = tuple([1, 512, 1, 1])
        self.split_259 = P.Split(axis=1, output_num=2)
        self.add_262_bias = 0.0
        self.conv2d_264 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_266 = nn.ReLU()
        self.module22_11 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_271 = P.Split(axis=1, output_num=2)
        self.add_272_bias = 0.0
        self.module20_10 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_283 = P.Reshape()
        self.reshape_283_shape = tuple([1, 512, 1, 1])
        self.split_284 = P.Split(axis=1, output_num=2)
        self.add_287_bias = 0.0
        self.conv2d_289 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_291 = nn.ReLU()
        self.module22_12 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_296 = P.Split(axis=1, output_num=2)
        self.add_297_bias = 0.0
        self.module20_11 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_308 = P.Reshape()
        self.reshape_308_shape = tuple([1, 512, 1, 1])
        self.split_309 = P.Split(axis=1, output_num=2)
        self.add_312_bias = 0.0
        self.conv2d_314 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_316 = nn.ReLU()
        self.module22_13 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_321 = P.Split(axis=1, output_num=2)
        self.add_322_bias = 0.0
        self.module20_12 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_333 = P.Reshape()
        self.reshape_333_shape = tuple([1, 512, 1, 1])
        self.split_334 = P.Split(axis=1, output_num=2)
        self.add_337_bias = 0.0
        self.conv2d_339 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_341 = nn.ReLU()
        self.module22_14 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_346 = P.Split(axis=1, output_num=2)
        self.add_347_bias = 0.0
        self.module20_13 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_358 = P.Reshape()
        self.reshape_358_shape = tuple([1, 512, 1, 1])
        self.split_359 = P.Split(axis=1, output_num=2)
        self.add_362_bias = 0.0
        self.conv2d_364 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_366 = nn.ReLU()
        self.module22_15 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_371 = P.Split(axis=1, output_num=2)
        self.add_372_bias = 0.0
        self.module20_14 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_383 = P.Reshape()
        self.reshape_383_shape = tuple([1, 512, 1, 1])
        self.split_384 = P.Split(axis=1, output_num=2)
        self.add_387_bias = 0.0
        self.conv2d_389 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_391 = nn.ReLU()
        self.module22_16 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_396 = P.Split(axis=1, output_num=2)
        self.add_397_bias = 0.0
        self.module20_15 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_408 = P.Reshape()
        self.reshape_408_shape = tuple([1, 512, 1, 1])
        self.split_409 = P.Split(axis=1, output_num=2)
        self.add_412_bias = 0.0
        self.conv2d_414 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_416 = nn.ReLU()
        self.module22_17 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_421 = P.Split(axis=1, output_num=2)
        self.add_422_bias = 0.0
        self.module20_16 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_433 = P.Reshape()
        self.reshape_433_shape = tuple([1, 512, 1, 1])
        self.split_434 = P.Split(axis=1, output_num=2)
        self.add_437_bias = 0.0
        self.conv2d_439 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_441 = nn.ReLU()
        self.module22_18 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_446 = P.Split(axis=1, output_num=2)
        self.add_447_bias = 0.0
        self.module20_17 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_458 = P.Reshape()
        self.reshape_458_shape = tuple([1, 512, 1, 1])
        self.split_459 = P.Split(axis=1, output_num=2)
        self.add_462_bias = 0.0
        self.conv2d_464 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_466 = nn.ReLU()
        self.module22_19 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_471 = P.Split(axis=1, output_num=2)
        self.add_472_bias = 0.0
        self.module20_18 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_483 = P.Reshape()
        self.reshape_483_shape = tuple([1, 512, 1, 1])
        self.split_484 = P.Split(axis=1, output_num=2)
        self.add_487_bias = 0.0
        self.conv2d_489 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_491 = nn.ReLU()
        self.module22_20 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_496 = P.Split(axis=1, output_num=2)
        self.add_497_bias = 0.0
        self.module20_19 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_508 = P.Reshape()
        self.reshape_508_shape = tuple([1, 512, 1, 1])
        self.split_509 = P.Split(axis=1, output_num=2)
        self.add_512_bias = 0.0
        self.conv2d_514 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_516 = nn.ReLU()
        self.module22_21 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_521 = P.Split(axis=1, output_num=2)
        self.add_522_bias = 0.0
        self.module20_20 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_533 = P.Reshape()
        self.reshape_533_shape = tuple([1, 512, 1, 1])
        self.split_534 = P.Split(axis=1, output_num=2)
        self.add_537_bias = 0.0
        self.conv2d_539 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_541 = nn.ReLU()
        self.module22_22 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_546 = P.Split(axis=1, output_num=2)
        self.add_547_bias = 0.0
        self.module20_21 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_558 = P.Reshape()
        self.reshape_558_shape = tuple([1, 512, 1, 1])
        self.split_559 = P.Split(axis=1, output_num=2)
        self.add_562_bias = 0.0
        self.conv2d_564 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_566 = nn.ReLU()
        self.module22_23 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_571 = P.Split(axis=1, output_num=2)
        self.add_572_bias = 0.0
        self.module20_22 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_583 = P.Reshape()
        self.reshape_583_shape = tuple([1, 512, 1, 1])
        self.split_584 = P.Split(axis=1, output_num=2)
        self.add_587_bias = 0.0
        self.conv2d_589 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_591 = nn.ReLU()
        self.module22_24 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_596 = P.Split(axis=1, output_num=2)
        self.add_597_bias = 0.0
        self.module20_23 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_608 = P.Reshape()
        self.reshape_608_shape = tuple([1, 512, 1, 1])
        self.split_609 = P.Split(axis=1, output_num=2)
        self.add_612_bias = 0.0
        self.conv2d_614 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_616 = nn.ReLU()
        self.module22_25 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_621 = P.Split(axis=1, output_num=2)
        self.add_622_bias = 0.0
        self.module20_24 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_633 = P.Reshape()
        self.reshape_633_shape = tuple([1, 512, 1, 1])
        self.split_634 = P.Split(axis=1, output_num=2)
        self.add_637_bias = 0.0
        self.conv2d_639 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_641 = nn.ReLU()
        self.module22_26 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_646 = P.Split(axis=1, output_num=2)
        self.add_647_bias = 0.0
        self.module20_25 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_658 = P.Reshape()
        self.reshape_658_shape = tuple([1, 512, 1, 1])
        self.split_659 = P.Split(axis=1, output_num=2)
        self.add_662_bias = 0.0
        self.conv2d_664 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_666 = nn.ReLU()
        self.module22_27 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_671 = P.Split(axis=1, output_num=2)
        self.add_672_bias = 0.0
        self.module20_26 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_683 = P.Reshape()
        self.reshape_683_shape = tuple([1, 512, 1, 1])
        self.split_684 = P.Split(axis=1, output_num=2)
        self.add_687_bias = 0.0
        self.conv2d_689 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_691 = nn.ReLU()
        self.module22_28 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_696 = P.Split(axis=1, output_num=2)
        self.add_697_bias = 0.0
        self.module20_27 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_708 = P.Reshape()
        self.reshape_708_shape = tuple([1, 512, 1, 1])
        self.split_709 = P.Split(axis=1, output_num=2)
        self.add_712_bias = 0.0
        self.conv2d_714 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_716 = nn.ReLU()
        self.module22_29 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_721 = P.Split(axis=1, output_num=2)
        self.add_722_bias = 0.0
        self.module20_28 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_733 = P.Reshape()
        self.reshape_733_shape = tuple([1, 512, 1, 1])
        self.split_734 = P.Split(axis=1, output_num=2)
        self.add_737_bias = 0.0
        self.conv2d_739 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_741 = nn.ReLU()
        self.module22_30 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=256,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=256,
                                    module5_1_conv2d_0_out_channels=512,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_746 = P.Split(axis=1, output_num=2)
        self.add_747_bias = 0.0
        self.module20_29 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_758 = P.Reshape()
        self.reshape_758_shape = tuple([1, 512, 1, 1])
        self.split_759 = P.Split(axis=1, output_num=2)
        self.add_762_bias = 0.0
        self.conv2d_764 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_766 = nn.ReLU()
        self.module22_31 = Module22(module5_0_conv2d_0_in_channels=1024,
                                    module5_0_conv2d_0_out_channels=512,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=512,
                                    module5_1_conv2d_0_out_channels=1024,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_773 = P.Split(axis=1, output_num=2)
        self.add_774_bias = 0.0
        self.module20_30 = Module20(reshape_2_shape=[1, 1024],
                                    module0_0_avgpool2d_0_kernel_size=(16, 16),
                                    module0_0_conv2d_1_in_channels=512,
                                    module0_0_conv2d_1_out_channels=256,
                                    module0_0_conv2d_3_in_channels=256,
                                    module0_0_conv2d_3_out_channels=1024,
                                    module0_0_reshape_4_shape=[1, 1, 2, 512])
        self.reshape_785 = P.Reshape()
        self.reshape_785_shape = tuple([1, 1024, 1, 1])
        self.split_786 = P.Split(axis=1, output_num=2)
        self.add_789_bias = 0.0
        self.pad_791 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.pad_avgpool2d_792 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_792 = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2d_793 = nn.Conv2d(in_channels=512,
                                    out_channels=2048,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_768 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_768 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2d_770 = nn.Conv2d(in_channels=1024,
                                    out_channels=2048,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_795 = nn.ReLU()
        self.module22_32 = Module22(module5_0_conv2d_0_in_channels=2048,
                                    module5_0_conv2d_0_out_channels=512,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=512,
                                    module5_1_conv2d_0_out_channels=1024,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_800 = P.Split(axis=1, output_num=2)
        self.add_801_bias = 0.0
        self.module20_31 = Module20(reshape_2_shape=[1, 1024],
                                    module0_0_avgpool2d_0_kernel_size=(8, 8),
                                    module0_0_conv2d_1_in_channels=512,
                                    module0_0_conv2d_1_out_channels=256,
                                    module0_0_conv2d_3_in_channels=256,
                                    module0_0_conv2d_3_out_channels=1024,
                                    module0_0_reshape_4_shape=[1, 1, 2, 512])
        self.reshape_812 = P.Reshape()
        self.reshape_812_shape = tuple([1, 1024, 1, 1])
        self.split_813 = P.Split(axis=1, output_num=2)
        self.add_816_bias = 0.0
        self.conv2d_818 = nn.Conv2d(in_channels=512,
                                    out_channels=2048,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_820 = nn.ReLU()
        self.module22_33 = Module22(module5_0_conv2d_0_in_channels=2048,
                                    module5_0_conv2d_0_out_channels=512,
                                    module5_0_conv2d_0_kernel_size=(1, 1),
                                    module5_0_conv2d_0_padding=0,
                                    module5_0_conv2d_0_pad_mode="valid",
                                    module5_0_conv2d_0_group=1,
                                    module5_1_conv2d_0_in_channels=512,
                                    module5_1_conv2d_0_out_channels=1024,
                                    module5_1_conv2d_0_kernel_size=(3, 3),
                                    module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                    module5_1_conv2d_0_pad_mode="pad",
                                    module5_1_conv2d_0_group=2)
        self.split_825 = P.Split(axis=1, output_num=2)
        self.add_826_bias = 0.0
        self.module20_32 = Module20(reshape_2_shape=[1, 1024],
                                    module0_0_avgpool2d_0_kernel_size=(8, 8),
                                    module0_0_conv2d_1_in_channels=512,
                                    module0_0_conv2d_1_out_channels=256,
                                    module0_0_conv2d_3_in_channels=256,
                                    module0_0_conv2d_3_out_channels=1024,
                                    module0_0_reshape_4_shape=[1, 1, 2, 512])
        self.reshape_837 = P.Reshape()
        self.reshape_837_shape = tuple([1, 1024, 1, 1])
        self.split_838 = P.Split(axis=1, output_num=2)
        self.add_841_bias = 0.0
        self.conv2d_843 = nn.Conv2d(in_channels=512,
                                    out_channels=2048,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_845 = nn.ReLU()
        self.avgpool2d_846 = nn.AvgPool2d(kernel_size=(8, 8))
        self.reshape_847 = P.Reshape()
        self.reshape_847_shape = tuple([1, 2048])
        self.flatten_848 = nn.Flatten()
        self.dense_849 = nn.Dense(in_channels=2048, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        module22_0_opt = self.module22_0(opt_relu_1)
        opt_maxpool2d_6 = self.pad_maxpool2d_6(module22_0_opt)
        opt_maxpool2d_6 = self.maxpool2d_6(opt_maxpool2d_6)
        module22_1_opt = self.module22_1(opt_maxpool2d_6)
        opt_split_13, opt_split_13_1 = self.split_13(module22_1_opt)
        opt_add_14 = opt_split_13 + self.add_14_bias
        opt_add_15 = P.Add()(opt_add_14, opt_split_13_1)
        module20_0_opt = self.module20_0(opt_add_15)
        opt_reshape_25 = self.reshape_25(module20_0_opt, self.reshape_25_shape)
        opt_split_26, opt_split_26_1 = self.split_26(opt_reshape_25)
        opt_mul_27 = P.Mul()(opt_split_26, opt_split_13)
        opt_mul_28 = P.Mul()(opt_split_26_1, opt_split_13_1)
        opt_add_29 = opt_mul_27 + self.add_29_bias
        opt_add_30 = P.Add()(opt_add_29, opt_mul_28)
        opt_conv2d_31 = self.conv2d_31(opt_add_30)
        opt_avgpool2d_8 = self.pad_avgpool2d_8(opt_maxpool2d_6)
        opt_avgpool2d_8 = self.avgpool2d_8(opt_avgpool2d_8)
        opt_conv2d_10 = self.conv2d_10(opt_avgpool2d_8)
        opt_add_32 = P.Add()(opt_conv2d_31, opt_conv2d_10)
        opt_relu_33 = self.relu_33(opt_add_32)
        module22_2_opt = self.module22_2(opt_relu_33)
        opt_split_38, opt_split_38_1 = self.split_38(module22_2_opt)
        opt_add_39 = opt_split_38 + self.add_39_bias
        opt_add_40 = P.Add()(opt_add_39, opt_split_38_1)
        module20_1_opt = self.module20_1(opt_add_40)
        opt_reshape_50 = self.reshape_50(module20_1_opt, self.reshape_50_shape)
        opt_split_51, opt_split_51_1 = self.split_51(opt_reshape_50)
        opt_mul_52 = P.Mul()(opt_split_51, opt_split_38)
        opt_mul_53 = P.Mul()(opt_split_51_1, opt_split_38_1)
        opt_add_54 = opt_mul_52 + self.add_54_bias
        opt_add_55 = P.Add()(opt_add_54, opt_mul_53)
        opt_conv2d_56 = self.conv2d_56(opt_add_55)
        opt_add_57 = P.Add()(opt_conv2d_56, opt_relu_33)
        opt_relu_58 = self.relu_58(opt_add_57)
        module22_3_opt = self.module22_3(opt_relu_58)
        opt_split_63, opt_split_63_1 = self.split_63(module22_3_opt)
        opt_add_64 = opt_split_63 + self.add_64_bias
        opt_add_65 = P.Add()(opt_add_64, opt_split_63_1)
        module20_2_opt = self.module20_2(opt_add_65)
        opt_reshape_75 = self.reshape_75(module20_2_opt, self.reshape_75_shape)
        opt_split_76, opt_split_76_1 = self.split_76(opt_reshape_75)
        opt_mul_77 = P.Mul()(opt_split_76, opt_split_63)
        opt_mul_78 = P.Mul()(opt_split_76_1, opt_split_63_1)
        opt_add_79 = opt_mul_77 + self.add_79_bias
        opt_add_80 = P.Add()(opt_add_79, opt_mul_78)
        opt_conv2d_81 = self.conv2d_81(opt_add_80)
        opt_add_82 = P.Add()(opt_conv2d_81, opt_relu_58)
        opt_relu_83 = self.relu_83(opt_add_82)
        module22_4_opt = self.module22_4(opt_relu_83)
        opt_split_90, opt_split_90_1 = self.split_90(module22_4_opt)
        opt_add_91 = opt_split_90 + self.add_91_bias
        opt_add_92 = P.Add()(opt_add_91, opt_split_90_1)
        module20_3_opt = self.module20_3(opt_add_92)
        opt_reshape_102 = self.reshape_102(module20_3_opt, self.reshape_102_shape)
        opt_split_103, opt_split_103_1 = self.split_103(opt_reshape_102)
        opt_mul_104 = P.Mul()(opt_split_103, opt_split_90)
        opt_mul_105 = P.Mul()(opt_split_103_1, opt_split_90_1)
        opt_add_106 = opt_mul_104 + self.add_106_bias
        opt_add_107 = P.Add()(opt_add_106, opt_mul_105)
        opt_pad_108 = self.pad_108(opt_add_107)
        opt_avgpool2d_109 = self.pad_avgpool2d_109(opt_pad_108)
        opt_avgpool2d_109 = self.avgpool2d_109(opt_avgpool2d_109)
        opt_conv2d_110 = self.conv2d_110(opt_avgpool2d_109)
        opt_avgpool2d_85 = self.pad_avgpool2d_85(opt_relu_83)
        opt_avgpool2d_85 = self.avgpool2d_85(opt_avgpool2d_85)
        opt_conv2d_87 = self.conv2d_87(opt_avgpool2d_85)
        opt_add_111 = P.Add()(opt_conv2d_110, opt_conv2d_87)
        opt_relu_112 = self.relu_112(opt_add_111)
        module22_5_opt = self.module22_5(opt_relu_112)
        opt_split_117, opt_split_117_1 = self.split_117(module22_5_opt)
        opt_add_118 = opt_split_117 + self.add_118_bias
        opt_add_119 = P.Add()(opt_add_118, opt_split_117_1)
        module20_4_opt = self.module20_4(opt_add_119)
        opt_reshape_129 = self.reshape_129(module20_4_opt, self.reshape_129_shape)
        opt_split_130, opt_split_130_1 = self.split_130(opt_reshape_129)
        opt_mul_131 = P.Mul()(opt_split_130, opt_split_117)
        opt_mul_132 = P.Mul()(opt_split_130_1, opt_split_117_1)
        opt_add_133 = opt_mul_131 + self.add_133_bias
        opt_add_134 = P.Add()(opt_add_133, opt_mul_132)
        opt_conv2d_135 = self.conv2d_135(opt_add_134)
        opt_add_136 = P.Add()(opt_conv2d_135, opt_relu_112)
        opt_relu_137 = self.relu_137(opt_add_136)
        module22_6_opt = self.module22_6(opt_relu_137)
        opt_split_142, opt_split_142_1 = self.split_142(module22_6_opt)
        opt_add_143 = opt_split_142 + self.add_143_bias
        opt_add_144 = P.Add()(opt_add_143, opt_split_142_1)
        module20_5_opt = self.module20_5(opt_add_144)
        opt_reshape_154 = self.reshape_154(module20_5_opt, self.reshape_154_shape)
        opt_split_155, opt_split_155_1 = self.split_155(opt_reshape_154)
        opt_mul_156 = P.Mul()(opt_split_155, opt_split_142)
        opt_mul_157 = P.Mul()(opt_split_155_1, opt_split_142_1)
        opt_add_158 = opt_mul_156 + self.add_158_bias
        opt_add_159 = P.Add()(opt_add_158, opt_mul_157)
        opt_conv2d_160 = self.conv2d_160(opt_add_159)
        opt_add_161 = P.Add()(opt_conv2d_160, opt_relu_137)
        opt_relu_162 = self.relu_162(opt_add_161)
        module22_7_opt = self.module22_7(opt_relu_162)
        opt_split_167, opt_split_167_1 = self.split_167(module22_7_opt)
        opt_add_168 = opt_split_167 + self.add_168_bias
        opt_add_169 = P.Add()(opt_add_168, opt_split_167_1)
        module20_6_opt = self.module20_6(opt_add_169)
        opt_reshape_179 = self.reshape_179(module20_6_opt, self.reshape_179_shape)
        opt_split_180, opt_split_180_1 = self.split_180(opt_reshape_179)
        opt_mul_181 = P.Mul()(opt_split_180, opt_split_167)
        opt_mul_182 = P.Mul()(opt_split_180_1, opt_split_167_1)
        opt_add_183 = opt_mul_181 + self.add_183_bias
        opt_add_184 = P.Add()(opt_add_183, opt_mul_182)
        opt_conv2d_185 = self.conv2d_185(opt_add_184)
        opt_add_186 = P.Add()(opt_conv2d_185, opt_relu_162)
        opt_relu_187 = self.relu_187(opt_add_186)
        module22_8_opt = self.module22_8(opt_relu_187)
        opt_split_194, opt_split_194_1 = self.split_194(module22_8_opt)
        opt_add_195 = opt_split_194 + self.add_195_bias
        opt_add_196 = P.Add()(opt_add_195, opt_split_194_1)
        module20_7_opt = self.module20_7(opt_add_196)
        opt_reshape_206 = self.reshape_206(module20_7_opt, self.reshape_206_shape)
        opt_split_207, opt_split_207_1 = self.split_207(opt_reshape_206)
        opt_mul_208 = P.Mul()(opt_split_207, opt_split_194)
        opt_mul_209 = P.Mul()(opt_split_207_1, opt_split_194_1)
        opt_add_210 = opt_mul_208 + self.add_210_bias
        opt_add_211 = P.Add()(opt_add_210, opt_mul_209)
        opt_pad_212 = self.pad_212(opt_add_211)
        opt_avgpool2d_213 = self.pad_avgpool2d_213(opt_pad_212)
        opt_avgpool2d_213 = self.avgpool2d_213(opt_avgpool2d_213)
        opt_conv2d_214 = self.conv2d_214(opt_avgpool2d_213)
        opt_avgpool2d_189 = self.pad_avgpool2d_189(opt_relu_187)
        opt_avgpool2d_189 = self.avgpool2d_189(opt_avgpool2d_189)
        opt_conv2d_191 = self.conv2d_191(opt_avgpool2d_189)
        opt_add_215 = P.Add()(opt_conv2d_214, opt_conv2d_191)
        opt_relu_216 = self.relu_216(opt_add_215)
        module22_9_opt = self.module22_9(opt_relu_216)
        opt_split_221, opt_split_221_1 = self.split_221(module22_9_opt)
        opt_add_222 = opt_split_221 + self.add_222_bias
        opt_add_223 = P.Add()(opt_add_222, opt_split_221_1)
        module20_8_opt = self.module20_8(opt_add_223)
        opt_reshape_233 = self.reshape_233(module20_8_opt, self.reshape_233_shape)
        opt_split_234, opt_split_234_1 = self.split_234(opt_reshape_233)
        opt_mul_235 = P.Mul()(opt_split_234, opt_split_221)
        opt_mul_236 = P.Mul()(opt_split_234_1, opt_split_221_1)
        opt_add_237 = opt_mul_235 + self.add_237_bias
        opt_add_238 = P.Add()(opt_add_237, opt_mul_236)
        opt_conv2d_239 = self.conv2d_239(opt_add_238)
        opt_add_240 = P.Add()(opt_conv2d_239, opt_relu_216)
        opt_relu_241 = self.relu_241(opt_add_240)
        module22_10_opt = self.module22_10(opt_relu_241)
        opt_split_246, opt_split_246_1 = self.split_246(module22_10_opt)
        opt_add_247 = opt_split_246 + self.add_247_bias
        opt_add_248 = P.Add()(opt_add_247, opt_split_246_1)
        module20_9_opt = self.module20_9(opt_add_248)
        opt_reshape_258 = self.reshape_258(module20_9_opt, self.reshape_258_shape)
        opt_split_259, opt_split_259_1 = self.split_259(opt_reshape_258)
        opt_mul_260 = P.Mul()(opt_split_259, opt_split_246)
        opt_mul_261 = P.Mul()(opt_split_259_1, opt_split_246_1)
        opt_add_262 = opt_mul_260 + self.add_262_bias
        opt_add_263 = P.Add()(opt_add_262, opt_mul_261)
        opt_conv2d_264 = self.conv2d_264(opt_add_263)
        opt_add_265 = P.Add()(opt_conv2d_264, opt_relu_241)
        opt_relu_266 = self.relu_266(opt_add_265)
        module22_11_opt = self.module22_11(opt_relu_266)
        opt_split_271, opt_split_271_1 = self.split_271(module22_11_opt)
        opt_add_272 = opt_split_271 + self.add_272_bias
        opt_add_273 = P.Add()(opt_add_272, opt_split_271_1)
        module20_10_opt = self.module20_10(opt_add_273)
        opt_reshape_283 = self.reshape_283(module20_10_opt, self.reshape_283_shape)
        opt_split_284, opt_split_284_1 = self.split_284(opt_reshape_283)
        opt_mul_285 = P.Mul()(opt_split_284, opt_split_271)
        opt_mul_286 = P.Mul()(opt_split_284_1, opt_split_271_1)
        opt_add_287 = opt_mul_285 + self.add_287_bias
        opt_add_288 = P.Add()(opt_add_287, opt_mul_286)
        opt_conv2d_289 = self.conv2d_289(opt_add_288)
        opt_add_290 = P.Add()(opt_conv2d_289, opt_relu_266)
        opt_relu_291 = self.relu_291(opt_add_290)
        module22_12_opt = self.module22_12(opt_relu_291)
        opt_split_296, opt_split_296_1 = self.split_296(module22_12_opt)
        opt_add_297 = opt_split_296 + self.add_297_bias
        opt_add_298 = P.Add()(opt_add_297, opt_split_296_1)
        module20_11_opt = self.module20_11(opt_add_298)
        opt_reshape_308 = self.reshape_308(module20_11_opt, self.reshape_308_shape)
        opt_split_309, opt_split_309_1 = self.split_309(opt_reshape_308)
        opt_mul_310 = P.Mul()(opt_split_309, opt_split_296)
        opt_mul_311 = P.Mul()(opt_split_309_1, opt_split_296_1)
        opt_add_312 = opt_mul_310 + self.add_312_bias
        opt_add_313 = P.Add()(opt_add_312, opt_mul_311)
        opt_conv2d_314 = self.conv2d_314(opt_add_313)
        opt_add_315 = P.Add()(opt_conv2d_314, opt_relu_291)
        opt_relu_316 = self.relu_316(opt_add_315)
        module22_13_opt = self.module22_13(opt_relu_316)
        opt_split_321, opt_split_321_1 = self.split_321(module22_13_opt)
        opt_add_322 = opt_split_321 + self.add_322_bias
        opt_add_323 = P.Add()(opt_add_322, opt_split_321_1)
        module20_12_opt = self.module20_12(opt_add_323)
        opt_reshape_333 = self.reshape_333(module20_12_opt, self.reshape_333_shape)
        opt_split_334, opt_split_334_1 = self.split_334(opt_reshape_333)
        opt_mul_335 = P.Mul()(opt_split_334, opt_split_321)
        opt_mul_336 = P.Mul()(opt_split_334_1, opt_split_321_1)
        opt_add_337 = opt_mul_335 + self.add_337_bias
        opt_add_338 = P.Add()(opt_add_337, opt_mul_336)
        opt_conv2d_339 = self.conv2d_339(opt_add_338)
        opt_add_340 = P.Add()(opt_conv2d_339, opt_relu_316)
        opt_relu_341 = self.relu_341(opt_add_340)
        module22_14_opt = self.module22_14(opt_relu_341)
        opt_split_346, opt_split_346_1 = self.split_346(module22_14_opt)
        opt_add_347 = opt_split_346 + self.add_347_bias
        opt_add_348 = P.Add()(opt_add_347, opt_split_346_1)
        module20_13_opt = self.module20_13(opt_add_348)
        opt_reshape_358 = self.reshape_358(module20_13_opt, self.reshape_358_shape)
        opt_split_359, opt_split_359_1 = self.split_359(opt_reshape_358)
        opt_mul_360 = P.Mul()(opt_split_359, opt_split_346)
        opt_mul_361 = P.Mul()(opt_split_359_1, opt_split_346_1)
        opt_add_362 = opt_mul_360 + self.add_362_bias
        opt_add_363 = P.Add()(opt_add_362, opt_mul_361)
        opt_conv2d_364 = self.conv2d_364(opt_add_363)
        opt_add_365 = P.Add()(opt_conv2d_364, opt_relu_341)
        opt_relu_366 = self.relu_366(opt_add_365)
        module22_15_opt = self.module22_15(opt_relu_366)
        opt_split_371, opt_split_371_1 = self.split_371(module22_15_opt)
        opt_add_372 = opt_split_371 + self.add_372_bias
        opt_add_373 = P.Add()(opt_add_372, opt_split_371_1)
        module20_14_opt = self.module20_14(opt_add_373)
        opt_reshape_383 = self.reshape_383(module20_14_opt, self.reshape_383_shape)
        opt_split_384, opt_split_384_1 = self.split_384(opt_reshape_383)
        opt_mul_385 = P.Mul()(opt_split_384, opt_split_371)
        opt_mul_386 = P.Mul()(opt_split_384_1, opt_split_371_1)
        opt_add_387 = opt_mul_385 + self.add_387_bias
        opt_add_388 = P.Add()(opt_add_387, opt_mul_386)
        opt_conv2d_389 = self.conv2d_389(opt_add_388)
        opt_add_390 = P.Add()(opt_conv2d_389, opt_relu_366)
        opt_relu_391 = self.relu_391(opt_add_390)
        module22_16_opt = self.module22_16(opt_relu_391)
        opt_split_396, opt_split_396_1 = self.split_396(module22_16_opt)
        opt_add_397 = opt_split_396 + self.add_397_bias
        opt_add_398 = P.Add()(opt_add_397, opt_split_396_1)
        module20_15_opt = self.module20_15(opt_add_398)
        opt_reshape_408 = self.reshape_408(module20_15_opt, self.reshape_408_shape)
        opt_split_409, opt_split_409_1 = self.split_409(opt_reshape_408)
        opt_mul_410 = P.Mul()(opt_split_409, opt_split_396)
        opt_mul_411 = P.Mul()(opt_split_409_1, opt_split_396_1)
        opt_add_412 = opt_mul_410 + self.add_412_bias
        opt_add_413 = P.Add()(opt_add_412, opt_mul_411)
        opt_conv2d_414 = self.conv2d_414(opt_add_413)
        opt_add_415 = P.Add()(opt_conv2d_414, opt_relu_391)
        opt_relu_416 = self.relu_416(opt_add_415)
        module22_17_opt = self.module22_17(opt_relu_416)
        opt_split_421, opt_split_421_1 = self.split_421(module22_17_opt)
        opt_add_422 = opt_split_421 + self.add_422_bias
        opt_add_423 = P.Add()(opt_add_422, opt_split_421_1)
        module20_16_opt = self.module20_16(opt_add_423)
        opt_reshape_433 = self.reshape_433(module20_16_opt, self.reshape_433_shape)
        opt_split_434, opt_split_434_1 = self.split_434(opt_reshape_433)
        opt_mul_435 = P.Mul()(opt_split_434, opt_split_421)
        opt_mul_436 = P.Mul()(opt_split_434_1, opt_split_421_1)
        opt_add_437 = opt_mul_435 + self.add_437_bias
        opt_add_438 = P.Add()(opt_add_437, opt_mul_436)
        opt_conv2d_439 = self.conv2d_439(opt_add_438)
        opt_add_440 = P.Add()(opt_conv2d_439, opt_relu_416)
        opt_relu_441 = self.relu_441(opt_add_440)
        module22_18_opt = self.module22_18(opt_relu_441)
        opt_split_446, opt_split_446_1 = self.split_446(module22_18_opt)
        opt_add_447 = opt_split_446 + self.add_447_bias
        opt_add_448 = P.Add()(opt_add_447, opt_split_446_1)
        module20_17_opt = self.module20_17(opt_add_448)
        opt_reshape_458 = self.reshape_458(module20_17_opt, self.reshape_458_shape)
        opt_split_459, opt_split_459_1 = self.split_459(opt_reshape_458)
        opt_mul_460 = P.Mul()(opt_split_459, opt_split_446)
        opt_mul_461 = P.Mul()(opt_split_459_1, opt_split_446_1)
        opt_add_462 = opt_mul_460 + self.add_462_bias
        opt_add_463 = P.Add()(opt_add_462, opt_mul_461)
        opt_conv2d_464 = self.conv2d_464(opt_add_463)
        opt_add_465 = P.Add()(opt_conv2d_464, opt_relu_441)
        opt_relu_466 = self.relu_466(opt_add_465)
        module22_19_opt = self.module22_19(opt_relu_466)
        opt_split_471, opt_split_471_1 = self.split_471(module22_19_opt)
        opt_add_472 = opt_split_471 + self.add_472_bias
        opt_add_473 = P.Add()(opt_add_472, opt_split_471_1)
        module20_18_opt = self.module20_18(opt_add_473)
        opt_reshape_483 = self.reshape_483(module20_18_opt, self.reshape_483_shape)
        opt_split_484, opt_split_484_1 = self.split_484(opt_reshape_483)
        opt_mul_485 = P.Mul()(opt_split_484, opt_split_471)
        opt_mul_486 = P.Mul()(opt_split_484_1, opt_split_471_1)
        opt_add_487 = opt_mul_485 + self.add_487_bias
        opt_add_488 = P.Add()(opt_add_487, opt_mul_486)
        opt_conv2d_489 = self.conv2d_489(opt_add_488)
        opt_add_490 = P.Add()(opt_conv2d_489, opt_relu_466)
        opt_relu_491 = self.relu_491(opt_add_490)
        module22_20_opt = self.module22_20(opt_relu_491)
        opt_split_496, opt_split_496_1 = self.split_496(module22_20_opt)
        opt_add_497 = opt_split_496 + self.add_497_bias
        opt_add_498 = P.Add()(opt_add_497, opt_split_496_1)
        module20_19_opt = self.module20_19(opt_add_498)
        opt_reshape_508 = self.reshape_508(module20_19_opt, self.reshape_508_shape)
        opt_split_509, opt_split_509_1 = self.split_509(opt_reshape_508)
        opt_mul_510 = P.Mul()(opt_split_509, opt_split_496)
        opt_mul_511 = P.Mul()(opt_split_509_1, opt_split_496_1)
        opt_add_512 = opt_mul_510 + self.add_512_bias
        opt_add_513 = P.Add()(opt_add_512, opt_mul_511)
        opt_conv2d_514 = self.conv2d_514(opt_add_513)
        opt_add_515 = P.Add()(opt_conv2d_514, opt_relu_491)
        opt_relu_516 = self.relu_516(opt_add_515)
        module22_21_opt = self.module22_21(opt_relu_516)
        opt_split_521, opt_split_521_1 = self.split_521(module22_21_opt)
        opt_add_522 = opt_split_521 + self.add_522_bias
        opt_add_523 = P.Add()(opt_add_522, opt_split_521_1)
        module20_20_opt = self.module20_20(opt_add_523)
        opt_reshape_533 = self.reshape_533(module20_20_opt, self.reshape_533_shape)
        opt_split_534, opt_split_534_1 = self.split_534(opt_reshape_533)
        opt_mul_535 = P.Mul()(opt_split_534, opt_split_521)
        opt_mul_536 = P.Mul()(opt_split_534_1, opt_split_521_1)
        opt_add_537 = opt_mul_535 + self.add_537_bias
        opt_add_538 = P.Add()(opt_add_537, opt_mul_536)
        opt_conv2d_539 = self.conv2d_539(opt_add_538)
        opt_add_540 = P.Add()(opt_conv2d_539, opt_relu_516)
        opt_relu_541 = self.relu_541(opt_add_540)
        module22_22_opt = self.module22_22(opt_relu_541)
        opt_split_546, opt_split_546_1 = self.split_546(module22_22_opt)
        opt_add_547 = opt_split_546 + self.add_547_bias
        opt_add_548 = P.Add()(opt_add_547, opt_split_546_1)
        module20_21_opt = self.module20_21(opt_add_548)
        opt_reshape_558 = self.reshape_558(module20_21_opt, self.reshape_558_shape)
        opt_split_559, opt_split_559_1 = self.split_559(opt_reshape_558)
        opt_mul_560 = P.Mul()(opt_split_559, opt_split_546)
        opt_mul_561 = P.Mul()(opt_split_559_1, opt_split_546_1)
        opt_add_562 = opt_mul_560 + self.add_562_bias
        opt_add_563 = P.Add()(opt_add_562, opt_mul_561)
        opt_conv2d_564 = self.conv2d_564(opt_add_563)
        opt_add_565 = P.Add()(opt_conv2d_564, opt_relu_541)
        opt_relu_566 = self.relu_566(opt_add_565)
        module22_23_opt = self.module22_23(opt_relu_566)
        opt_split_571, opt_split_571_1 = self.split_571(module22_23_opt)
        opt_add_572 = opt_split_571 + self.add_572_bias
        opt_add_573 = P.Add()(opt_add_572, opt_split_571_1)
        module20_22_opt = self.module20_22(opt_add_573)
        opt_reshape_583 = self.reshape_583(module20_22_opt, self.reshape_583_shape)
        opt_split_584, opt_split_584_1 = self.split_584(opt_reshape_583)
        opt_mul_585 = P.Mul()(opt_split_584, opt_split_571)
        opt_mul_586 = P.Mul()(opt_split_584_1, opt_split_571_1)
        opt_add_587 = opt_mul_585 + self.add_587_bias
        opt_add_588 = P.Add()(opt_add_587, opt_mul_586)
        opt_conv2d_589 = self.conv2d_589(opt_add_588)
        opt_add_590 = P.Add()(opt_conv2d_589, opt_relu_566)
        opt_relu_591 = self.relu_591(opt_add_590)
        module22_24_opt = self.module22_24(opt_relu_591)
        opt_split_596, opt_split_596_1 = self.split_596(module22_24_opt)
        opt_add_597 = opt_split_596 + self.add_597_bias
        opt_add_598 = P.Add()(opt_add_597, opt_split_596_1)
        module20_23_opt = self.module20_23(opt_add_598)
        opt_reshape_608 = self.reshape_608(module20_23_opt, self.reshape_608_shape)
        opt_split_609, opt_split_609_1 = self.split_609(opt_reshape_608)
        opt_mul_610 = P.Mul()(opt_split_609, opt_split_596)
        opt_mul_611 = P.Mul()(opt_split_609_1, opt_split_596_1)
        opt_add_612 = opt_mul_610 + self.add_612_bias
        opt_add_613 = P.Add()(opt_add_612, opt_mul_611)
        opt_conv2d_614 = self.conv2d_614(opt_add_613)
        opt_add_615 = P.Add()(opt_conv2d_614, opt_relu_591)
        opt_relu_616 = self.relu_616(opt_add_615)
        module22_25_opt = self.module22_25(opt_relu_616)
        opt_split_621, opt_split_621_1 = self.split_621(module22_25_opt)
        opt_add_622 = opt_split_621 + self.add_622_bias
        opt_add_623 = P.Add()(opt_add_622, opt_split_621_1)
        module20_24_opt = self.module20_24(opt_add_623)
        opt_reshape_633 = self.reshape_633(module20_24_opt, self.reshape_633_shape)
        opt_split_634, opt_split_634_1 = self.split_634(opt_reshape_633)
        opt_mul_635 = P.Mul()(opt_split_634, opt_split_621)
        opt_mul_636 = P.Mul()(opt_split_634_1, opt_split_621_1)
        opt_add_637 = opt_mul_635 + self.add_637_bias
        opt_add_638 = P.Add()(opt_add_637, opt_mul_636)
        opt_conv2d_639 = self.conv2d_639(opt_add_638)
        opt_add_640 = P.Add()(opt_conv2d_639, opt_relu_616)
        opt_relu_641 = self.relu_641(opt_add_640)
        module22_26_opt = self.module22_26(opt_relu_641)
        opt_split_646, opt_split_646_1 = self.split_646(module22_26_opt)
        opt_add_647 = opt_split_646 + self.add_647_bias
        opt_add_648 = P.Add()(opt_add_647, opt_split_646_1)
        module20_25_opt = self.module20_25(opt_add_648)
        opt_reshape_658 = self.reshape_658(module20_25_opt, self.reshape_658_shape)
        opt_split_659, opt_split_659_1 = self.split_659(opt_reshape_658)
        opt_mul_660 = P.Mul()(opt_split_659, opt_split_646)
        opt_mul_661 = P.Mul()(opt_split_659_1, opt_split_646_1)
        opt_add_662 = opt_mul_660 + self.add_662_bias
        opt_add_663 = P.Add()(opt_add_662, opt_mul_661)
        opt_conv2d_664 = self.conv2d_664(opt_add_663)
        opt_add_665 = P.Add()(opt_conv2d_664, opt_relu_641)
        opt_relu_666 = self.relu_666(opt_add_665)
        module22_27_opt = self.module22_27(opt_relu_666)
        opt_split_671, opt_split_671_1 = self.split_671(module22_27_opt)
        opt_add_672 = opt_split_671 + self.add_672_bias
        opt_add_673 = P.Add()(opt_add_672, opt_split_671_1)
        module20_26_opt = self.module20_26(opt_add_673)
        opt_reshape_683 = self.reshape_683(module20_26_opt, self.reshape_683_shape)
        opt_split_684, opt_split_684_1 = self.split_684(opt_reshape_683)
        opt_mul_685 = P.Mul()(opt_split_684, opt_split_671)
        opt_mul_686 = P.Mul()(opt_split_684_1, opt_split_671_1)
        opt_add_687 = opt_mul_685 + self.add_687_bias
        opt_add_688 = P.Add()(opt_add_687, opt_mul_686)
        opt_conv2d_689 = self.conv2d_689(opt_add_688)
        opt_add_690 = P.Add()(opt_conv2d_689, opt_relu_666)
        opt_relu_691 = self.relu_691(opt_add_690)
        module22_28_opt = self.module22_28(opt_relu_691)
        opt_split_696, opt_split_696_1 = self.split_696(module22_28_opt)
        opt_add_697 = opt_split_696 + self.add_697_bias
        opt_add_698 = P.Add()(opt_add_697, opt_split_696_1)
        module20_27_opt = self.module20_27(opt_add_698)
        opt_reshape_708 = self.reshape_708(module20_27_opt, self.reshape_708_shape)
        opt_split_709, opt_split_709_1 = self.split_709(opt_reshape_708)
        opt_mul_710 = P.Mul()(opt_split_709, opt_split_696)
        opt_mul_711 = P.Mul()(opt_split_709_1, opt_split_696_1)
        opt_add_712 = opt_mul_710 + self.add_712_bias
        opt_add_713 = P.Add()(opt_add_712, opt_mul_711)
        opt_conv2d_714 = self.conv2d_714(opt_add_713)
        opt_add_715 = P.Add()(opt_conv2d_714, opt_relu_691)
        opt_relu_716 = self.relu_716(opt_add_715)
        module22_29_opt = self.module22_29(opt_relu_716)
        opt_split_721, opt_split_721_1 = self.split_721(module22_29_opt)
        opt_add_722 = opt_split_721 + self.add_722_bias
        opt_add_723 = P.Add()(opt_add_722, opt_split_721_1)
        module20_28_opt = self.module20_28(opt_add_723)
        opt_reshape_733 = self.reshape_733(module20_28_opt, self.reshape_733_shape)
        opt_split_734, opt_split_734_1 = self.split_734(opt_reshape_733)
        opt_mul_735 = P.Mul()(opt_split_734, opt_split_721)
        opt_mul_736 = P.Mul()(opt_split_734_1, opt_split_721_1)
        opt_add_737 = opt_mul_735 + self.add_737_bias
        opt_add_738 = P.Add()(opt_add_737, opt_mul_736)
        opt_conv2d_739 = self.conv2d_739(opt_add_738)
        opt_add_740 = P.Add()(opt_conv2d_739, opt_relu_716)
        opt_relu_741 = self.relu_741(opt_add_740)
        module22_30_opt = self.module22_30(opt_relu_741)
        opt_split_746, opt_split_746_1 = self.split_746(module22_30_opt)
        opt_add_747 = opt_split_746 + self.add_747_bias
        opt_add_748 = P.Add()(opt_add_747, opt_split_746_1)
        module20_29_opt = self.module20_29(opt_add_748)
        opt_reshape_758 = self.reshape_758(module20_29_opt, self.reshape_758_shape)
        opt_split_759, opt_split_759_1 = self.split_759(opt_reshape_758)
        opt_mul_760 = P.Mul()(opt_split_759, opt_split_746)
        opt_mul_761 = P.Mul()(opt_split_759_1, opt_split_746_1)
        opt_add_762 = opt_mul_760 + self.add_762_bias
        opt_add_763 = P.Add()(opt_add_762, opt_mul_761)
        opt_conv2d_764 = self.conv2d_764(opt_add_763)
        opt_add_765 = P.Add()(opt_conv2d_764, opt_relu_741)
        opt_relu_766 = self.relu_766(opt_add_765)
        module22_31_opt = self.module22_31(opt_relu_766)
        opt_split_773, opt_split_773_1 = self.split_773(module22_31_opt)
        opt_add_774 = opt_split_773 + self.add_774_bias
        opt_add_775 = P.Add()(opt_add_774, opt_split_773_1)
        module20_30_opt = self.module20_30(opt_add_775)
        opt_reshape_785 = self.reshape_785(module20_30_opt, self.reshape_785_shape)
        opt_split_786, opt_split_786_1 = self.split_786(opt_reshape_785)
        opt_mul_787 = P.Mul()(opt_split_786, opt_split_773)
        opt_mul_788 = P.Mul()(opt_split_786_1, opt_split_773_1)
        opt_add_789 = opt_mul_787 + self.add_789_bias
        opt_add_790 = P.Add()(opt_add_789, opt_mul_788)
        opt_pad_791 = self.pad_791(opt_add_790)
        opt_avgpool2d_792 = self.pad_avgpool2d_792(opt_pad_791)
        opt_avgpool2d_792 = self.avgpool2d_792(opt_avgpool2d_792)
        opt_conv2d_793 = self.conv2d_793(opt_avgpool2d_792)
        opt_avgpool2d_768 = self.pad_avgpool2d_768(opt_relu_766)
        opt_avgpool2d_768 = self.avgpool2d_768(opt_avgpool2d_768)
        opt_conv2d_770 = self.conv2d_770(opt_avgpool2d_768)
        opt_add_794 = P.Add()(opt_conv2d_793, opt_conv2d_770)
        opt_relu_795 = self.relu_795(opt_add_794)
        module22_32_opt = self.module22_32(opt_relu_795)
        opt_split_800, opt_split_800_1 = self.split_800(module22_32_opt)
        opt_add_801 = opt_split_800 + self.add_801_bias
        opt_add_802 = P.Add()(opt_add_801, opt_split_800_1)
        module20_31_opt = self.module20_31(opt_add_802)
        opt_reshape_812 = self.reshape_812(module20_31_opt, self.reshape_812_shape)
        opt_split_813, opt_split_813_1 = self.split_813(opt_reshape_812)
        opt_mul_814 = P.Mul()(opt_split_813, opt_split_800)
        opt_mul_815 = P.Mul()(opt_split_813_1, opt_split_800_1)
        opt_add_816 = opt_mul_814 + self.add_816_bias
        opt_add_817 = P.Add()(opt_add_816, opt_mul_815)
        opt_conv2d_818 = self.conv2d_818(opt_add_817)
        opt_add_819 = P.Add()(opt_conv2d_818, opt_relu_795)
        opt_relu_820 = self.relu_820(opt_add_819)
        module22_33_opt = self.module22_33(opt_relu_820)
        opt_split_825, opt_split_825_1 = self.split_825(module22_33_opt)
        opt_add_826 = opt_split_825 + self.add_826_bias
        opt_add_827 = P.Add()(opt_add_826, opt_split_825_1)
        module20_32_opt = self.module20_32(opt_add_827)
        opt_reshape_837 = self.reshape_837(module20_32_opt, self.reshape_837_shape)
        opt_split_838, opt_split_838_1 = self.split_838(opt_reshape_837)
        opt_mul_839 = P.Mul()(opt_split_838, opt_split_825)
        opt_mul_840 = P.Mul()(opt_split_838_1, opt_split_825_1)
        opt_add_841 = opt_mul_839 + self.add_841_bias
        opt_add_842 = P.Add()(opt_add_841, opt_mul_840)
        opt_conv2d_843 = self.conv2d_843(opt_add_842)
        opt_add_844 = P.Add()(opt_conv2d_843, opt_relu_820)
        opt_relu_845 = self.relu_845(opt_add_844)
        opt_avgpool2d_846 = self.avgpool2d_846(opt_relu_845)
        opt_reshape_847 = self.reshape_847(opt_avgpool2d_846, self.reshape_847_shape)
        opt_flatten_848 = self.flatten_848(opt_reshape_847)
        opt_dense_849 = self.dense_849(opt_flatten_848)
        return opt_dense_849
