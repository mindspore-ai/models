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
                                   module0_0_avgpool2d_0_kernel_size=(104, 104),
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
                                   module0_0_avgpool2d_0_kernel_size=(104, 104),
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
                                   module0_0_avgpool2d_0_kernel_size=(104, 104),
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
                                   module0_0_avgpool2d_0_kernel_size=(104, 104),
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
                                   module0_0_avgpool2d_0_kernel_size=(52, 52),
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
                                   module0_0_avgpool2d_0_kernel_size=(52, 52),
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
                                   module0_0_avgpool2d_0_kernel_size=(52, 52),
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
        self.split_192 = P.Split(axis=1, output_num=2)
        self.add_193_bias = 0.0
        self.module20_7 = Module20(reshape_2_shape=[1, 256],
                                   module0_0_avgpool2d_0_kernel_size=(52, 52),
                                   module0_0_conv2d_1_in_channels=128,
                                   module0_0_conv2d_1_out_channels=64,
                                   module0_0_conv2d_3_in_channels=64,
                                   module0_0_conv2d_3_out_channels=256,
                                   module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_204 = P.Reshape()
        self.reshape_204_shape = tuple([1, 256, 1, 1])
        self.split_205 = P.Split(axis=1, output_num=2)
        self.add_208_bias = 0.0
        self.conv2d_210 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_212 = nn.ReLU()
        self.module22_9 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_217 = P.Split(axis=1, output_num=2)
        self.add_218_bias = 0.0
        self.module20_8 = Module20(reshape_2_shape=[1, 256],
                                   module0_0_avgpool2d_0_kernel_size=(52, 52),
                                   module0_0_conv2d_1_in_channels=128,
                                   module0_0_conv2d_1_out_channels=64,
                                   module0_0_conv2d_3_in_channels=64,
                                   module0_0_conv2d_3_out_channels=256,
                                   module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_229 = P.Reshape()
        self.reshape_229_shape = tuple([1, 256, 1, 1])
        self.split_230 = P.Split(axis=1, output_num=2)
        self.add_233_bias = 0.0
        self.conv2d_235 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_237 = nn.ReLU()
        self.module22_10 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_242 = P.Split(axis=1, output_num=2)
        self.add_243_bias = 0.0
        self.module20_9 = Module20(reshape_2_shape=[1, 256],
                                   module0_0_avgpool2d_0_kernel_size=(52, 52),
                                   module0_0_conv2d_1_in_channels=128,
                                   module0_0_conv2d_1_out_channels=64,
                                   module0_0_conv2d_3_in_channels=64,
                                   module0_0_conv2d_3_out_channels=256,
                                   module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_254 = P.Reshape()
        self.reshape_254_shape = tuple([1, 256, 1, 1])
        self.split_255 = P.Split(axis=1, output_num=2)
        self.add_258_bias = 0.0
        self.conv2d_260 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_262 = nn.ReLU()
        self.module22_11 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_267 = P.Split(axis=1, output_num=2)
        self.add_268_bias = 0.0
        self.module20_10 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_279 = P.Reshape()
        self.reshape_279_shape = tuple([1, 256, 1, 1])
        self.split_280 = P.Split(axis=1, output_num=2)
        self.add_283_bias = 0.0
        self.conv2d_285 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_287 = nn.ReLU()
        self.module22_12 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_292 = P.Split(axis=1, output_num=2)
        self.add_293_bias = 0.0
        self.module20_11 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_304 = P.Reshape()
        self.reshape_304_shape = tuple([1, 256, 1, 1])
        self.split_305 = P.Split(axis=1, output_num=2)
        self.add_308_bias = 0.0
        self.conv2d_310 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_312 = nn.ReLU()
        self.module22_13 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_317 = P.Split(axis=1, output_num=2)
        self.add_318_bias = 0.0
        self.module20_12 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_329 = P.Reshape()
        self.reshape_329_shape = tuple([1, 256, 1, 1])
        self.split_330 = P.Split(axis=1, output_num=2)
        self.add_333_bias = 0.0
        self.conv2d_335 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_337 = nn.ReLU()
        self.module22_14 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_342 = P.Split(axis=1, output_num=2)
        self.add_343_bias = 0.0
        self.module20_13 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_354 = P.Reshape()
        self.reshape_354_shape = tuple([1, 256, 1, 1])
        self.split_355 = P.Split(axis=1, output_num=2)
        self.add_358_bias = 0.0
        self.conv2d_360 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_362 = nn.ReLU()
        self.module22_15 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_367 = P.Split(axis=1, output_num=2)
        self.add_368_bias = 0.0
        self.module20_14 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_379 = P.Reshape()
        self.reshape_379_shape = tuple([1, 256, 1, 1])
        self.split_380 = P.Split(axis=1, output_num=2)
        self.add_383_bias = 0.0
        self.conv2d_385 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_387 = nn.ReLU()
        self.module22_16 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_392 = P.Split(axis=1, output_num=2)
        self.add_393_bias = 0.0
        self.module20_15 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_404 = P.Reshape()
        self.reshape_404_shape = tuple([1, 256, 1, 1])
        self.split_405 = P.Split(axis=1, output_num=2)
        self.add_408_bias = 0.0
        self.conv2d_410 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_412 = nn.ReLU()
        self.module22_17 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_417 = P.Split(axis=1, output_num=2)
        self.add_418_bias = 0.0
        self.module20_16 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_429 = P.Reshape()
        self.reshape_429_shape = tuple([1, 256, 1, 1])
        self.split_430 = P.Split(axis=1, output_num=2)
        self.add_433_bias = 0.0
        self.conv2d_435 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_437 = nn.ReLU()
        self.module22_18 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_442 = P.Split(axis=1, output_num=2)
        self.add_443_bias = 0.0
        self.module20_17 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_454 = P.Reshape()
        self.reshape_454_shape = tuple([1, 256, 1, 1])
        self.split_455 = P.Split(axis=1, output_num=2)
        self.add_458_bias = 0.0
        self.conv2d_460 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_462 = nn.ReLU()
        self.module22_19 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_467 = P.Split(axis=1, output_num=2)
        self.add_468_bias = 0.0
        self.module20_18 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_479 = P.Reshape()
        self.reshape_479_shape = tuple([1, 256, 1, 1])
        self.split_480 = P.Split(axis=1, output_num=2)
        self.add_483_bias = 0.0
        self.conv2d_485 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_487 = nn.ReLU()
        self.module22_20 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_492 = P.Split(axis=1, output_num=2)
        self.add_493_bias = 0.0
        self.module20_19 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_504 = P.Reshape()
        self.reshape_504_shape = tuple([1, 256, 1, 1])
        self.split_505 = P.Split(axis=1, output_num=2)
        self.add_508_bias = 0.0
        self.conv2d_510 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_512 = nn.ReLU()
        self.module22_21 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_517 = P.Split(axis=1, output_num=2)
        self.add_518_bias = 0.0
        self.module20_20 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_529 = P.Reshape()
        self.reshape_529_shape = tuple([1, 256, 1, 1])
        self.split_530 = P.Split(axis=1, output_num=2)
        self.add_533_bias = 0.0
        self.conv2d_535 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_537 = nn.ReLU()
        self.module22_22 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_542 = P.Split(axis=1, output_num=2)
        self.add_543_bias = 0.0
        self.module20_21 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_554 = P.Reshape()
        self.reshape_554_shape = tuple([1, 256, 1, 1])
        self.split_555 = P.Split(axis=1, output_num=2)
        self.add_558_bias = 0.0
        self.conv2d_560 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_562 = nn.ReLU()
        self.module22_23 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_567 = P.Split(axis=1, output_num=2)
        self.add_568_bias = 0.0
        self.module20_22 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_579 = P.Reshape()
        self.reshape_579_shape = tuple([1, 256, 1, 1])
        self.split_580 = P.Split(axis=1, output_num=2)
        self.add_583_bias = 0.0
        self.conv2d_585 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_587 = nn.ReLU()
        self.module22_24 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_592 = P.Split(axis=1, output_num=2)
        self.add_593_bias = 0.0
        self.module20_23 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_604 = P.Reshape()
        self.reshape_604_shape = tuple([1, 256, 1, 1])
        self.split_605 = P.Split(axis=1, output_num=2)
        self.add_608_bias = 0.0
        self.conv2d_610 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_612 = nn.ReLU()
        self.module22_25 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_617 = P.Split(axis=1, output_num=2)
        self.add_618_bias = 0.0
        self.module20_24 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_629 = P.Reshape()
        self.reshape_629_shape = tuple([1, 256, 1, 1])
        self.split_630 = P.Split(axis=1, output_num=2)
        self.add_633_bias = 0.0
        self.conv2d_635 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_637 = nn.ReLU()
        self.module22_26 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_642 = P.Split(axis=1, output_num=2)
        self.add_643_bias = 0.0
        self.module20_25 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_654 = P.Reshape()
        self.reshape_654_shape = tuple([1, 256, 1, 1])
        self.split_655 = P.Split(axis=1, output_num=2)
        self.add_658_bias = 0.0
        self.conv2d_660 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_662 = nn.ReLU()
        self.module22_27 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_667 = P.Split(axis=1, output_num=2)
        self.add_668_bias = 0.0
        self.module20_26 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_679 = P.Reshape()
        self.reshape_679_shape = tuple([1, 256, 1, 1])
        self.split_680 = P.Split(axis=1, output_num=2)
        self.add_683_bias = 0.0
        self.conv2d_685 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_687 = nn.ReLU()
        self.module22_28 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_692 = P.Split(axis=1, output_num=2)
        self.add_693_bias = 0.0
        self.module20_27 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_704 = P.Reshape()
        self.reshape_704_shape = tuple([1, 256, 1, 1])
        self.split_705 = P.Split(axis=1, output_num=2)
        self.add_708_bias = 0.0
        self.conv2d_710 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_712 = nn.ReLU()
        self.module22_29 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_717 = P.Split(axis=1, output_num=2)
        self.add_718_bias = 0.0
        self.module20_28 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_729 = P.Reshape()
        self.reshape_729_shape = tuple([1, 256, 1, 1])
        self.split_730 = P.Split(axis=1, output_num=2)
        self.add_733_bias = 0.0
        self.conv2d_735 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_737 = nn.ReLU()
        self.module22_30 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_742 = P.Split(axis=1, output_num=2)
        self.add_743_bias = 0.0
        self.module20_29 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_754 = P.Reshape()
        self.reshape_754_shape = tuple([1, 256, 1, 1])
        self.split_755 = P.Split(axis=1, output_num=2)
        self.add_758_bias = 0.0
        self.conv2d_760 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_762 = nn.ReLU()
        self.module22_31 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_767 = P.Split(axis=1, output_num=2)
        self.add_768_bias = 0.0
        self.module20_30 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_779 = P.Reshape()
        self.reshape_779_shape = tuple([1, 256, 1, 1])
        self.split_780 = P.Split(axis=1, output_num=2)
        self.add_783_bias = 0.0
        self.conv2d_785 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_787 = nn.ReLU()
        self.module22_32 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_792 = P.Split(axis=1, output_num=2)
        self.add_793_bias = 0.0
        self.module20_31 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_804 = P.Reshape()
        self.reshape_804_shape = tuple([1, 256, 1, 1])
        self.split_805 = P.Split(axis=1, output_num=2)
        self.add_808_bias = 0.0
        self.conv2d_810 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_812 = nn.ReLU()
        self.module22_33 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_817 = P.Split(axis=1, output_num=2)
        self.add_818_bias = 0.0
        self.module20_32 = Module20(reshape_2_shape=[1, 256],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=128,
                                    module0_0_conv2d_1_out_channels=64,
                                    module0_0_conv2d_3_in_channels=64,
                                    module0_0_conv2d_3_out_channels=256,
                                    module0_0_reshape_4_shape=[1, 1, 2, 128])
        self.reshape_829 = P.Reshape()
        self.reshape_829_shape = tuple([1, 256, 1, 1])
        self.split_830 = P.Split(axis=1, output_num=2)
        self.add_833_bias = 0.0
        self.conv2d_835 = nn.Conv2d(in_channels=128,
                                    out_channels=512,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_837 = nn.ReLU()
        self.module22_34 = Module22(module5_0_conv2d_0_in_channels=512,
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
        self.split_844 = P.Split(axis=1, output_num=2)
        self.add_845_bias = 0.0
        self.module20_33 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(52, 52),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_856 = P.Reshape()
        self.reshape_856_shape = tuple([1, 512, 1, 1])
        self.split_857 = P.Split(axis=1, output_num=2)
        self.add_860_bias = 0.0
        self.pad_862 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.pad_avgpool2d_863 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_863 = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2d_864 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_839 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_839 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2d_841 = nn.Conv2d(in_channels=512,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_866 = nn.ReLU()
        self.module22_35 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_871 = P.Split(axis=1, output_num=2)
        self.add_872_bias = 0.0
        self.module20_34 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_883 = P.Reshape()
        self.reshape_883_shape = tuple([1, 512, 1, 1])
        self.split_884 = P.Split(axis=1, output_num=2)
        self.add_887_bias = 0.0
        self.conv2d_889 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_891 = nn.ReLU()
        self.module22_36 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_896 = P.Split(axis=1, output_num=2)
        self.add_897_bias = 0.0
        self.module20_35 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_908 = P.Reshape()
        self.reshape_908_shape = tuple([1, 512, 1, 1])
        self.split_909 = P.Split(axis=1, output_num=2)
        self.add_912_bias = 0.0
        self.conv2d_914 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_916 = nn.ReLU()
        self.module22_37 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_921 = P.Split(axis=1, output_num=2)
        self.add_922_bias = 0.0
        self.module20_36 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_933 = P.Reshape()
        self.reshape_933_shape = tuple([1, 512, 1, 1])
        self.split_934 = P.Split(axis=1, output_num=2)
        self.add_937_bias = 0.0
        self.conv2d_939 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_941 = nn.ReLU()
        self.module22_38 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_946 = P.Split(axis=1, output_num=2)
        self.add_947_bias = 0.0
        self.module20_37 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_958 = P.Reshape()
        self.reshape_958_shape = tuple([1, 512, 1, 1])
        self.split_959 = P.Split(axis=1, output_num=2)
        self.add_962_bias = 0.0
        self.conv2d_964 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_966 = nn.ReLU()
        self.module22_39 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_971 = P.Split(axis=1, output_num=2)
        self.add_972_bias = 0.0
        self.module20_38 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_983 = P.Reshape()
        self.reshape_983_shape = tuple([1, 512, 1, 1])
        self.split_984 = P.Split(axis=1, output_num=2)
        self.add_987_bias = 0.0
        self.conv2d_989 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_991 = nn.ReLU()
        self.module22_40 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_996 = P.Split(axis=1, output_num=2)
        self.add_997_bias = 0.0
        self.module20_39 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1008 = P.Reshape()
        self.reshape_1008_shape = tuple([1, 512, 1, 1])
        self.split_1009 = P.Split(axis=1, output_num=2)
        self.add_1012_bias = 0.0
        self.conv2d_1014 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1016 = nn.ReLU()
        self.module22_41 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1021 = P.Split(axis=1, output_num=2)
        self.add_1022_bias = 0.0
        self.module20_40 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1033 = P.Reshape()
        self.reshape_1033_shape = tuple([1, 512, 1, 1])
        self.split_1034 = P.Split(axis=1, output_num=2)
        self.add_1037_bias = 0.0
        self.conv2d_1039 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1041 = nn.ReLU()
        self.module22_42 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1046 = P.Split(axis=1, output_num=2)
        self.add_1047_bias = 0.0
        self.module20_41 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1058 = P.Reshape()
        self.reshape_1058_shape = tuple([1, 512, 1, 1])
        self.split_1059 = P.Split(axis=1, output_num=2)
        self.add_1062_bias = 0.0
        self.conv2d_1064 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1066 = nn.ReLU()
        self.module22_43 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1071 = P.Split(axis=1, output_num=2)
        self.add_1072_bias = 0.0
        self.module20_42 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1083 = P.Reshape()
        self.reshape_1083_shape = tuple([1, 512, 1, 1])
        self.split_1084 = P.Split(axis=1, output_num=2)
        self.add_1087_bias = 0.0
        self.conv2d_1089 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1091 = nn.ReLU()
        self.module22_44 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1096 = P.Split(axis=1, output_num=2)
        self.add_1097_bias = 0.0
        self.module20_43 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1108 = P.Reshape()
        self.reshape_1108_shape = tuple([1, 512, 1, 1])
        self.split_1109 = P.Split(axis=1, output_num=2)
        self.add_1112_bias = 0.0
        self.conv2d_1114 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1116 = nn.ReLU()
        self.module22_45 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1121 = P.Split(axis=1, output_num=2)
        self.add_1122_bias = 0.0
        self.module20_44 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1133 = P.Reshape()
        self.reshape_1133_shape = tuple([1, 512, 1, 1])
        self.split_1134 = P.Split(axis=1, output_num=2)
        self.add_1137_bias = 0.0
        self.conv2d_1139 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1141 = nn.ReLU()
        self.module22_46 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1146 = P.Split(axis=1, output_num=2)
        self.add_1147_bias = 0.0
        self.module20_45 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1158 = P.Reshape()
        self.reshape_1158_shape = tuple([1, 512, 1, 1])
        self.split_1159 = P.Split(axis=1, output_num=2)
        self.add_1162_bias = 0.0
        self.conv2d_1164 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1166 = nn.ReLU()
        self.module22_47 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1171 = P.Split(axis=1, output_num=2)
        self.add_1172_bias = 0.0
        self.module20_46 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1183 = P.Reshape()
        self.reshape_1183_shape = tuple([1, 512, 1, 1])
        self.split_1184 = P.Split(axis=1, output_num=2)
        self.add_1187_bias = 0.0
        self.conv2d_1189 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1191 = nn.ReLU()
        self.module22_48 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1196 = P.Split(axis=1, output_num=2)
        self.add_1197_bias = 0.0
        self.module20_47 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1208 = P.Reshape()
        self.reshape_1208_shape = tuple([1, 512, 1, 1])
        self.split_1209 = P.Split(axis=1, output_num=2)
        self.add_1212_bias = 0.0
        self.conv2d_1214 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1216 = nn.ReLU()
        self.module22_49 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1221 = P.Split(axis=1, output_num=2)
        self.add_1222_bias = 0.0
        self.module20_48 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1233 = P.Reshape()
        self.reshape_1233_shape = tuple([1, 512, 1, 1])
        self.split_1234 = P.Split(axis=1, output_num=2)
        self.add_1237_bias = 0.0
        self.conv2d_1239 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1241 = nn.ReLU()
        self.module22_50 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1246 = P.Split(axis=1, output_num=2)
        self.add_1247_bias = 0.0
        self.module20_49 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1258 = P.Reshape()
        self.reshape_1258_shape = tuple([1, 512, 1, 1])
        self.split_1259 = P.Split(axis=1, output_num=2)
        self.add_1262_bias = 0.0
        self.conv2d_1264 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1266 = nn.ReLU()
        self.module22_51 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1271 = P.Split(axis=1, output_num=2)
        self.add_1272_bias = 0.0
        self.module20_50 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1283 = P.Reshape()
        self.reshape_1283_shape = tuple([1, 512, 1, 1])
        self.split_1284 = P.Split(axis=1, output_num=2)
        self.add_1287_bias = 0.0
        self.conv2d_1289 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1291 = nn.ReLU()
        self.module22_52 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1296 = P.Split(axis=1, output_num=2)
        self.add_1297_bias = 0.0
        self.module20_51 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1308 = P.Reshape()
        self.reshape_1308_shape = tuple([1, 512, 1, 1])
        self.split_1309 = P.Split(axis=1, output_num=2)
        self.add_1312_bias = 0.0
        self.conv2d_1314 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1316 = nn.ReLU()
        self.module22_53 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1321 = P.Split(axis=1, output_num=2)
        self.add_1322_bias = 0.0
        self.module20_52 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1333 = P.Reshape()
        self.reshape_1333_shape = tuple([1, 512, 1, 1])
        self.split_1334 = P.Split(axis=1, output_num=2)
        self.add_1337_bias = 0.0
        self.conv2d_1339 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1341 = nn.ReLU()
        self.module22_54 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1346 = P.Split(axis=1, output_num=2)
        self.add_1347_bias = 0.0
        self.module20_53 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1358 = P.Reshape()
        self.reshape_1358_shape = tuple([1, 512, 1, 1])
        self.split_1359 = P.Split(axis=1, output_num=2)
        self.add_1362_bias = 0.0
        self.conv2d_1364 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1366 = nn.ReLU()
        self.module22_55 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1371 = P.Split(axis=1, output_num=2)
        self.add_1372_bias = 0.0
        self.module20_54 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1383 = P.Reshape()
        self.reshape_1383_shape = tuple([1, 512, 1, 1])
        self.split_1384 = P.Split(axis=1, output_num=2)
        self.add_1387_bias = 0.0
        self.conv2d_1389 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1391 = nn.ReLU()
        self.module22_56 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1396 = P.Split(axis=1, output_num=2)
        self.add_1397_bias = 0.0
        self.module20_55 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1408 = P.Reshape()
        self.reshape_1408_shape = tuple([1, 512, 1, 1])
        self.split_1409 = P.Split(axis=1, output_num=2)
        self.add_1412_bias = 0.0
        self.conv2d_1414 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1416 = nn.ReLU()
        self.module22_57 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1421 = P.Split(axis=1, output_num=2)
        self.add_1422_bias = 0.0
        self.module20_56 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1433 = P.Reshape()
        self.reshape_1433_shape = tuple([1, 512, 1, 1])
        self.split_1434 = P.Split(axis=1, output_num=2)
        self.add_1437_bias = 0.0
        self.conv2d_1439 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1441 = nn.ReLU()
        self.module22_58 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1446 = P.Split(axis=1, output_num=2)
        self.add_1447_bias = 0.0
        self.module20_57 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1458 = P.Reshape()
        self.reshape_1458_shape = tuple([1, 512, 1, 1])
        self.split_1459 = P.Split(axis=1, output_num=2)
        self.add_1462_bias = 0.0
        self.conv2d_1464 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1466 = nn.ReLU()
        self.module22_59 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1471 = P.Split(axis=1, output_num=2)
        self.add_1472_bias = 0.0
        self.module20_58 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1483 = P.Reshape()
        self.reshape_1483_shape = tuple([1, 512, 1, 1])
        self.split_1484 = P.Split(axis=1, output_num=2)
        self.add_1487_bias = 0.0
        self.conv2d_1489 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1491 = nn.ReLU()
        self.module22_60 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1496 = P.Split(axis=1, output_num=2)
        self.add_1497_bias = 0.0
        self.module20_59 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1508 = P.Reshape()
        self.reshape_1508_shape = tuple([1, 512, 1, 1])
        self.split_1509 = P.Split(axis=1, output_num=2)
        self.add_1512_bias = 0.0
        self.conv2d_1514 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1516 = nn.ReLU()
        self.module22_61 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1521 = P.Split(axis=1, output_num=2)
        self.add_1522_bias = 0.0
        self.module20_60 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1533 = P.Reshape()
        self.reshape_1533_shape = tuple([1, 512, 1, 1])
        self.split_1534 = P.Split(axis=1, output_num=2)
        self.add_1537_bias = 0.0
        self.conv2d_1539 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1541 = nn.ReLU()
        self.module22_62 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1546 = P.Split(axis=1, output_num=2)
        self.add_1547_bias = 0.0
        self.module20_61 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1558 = P.Reshape()
        self.reshape_1558_shape = tuple([1, 512, 1, 1])
        self.split_1559 = P.Split(axis=1, output_num=2)
        self.add_1562_bias = 0.0
        self.conv2d_1564 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1566 = nn.ReLU()
        self.module22_63 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1571 = P.Split(axis=1, output_num=2)
        self.add_1572_bias = 0.0
        self.module20_62 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1583 = P.Reshape()
        self.reshape_1583_shape = tuple([1, 512, 1, 1])
        self.split_1584 = P.Split(axis=1, output_num=2)
        self.add_1587_bias = 0.0
        self.conv2d_1589 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1591 = nn.ReLU()
        self.module22_64 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1596 = P.Split(axis=1, output_num=2)
        self.add_1597_bias = 0.0
        self.module20_63 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1608 = P.Reshape()
        self.reshape_1608_shape = tuple([1, 512, 1, 1])
        self.split_1609 = P.Split(axis=1, output_num=2)
        self.add_1612_bias = 0.0
        self.conv2d_1614 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1616 = nn.ReLU()
        self.module22_65 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1621 = P.Split(axis=1, output_num=2)
        self.add_1622_bias = 0.0
        self.module20_64 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1633 = P.Reshape()
        self.reshape_1633_shape = tuple([1, 512, 1, 1])
        self.split_1634 = P.Split(axis=1, output_num=2)
        self.add_1637_bias = 0.0
        self.conv2d_1639 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1641 = nn.ReLU()
        self.module22_66 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1646 = P.Split(axis=1, output_num=2)
        self.add_1647_bias = 0.0
        self.module20_65 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1658 = P.Reshape()
        self.reshape_1658_shape = tuple([1, 512, 1, 1])
        self.split_1659 = P.Split(axis=1, output_num=2)
        self.add_1662_bias = 0.0
        self.conv2d_1664 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1666 = nn.ReLU()
        self.module22_67 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1671 = P.Split(axis=1, output_num=2)
        self.add_1672_bias = 0.0
        self.module20_66 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1683 = P.Reshape()
        self.reshape_1683_shape = tuple([1, 512, 1, 1])
        self.split_1684 = P.Split(axis=1, output_num=2)
        self.add_1687_bias = 0.0
        self.conv2d_1689 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1691 = nn.ReLU()
        self.module22_68 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1696 = P.Split(axis=1, output_num=2)
        self.add_1697_bias = 0.0
        self.module20_67 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1708 = P.Reshape()
        self.reshape_1708_shape = tuple([1, 512, 1, 1])
        self.split_1709 = P.Split(axis=1, output_num=2)
        self.add_1712_bias = 0.0
        self.conv2d_1714 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1716 = nn.ReLU()
        self.module22_69 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1721 = P.Split(axis=1, output_num=2)
        self.add_1722_bias = 0.0
        self.module20_68 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1733 = P.Reshape()
        self.reshape_1733_shape = tuple([1, 512, 1, 1])
        self.split_1734 = P.Split(axis=1, output_num=2)
        self.add_1737_bias = 0.0
        self.conv2d_1739 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1741 = nn.ReLU()
        self.module22_70 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1746 = P.Split(axis=1, output_num=2)
        self.add_1747_bias = 0.0
        self.module20_69 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1758 = P.Reshape()
        self.reshape_1758_shape = tuple([1, 512, 1, 1])
        self.split_1759 = P.Split(axis=1, output_num=2)
        self.add_1762_bias = 0.0
        self.conv2d_1764 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1766 = nn.ReLU()
        self.module22_71 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1771 = P.Split(axis=1, output_num=2)
        self.add_1772_bias = 0.0
        self.module20_70 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1783 = P.Reshape()
        self.reshape_1783_shape = tuple([1, 512, 1, 1])
        self.split_1784 = P.Split(axis=1, output_num=2)
        self.add_1787_bias = 0.0
        self.conv2d_1789 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1791 = nn.ReLU()
        self.module22_72 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1796 = P.Split(axis=1, output_num=2)
        self.add_1797_bias = 0.0
        self.module20_71 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1808 = P.Reshape()
        self.reshape_1808_shape = tuple([1, 512, 1, 1])
        self.split_1809 = P.Split(axis=1, output_num=2)
        self.add_1812_bias = 0.0
        self.conv2d_1814 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1816 = nn.ReLU()
        self.module22_73 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1821 = P.Split(axis=1, output_num=2)
        self.add_1822_bias = 0.0
        self.module20_72 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1833 = P.Reshape()
        self.reshape_1833_shape = tuple([1, 512, 1, 1])
        self.split_1834 = P.Split(axis=1, output_num=2)
        self.add_1837_bias = 0.0
        self.conv2d_1839 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1841 = nn.ReLU()
        self.module22_74 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1846 = P.Split(axis=1, output_num=2)
        self.add_1847_bias = 0.0
        self.module20_73 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1858 = P.Reshape()
        self.reshape_1858_shape = tuple([1, 512, 1, 1])
        self.split_1859 = P.Split(axis=1, output_num=2)
        self.add_1862_bias = 0.0
        self.conv2d_1864 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1866 = nn.ReLU()
        self.module22_75 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1871 = P.Split(axis=1, output_num=2)
        self.add_1872_bias = 0.0
        self.module20_74 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1883 = P.Reshape()
        self.reshape_1883_shape = tuple([1, 512, 1, 1])
        self.split_1884 = P.Split(axis=1, output_num=2)
        self.add_1887_bias = 0.0
        self.conv2d_1889 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1891 = nn.ReLU()
        self.module22_76 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1896 = P.Split(axis=1, output_num=2)
        self.add_1897_bias = 0.0
        self.module20_75 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1908 = P.Reshape()
        self.reshape_1908_shape = tuple([1, 512, 1, 1])
        self.split_1909 = P.Split(axis=1, output_num=2)
        self.add_1912_bias = 0.0
        self.conv2d_1914 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1916 = nn.ReLU()
        self.module22_77 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1921 = P.Split(axis=1, output_num=2)
        self.add_1922_bias = 0.0
        self.module20_76 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1933 = P.Reshape()
        self.reshape_1933_shape = tuple([1, 512, 1, 1])
        self.split_1934 = P.Split(axis=1, output_num=2)
        self.add_1937_bias = 0.0
        self.conv2d_1939 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1941 = nn.ReLU()
        self.module22_78 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1946 = P.Split(axis=1, output_num=2)
        self.add_1947_bias = 0.0
        self.module20_77 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1958 = P.Reshape()
        self.reshape_1958_shape = tuple([1, 512, 1, 1])
        self.split_1959 = P.Split(axis=1, output_num=2)
        self.add_1962_bias = 0.0
        self.conv2d_1964 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1966 = nn.ReLU()
        self.module22_79 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1971 = P.Split(axis=1, output_num=2)
        self.add_1972_bias = 0.0
        self.module20_78 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_1983 = P.Reshape()
        self.reshape_1983_shape = tuple([1, 512, 1, 1])
        self.split_1984 = P.Split(axis=1, output_num=2)
        self.add_1987_bias = 0.0
        self.conv2d_1989 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1991 = nn.ReLU()
        self.module22_80 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_1996 = P.Split(axis=1, output_num=2)
        self.add_1997_bias = 0.0
        self.module20_79 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_2008 = P.Reshape()
        self.reshape_2008_shape = tuple([1, 512, 1, 1])
        self.split_2009 = P.Split(axis=1, output_num=2)
        self.add_2012_bias = 0.0
        self.conv2d_2014 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_2016 = nn.ReLU()
        self.module22_81 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_2021 = P.Split(axis=1, output_num=2)
        self.add_2022_bias = 0.0
        self.module20_80 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_2033 = P.Reshape()
        self.reshape_2033_shape = tuple([1, 512, 1, 1])
        self.split_2034 = P.Split(axis=1, output_num=2)
        self.add_2037_bias = 0.0
        self.conv2d_2039 = nn.Conv2d(in_channels=256,
                                     out_channels=1024,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_2041 = nn.ReLU()
        self.module22_82 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_2048 = P.Split(axis=1, output_num=2)
        self.add_2049_bias = 0.0
        self.module20_81 = Module20(reshape_2_shape=[1, 1024],
                                    module0_0_avgpool2d_0_kernel_size=(26, 26),
                                    module0_0_conv2d_1_in_channels=512,
                                    module0_0_conv2d_1_out_channels=256,
                                    module0_0_conv2d_3_in_channels=256,
                                    module0_0_conv2d_3_out_channels=1024,
                                    module0_0_reshape_4_shape=[1, 1, 2, 512])
        self.reshape_2060 = P.Reshape()
        self.reshape_2060_shape = tuple([1, 1024, 1, 1])
        self.split_2061 = P.Split(axis=1, output_num=2)
        self.add_2064_bias = 0.0
        self.pad_2066 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.pad_avgpool2d_2067 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_2067 = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2d_2068 = nn.Conv2d(in_channels=512,
                                     out_channels=2048,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.pad_avgpool2d_2043 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_2043 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2d_2045 = nn.Conv2d(in_channels=1024,
                                     out_channels=2048,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_2070 = nn.ReLU()
        self.module22_83 = Module22(module5_0_conv2d_0_in_channels=2048,
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
        self.split_2075 = P.Split(axis=1, output_num=2)
        self.add_2076_bias = 0.0
        self.module20_82 = Module20(reshape_2_shape=[1, 1024],
                                    module0_0_avgpool2d_0_kernel_size=(13, 13),
                                    module0_0_conv2d_1_in_channels=512,
                                    module0_0_conv2d_1_out_channels=256,
                                    module0_0_conv2d_3_in_channels=256,
                                    module0_0_conv2d_3_out_channels=1024,
                                    module0_0_reshape_4_shape=[1, 1, 2, 512])
        self.reshape_2087 = P.Reshape()
        self.reshape_2087_shape = tuple([1, 1024, 1, 1])
        self.split_2088 = P.Split(axis=1, output_num=2)
        self.add_2091_bias = 0.0
        self.conv2d_2093 = nn.Conv2d(in_channels=512,
                                     out_channels=2048,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_2095 = nn.ReLU()
        self.module22_84 = Module22(module5_0_conv2d_0_in_channels=2048,
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
        self.split_2100 = P.Split(axis=1, output_num=2)
        self.add_2101_bias = 0.0
        self.module20_83 = Module20(reshape_2_shape=[1, 1024],
                                    module0_0_avgpool2d_0_kernel_size=(13, 13),
                                    module0_0_conv2d_1_in_channels=512,
                                    module0_0_conv2d_1_out_channels=256,
                                    module0_0_conv2d_3_in_channels=256,
                                    module0_0_conv2d_3_out_channels=1024,
                                    module0_0_reshape_4_shape=[1, 1, 2, 512])
        self.reshape_2112 = P.Reshape()
        self.reshape_2112_shape = tuple([1, 1024, 1, 1])
        self.split_2113 = P.Split(axis=1, output_num=2)
        self.add_2116_bias = 0.0
        self.conv2d_2118 = nn.Conv2d(in_channels=512,
                                     out_channels=2048,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_2120 = nn.ReLU()
        self.module22_85 = Module22(module5_0_conv2d_0_in_channels=2048,
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
        self.split_2125 = P.Split(axis=1, output_num=2)
        self.add_2126_bias = 0.0
        self.module20_84 = Module20(reshape_2_shape=[1, 1024],
                                    module0_0_avgpool2d_0_kernel_size=(13, 13),
                                    module0_0_conv2d_1_in_channels=512,
                                    module0_0_conv2d_1_out_channels=256,
                                    module0_0_conv2d_3_in_channels=256,
                                    module0_0_conv2d_3_out_channels=1024,
                                    module0_0_reshape_4_shape=[1, 1, 2, 512])
        self.reshape_2137 = P.Reshape()
        self.reshape_2137_shape = tuple([1, 1024, 1, 1])
        self.split_2138 = P.Split(axis=1, output_num=2)
        self.add_2141_bias = 0.0
        self.conv2d_2143 = nn.Conv2d(in_channels=512,
                                     out_channels=2048,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_2145 = nn.ReLU()
        self.module22_86 = Module22(module5_0_conv2d_0_in_channels=2048,
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
        self.split_2150 = P.Split(axis=1, output_num=2)
        self.add_2151_bias = 0.0
        self.module20_85 = Module20(reshape_2_shape=[1, 1024],
                                    module0_0_avgpool2d_0_kernel_size=(13, 13),
                                    module0_0_conv2d_1_in_channels=512,
                                    module0_0_conv2d_1_out_channels=256,
                                    module0_0_conv2d_3_in_channels=256,
                                    module0_0_conv2d_3_out_channels=1024,
                                    module0_0_reshape_4_shape=[1, 1, 2, 512])
        self.reshape_2162 = P.Reshape()
        self.reshape_2162_shape = tuple([1, 1024, 1, 1])
        self.split_2163 = P.Split(axis=1, output_num=2)
        self.add_2166_bias = 0.0
        self.conv2d_2168 = nn.Conv2d(in_channels=512,
                                     out_channels=2048,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_2170 = nn.ReLU()
        self.module22_87 = Module22(module5_0_conv2d_0_in_channels=2048,
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
        self.split_2175 = P.Split(axis=1, output_num=2)
        self.add_2176_bias = 0.0
        self.module20_86 = Module20(reshape_2_shape=[1, 1024],
                                    module0_0_avgpool2d_0_kernel_size=(13, 13),
                                    module0_0_conv2d_1_in_channels=512,
                                    module0_0_conv2d_1_out_channels=256,
                                    module0_0_conv2d_3_in_channels=256,
                                    module0_0_conv2d_3_out_channels=1024,
                                    module0_0_reshape_4_shape=[1, 1, 2, 512])
        self.reshape_2187 = P.Reshape()
        self.reshape_2187_shape = tuple([1, 1024, 1, 1])
        self.split_2188 = P.Split(axis=1, output_num=2)
        self.add_2191_bias = 0.0
        self.conv2d_2193 = nn.Conv2d(in_channels=512,
                                     out_channels=2048,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_2195 = nn.ReLU()
        self.module22_88 = Module22(module5_0_conv2d_0_in_channels=2048,
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
        self.split_2200 = P.Split(axis=1, output_num=2)
        self.add_2201_bias = 0.0
        self.module20_87 = Module20(reshape_2_shape=[1, 1024],
                                    module0_0_avgpool2d_0_kernel_size=(13, 13),
                                    module0_0_conv2d_1_in_channels=512,
                                    module0_0_conv2d_1_out_channels=256,
                                    module0_0_conv2d_3_in_channels=256,
                                    module0_0_conv2d_3_out_channels=1024,
                                    module0_0_reshape_4_shape=[1, 1, 2, 512])
        self.reshape_2212 = P.Reshape()
        self.reshape_2212_shape = tuple([1, 1024, 1, 1])
        self.split_2213 = P.Split(axis=1, output_num=2)
        self.add_2216_bias = 0.0
        self.conv2d_2218 = nn.Conv2d(in_channels=512,
                                     out_channels=2048,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_2220 = nn.ReLU()
        self.module22_89 = Module22(module5_0_conv2d_0_in_channels=2048,
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
        self.split_2225 = P.Split(axis=1, output_num=2)
        self.add_2226_bias = 0.0
        self.module20_88 = Module20(reshape_2_shape=[1, 1024],
                                    module0_0_avgpool2d_0_kernel_size=(13, 13),
                                    module0_0_conv2d_1_in_channels=512,
                                    module0_0_conv2d_1_out_channels=256,
                                    module0_0_conv2d_3_in_channels=256,
                                    module0_0_conv2d_3_out_channels=1024,
                                    module0_0_reshape_4_shape=[1, 1, 2, 512])
        self.reshape_2237 = P.Reshape()
        self.reshape_2237_shape = tuple([1, 1024, 1, 1])
        self.split_2238 = P.Split(axis=1, output_num=2)
        self.add_2241_bias = 0.0
        self.conv2d_2243 = nn.Conv2d(in_channels=512,
                                     out_channels=2048,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_2245 = nn.ReLU()
        self.avgpool2d_2246 = nn.AvgPool2d(kernel_size=(13, 13))
        self.reshape_2247 = P.Reshape()
        self.reshape_2247_shape = tuple([1, 2048])
        self.flatten_2248 = nn.Flatten()
        self.dense_2249 = nn.Dense(in_channels=2048, out_channels=1000, has_bias=True)

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
        opt_split_192, opt_split_192_1 = self.split_192(module22_8_opt)
        opt_add_193 = opt_split_192 + self.add_193_bias
        opt_add_194 = P.Add()(opt_add_193, opt_split_192_1)
        module20_7_opt = self.module20_7(opt_add_194)
        opt_reshape_204 = self.reshape_204(module20_7_opt, self.reshape_204_shape)
        opt_split_205, opt_split_205_1 = self.split_205(opt_reshape_204)
        opt_mul_206 = P.Mul()(opt_split_205, opt_split_192)
        opt_mul_207 = P.Mul()(opt_split_205_1, opt_split_192_1)
        opt_add_208 = opt_mul_206 + self.add_208_bias
        opt_add_209 = P.Add()(opt_add_208, opt_mul_207)
        opt_conv2d_210 = self.conv2d_210(opt_add_209)
        opt_add_211 = P.Add()(opt_conv2d_210, opt_relu_187)
        opt_relu_212 = self.relu_212(opt_add_211)
        module22_9_opt = self.module22_9(opt_relu_212)
        opt_split_217, opt_split_217_1 = self.split_217(module22_9_opt)
        opt_add_218 = opt_split_217 + self.add_218_bias
        opt_add_219 = P.Add()(opt_add_218, opt_split_217_1)
        module20_8_opt = self.module20_8(opt_add_219)
        opt_reshape_229 = self.reshape_229(module20_8_opt, self.reshape_229_shape)
        opt_split_230, opt_split_230_1 = self.split_230(opt_reshape_229)
        opt_mul_231 = P.Mul()(opt_split_230, opt_split_217)
        opt_mul_232 = P.Mul()(opt_split_230_1, opt_split_217_1)
        opt_add_233 = opt_mul_231 + self.add_233_bias
        opt_add_234 = P.Add()(opt_add_233, opt_mul_232)
        opt_conv2d_235 = self.conv2d_235(opt_add_234)
        opt_add_236 = P.Add()(opt_conv2d_235, opt_relu_212)
        opt_relu_237 = self.relu_237(opt_add_236)
        module22_10_opt = self.module22_10(opt_relu_237)
        opt_split_242, opt_split_242_1 = self.split_242(module22_10_opt)
        opt_add_243 = opt_split_242 + self.add_243_bias
        opt_add_244 = P.Add()(opt_add_243, opt_split_242_1)
        module20_9_opt = self.module20_9(opt_add_244)
        opt_reshape_254 = self.reshape_254(module20_9_opt, self.reshape_254_shape)
        opt_split_255, opt_split_255_1 = self.split_255(opt_reshape_254)
        opt_mul_256 = P.Mul()(opt_split_255, opt_split_242)
        opt_mul_257 = P.Mul()(opt_split_255_1, opt_split_242_1)
        opt_add_258 = opt_mul_256 + self.add_258_bias
        opt_add_259 = P.Add()(opt_add_258, opt_mul_257)
        opt_conv2d_260 = self.conv2d_260(opt_add_259)
        opt_add_261 = P.Add()(opt_conv2d_260, opt_relu_237)
        opt_relu_262 = self.relu_262(opt_add_261)
        module22_11_opt = self.module22_11(opt_relu_262)
        opt_split_267, opt_split_267_1 = self.split_267(module22_11_opt)
        opt_add_268 = opt_split_267 + self.add_268_bias
        opt_add_269 = P.Add()(opt_add_268, opt_split_267_1)
        module20_10_opt = self.module20_10(opt_add_269)
        opt_reshape_279 = self.reshape_279(module20_10_opt, self.reshape_279_shape)
        opt_split_280, opt_split_280_1 = self.split_280(opt_reshape_279)
        opt_mul_281 = P.Mul()(opt_split_280, opt_split_267)
        opt_mul_282 = P.Mul()(opt_split_280_1, opt_split_267_1)
        opt_add_283 = opt_mul_281 + self.add_283_bias
        opt_add_284 = P.Add()(opt_add_283, opt_mul_282)
        opt_conv2d_285 = self.conv2d_285(opt_add_284)
        opt_add_286 = P.Add()(opt_conv2d_285, opt_relu_262)
        opt_relu_287 = self.relu_287(opt_add_286)
        module22_12_opt = self.module22_12(opt_relu_287)
        opt_split_292, opt_split_292_1 = self.split_292(module22_12_opt)
        opt_add_293 = opt_split_292 + self.add_293_bias
        opt_add_294 = P.Add()(opt_add_293, opt_split_292_1)
        module20_11_opt = self.module20_11(opt_add_294)
        opt_reshape_304 = self.reshape_304(module20_11_opt, self.reshape_304_shape)
        opt_split_305, opt_split_305_1 = self.split_305(opt_reshape_304)
        opt_mul_306 = P.Mul()(opt_split_305, opt_split_292)
        opt_mul_307 = P.Mul()(opt_split_305_1, opt_split_292_1)
        opt_add_308 = opt_mul_306 + self.add_308_bias
        opt_add_309 = P.Add()(opt_add_308, opt_mul_307)
        opt_conv2d_310 = self.conv2d_310(opt_add_309)
        opt_add_311 = P.Add()(opt_conv2d_310, opt_relu_287)
        opt_relu_312 = self.relu_312(opt_add_311)
        module22_13_opt = self.module22_13(opt_relu_312)
        opt_split_317, opt_split_317_1 = self.split_317(module22_13_opt)
        opt_add_318 = opt_split_317 + self.add_318_bias
        opt_add_319 = P.Add()(opt_add_318, opt_split_317_1)
        module20_12_opt = self.module20_12(opt_add_319)
        opt_reshape_329 = self.reshape_329(module20_12_opt, self.reshape_329_shape)
        opt_split_330, opt_split_330_1 = self.split_330(opt_reshape_329)
        opt_mul_331 = P.Mul()(opt_split_330, opt_split_317)
        opt_mul_332 = P.Mul()(opt_split_330_1, opt_split_317_1)
        opt_add_333 = opt_mul_331 + self.add_333_bias
        opt_add_334 = P.Add()(opt_add_333, opt_mul_332)
        opt_conv2d_335 = self.conv2d_335(opt_add_334)
        opt_add_336 = P.Add()(opt_conv2d_335, opt_relu_312)
        opt_relu_337 = self.relu_337(opt_add_336)
        module22_14_opt = self.module22_14(opt_relu_337)
        opt_split_342, opt_split_342_1 = self.split_342(module22_14_opt)
        opt_add_343 = opt_split_342 + self.add_343_bias
        opt_add_344 = P.Add()(opt_add_343, opt_split_342_1)
        module20_13_opt = self.module20_13(opt_add_344)
        opt_reshape_354 = self.reshape_354(module20_13_opt, self.reshape_354_shape)
        opt_split_355, opt_split_355_1 = self.split_355(opt_reshape_354)
        opt_mul_356 = P.Mul()(opt_split_355, opt_split_342)
        opt_mul_357 = P.Mul()(opt_split_355_1, opt_split_342_1)
        opt_add_358 = opt_mul_356 + self.add_358_bias
        opt_add_359 = P.Add()(opt_add_358, opt_mul_357)
        opt_conv2d_360 = self.conv2d_360(opt_add_359)
        opt_add_361 = P.Add()(opt_conv2d_360, opt_relu_337)
        opt_relu_362 = self.relu_362(opt_add_361)
        module22_15_opt = self.module22_15(opt_relu_362)
        opt_split_367, opt_split_367_1 = self.split_367(module22_15_opt)
        opt_add_368 = opt_split_367 + self.add_368_bias
        opt_add_369 = P.Add()(opt_add_368, opt_split_367_1)
        module20_14_opt = self.module20_14(opt_add_369)
        opt_reshape_379 = self.reshape_379(module20_14_opt, self.reshape_379_shape)
        opt_split_380, opt_split_380_1 = self.split_380(opt_reshape_379)
        opt_mul_381 = P.Mul()(opt_split_380, opt_split_367)
        opt_mul_382 = P.Mul()(opt_split_380_1, opt_split_367_1)
        opt_add_383 = opt_mul_381 + self.add_383_bias
        opt_add_384 = P.Add()(opt_add_383, opt_mul_382)
        opt_conv2d_385 = self.conv2d_385(opt_add_384)
        opt_add_386 = P.Add()(opt_conv2d_385, opt_relu_362)
        opt_relu_387 = self.relu_387(opt_add_386)
        module22_16_opt = self.module22_16(opt_relu_387)
        opt_split_392, opt_split_392_1 = self.split_392(module22_16_opt)
        opt_add_393 = opt_split_392 + self.add_393_bias
        opt_add_394 = P.Add()(opt_add_393, opt_split_392_1)
        module20_15_opt = self.module20_15(opt_add_394)
        opt_reshape_404 = self.reshape_404(module20_15_opt, self.reshape_404_shape)
        opt_split_405, opt_split_405_1 = self.split_405(opt_reshape_404)
        opt_mul_406 = P.Mul()(opt_split_405, opt_split_392)
        opt_mul_407 = P.Mul()(opt_split_405_1, opt_split_392_1)
        opt_add_408 = opt_mul_406 + self.add_408_bias
        opt_add_409 = P.Add()(opt_add_408, opt_mul_407)
        opt_conv2d_410 = self.conv2d_410(opt_add_409)
        opt_add_411 = P.Add()(opt_conv2d_410, opt_relu_387)
        opt_relu_412 = self.relu_412(opt_add_411)
        module22_17_opt = self.module22_17(opt_relu_412)
        opt_split_417, opt_split_417_1 = self.split_417(module22_17_opt)
        opt_add_418 = opt_split_417 + self.add_418_bias
        opt_add_419 = P.Add()(opt_add_418, opt_split_417_1)
        module20_16_opt = self.module20_16(opt_add_419)
        opt_reshape_429 = self.reshape_429(module20_16_opt, self.reshape_429_shape)
        opt_split_430, opt_split_430_1 = self.split_430(opt_reshape_429)
        opt_mul_431 = P.Mul()(opt_split_430, opt_split_417)
        opt_mul_432 = P.Mul()(opt_split_430_1, opt_split_417_1)
        opt_add_433 = opt_mul_431 + self.add_433_bias
        opt_add_434 = P.Add()(opt_add_433, opt_mul_432)
        opt_conv2d_435 = self.conv2d_435(opt_add_434)
        opt_add_436 = P.Add()(opt_conv2d_435, opt_relu_412)
        opt_relu_437 = self.relu_437(opt_add_436)
        module22_18_opt = self.module22_18(opt_relu_437)
        opt_split_442, opt_split_442_1 = self.split_442(module22_18_opt)
        opt_add_443 = opt_split_442 + self.add_443_bias
        opt_add_444 = P.Add()(opt_add_443, opt_split_442_1)
        module20_17_opt = self.module20_17(opt_add_444)
        opt_reshape_454 = self.reshape_454(module20_17_opt, self.reshape_454_shape)
        opt_split_455, opt_split_455_1 = self.split_455(opt_reshape_454)
        opt_mul_456 = P.Mul()(opt_split_455, opt_split_442)
        opt_mul_457 = P.Mul()(opt_split_455_1, opt_split_442_1)
        opt_add_458 = opt_mul_456 + self.add_458_bias
        opt_add_459 = P.Add()(opt_add_458, opt_mul_457)
        opt_conv2d_460 = self.conv2d_460(opt_add_459)
        opt_add_461 = P.Add()(opt_conv2d_460, opt_relu_437)
        opt_relu_462 = self.relu_462(opt_add_461)
        module22_19_opt = self.module22_19(opt_relu_462)
        opt_split_467, opt_split_467_1 = self.split_467(module22_19_opt)
        opt_add_468 = opt_split_467 + self.add_468_bias
        opt_add_469 = P.Add()(opt_add_468, opt_split_467_1)
        module20_18_opt = self.module20_18(opt_add_469)
        opt_reshape_479 = self.reshape_479(module20_18_opt, self.reshape_479_shape)
        opt_split_480, opt_split_480_1 = self.split_480(opt_reshape_479)
        opt_mul_481 = P.Mul()(opt_split_480, opt_split_467)
        opt_mul_482 = P.Mul()(opt_split_480_1, opt_split_467_1)
        opt_add_483 = opt_mul_481 + self.add_483_bias
        opt_add_484 = P.Add()(opt_add_483, opt_mul_482)
        opt_conv2d_485 = self.conv2d_485(opt_add_484)
        opt_add_486 = P.Add()(opt_conv2d_485, opt_relu_462)
        opt_relu_487 = self.relu_487(opt_add_486)
        module22_20_opt = self.module22_20(opt_relu_487)
        opt_split_492, opt_split_492_1 = self.split_492(module22_20_opt)
        opt_add_493 = opt_split_492 + self.add_493_bias
        opt_add_494 = P.Add()(opt_add_493, opt_split_492_1)
        module20_19_opt = self.module20_19(opt_add_494)
        opt_reshape_504 = self.reshape_504(module20_19_opt, self.reshape_504_shape)
        opt_split_505, opt_split_505_1 = self.split_505(opt_reshape_504)
        opt_mul_506 = P.Mul()(opt_split_505, opt_split_492)
        opt_mul_507 = P.Mul()(opt_split_505_1, opt_split_492_1)
        opt_add_508 = opt_mul_506 + self.add_508_bias
        opt_add_509 = P.Add()(opt_add_508, opt_mul_507)
        opt_conv2d_510 = self.conv2d_510(opt_add_509)
        opt_add_511 = P.Add()(opt_conv2d_510, opt_relu_487)
        opt_relu_512 = self.relu_512(opt_add_511)
        module22_21_opt = self.module22_21(opt_relu_512)
        opt_split_517, opt_split_517_1 = self.split_517(module22_21_opt)
        opt_add_518 = opt_split_517 + self.add_518_bias
        opt_add_519 = P.Add()(opt_add_518, opt_split_517_1)
        module20_20_opt = self.module20_20(opt_add_519)
        opt_reshape_529 = self.reshape_529(module20_20_opt, self.reshape_529_shape)
        opt_split_530, opt_split_530_1 = self.split_530(opt_reshape_529)
        opt_mul_531 = P.Mul()(opt_split_530, opt_split_517)
        opt_mul_532 = P.Mul()(opt_split_530_1, opt_split_517_1)
        opt_add_533 = opt_mul_531 + self.add_533_bias
        opt_add_534 = P.Add()(opt_add_533, opt_mul_532)
        opt_conv2d_535 = self.conv2d_535(opt_add_534)
        opt_add_536 = P.Add()(opt_conv2d_535, opt_relu_512)
        opt_relu_537 = self.relu_537(opt_add_536)
        module22_22_opt = self.module22_22(opt_relu_537)
        opt_split_542, opt_split_542_1 = self.split_542(module22_22_opt)
        opt_add_543 = opt_split_542 + self.add_543_bias
        opt_add_544 = P.Add()(opt_add_543, opt_split_542_1)
        module20_21_opt = self.module20_21(opt_add_544)
        opt_reshape_554 = self.reshape_554(module20_21_opt, self.reshape_554_shape)
        opt_split_555, opt_split_555_1 = self.split_555(opt_reshape_554)
        opt_mul_556 = P.Mul()(opt_split_555, opt_split_542)
        opt_mul_557 = P.Mul()(opt_split_555_1, opt_split_542_1)
        opt_add_558 = opt_mul_556 + self.add_558_bias
        opt_add_559 = P.Add()(opt_add_558, opt_mul_557)
        opt_conv2d_560 = self.conv2d_560(opt_add_559)
        opt_add_561 = P.Add()(opt_conv2d_560, opt_relu_537)
        opt_relu_562 = self.relu_562(opt_add_561)
        module22_23_opt = self.module22_23(opt_relu_562)
        opt_split_567, opt_split_567_1 = self.split_567(module22_23_opt)
        opt_add_568 = opt_split_567 + self.add_568_bias
        opt_add_569 = P.Add()(opt_add_568, opt_split_567_1)
        module20_22_opt = self.module20_22(opt_add_569)
        opt_reshape_579 = self.reshape_579(module20_22_opt, self.reshape_579_shape)
        opt_split_580, opt_split_580_1 = self.split_580(opt_reshape_579)
        opt_mul_581 = P.Mul()(opt_split_580, opt_split_567)
        opt_mul_582 = P.Mul()(opt_split_580_1, opt_split_567_1)
        opt_add_583 = opt_mul_581 + self.add_583_bias
        opt_add_584 = P.Add()(opt_add_583, opt_mul_582)
        opt_conv2d_585 = self.conv2d_585(opt_add_584)
        opt_add_586 = P.Add()(opt_conv2d_585, opt_relu_562)
        opt_relu_587 = self.relu_587(opt_add_586)
        module22_24_opt = self.module22_24(opt_relu_587)
        opt_split_592, opt_split_592_1 = self.split_592(module22_24_opt)
        opt_add_593 = opt_split_592 + self.add_593_bias
        opt_add_594 = P.Add()(opt_add_593, opt_split_592_1)
        module20_23_opt = self.module20_23(opt_add_594)
        opt_reshape_604 = self.reshape_604(module20_23_opt, self.reshape_604_shape)
        opt_split_605, opt_split_605_1 = self.split_605(opt_reshape_604)
        opt_mul_606 = P.Mul()(opt_split_605, opt_split_592)
        opt_mul_607 = P.Mul()(opt_split_605_1, opt_split_592_1)
        opt_add_608 = opt_mul_606 + self.add_608_bias
        opt_add_609 = P.Add()(opt_add_608, opt_mul_607)
        opt_conv2d_610 = self.conv2d_610(opt_add_609)
        opt_add_611 = P.Add()(opt_conv2d_610, opt_relu_587)
        opt_relu_612 = self.relu_612(opt_add_611)
        module22_25_opt = self.module22_25(opt_relu_612)
        opt_split_617, opt_split_617_1 = self.split_617(module22_25_opt)
        opt_add_618 = opt_split_617 + self.add_618_bias
        opt_add_619 = P.Add()(opt_add_618, opt_split_617_1)
        module20_24_opt = self.module20_24(opt_add_619)
        opt_reshape_629 = self.reshape_629(module20_24_opt, self.reshape_629_shape)
        opt_split_630, opt_split_630_1 = self.split_630(opt_reshape_629)
        opt_mul_631 = P.Mul()(opt_split_630, opt_split_617)
        opt_mul_632 = P.Mul()(opt_split_630_1, opt_split_617_1)
        opt_add_633 = opt_mul_631 + self.add_633_bias
        opt_add_634 = P.Add()(opt_add_633, opt_mul_632)
        opt_conv2d_635 = self.conv2d_635(opt_add_634)
        opt_add_636 = P.Add()(opt_conv2d_635, opt_relu_612)
        opt_relu_637 = self.relu_637(opt_add_636)
        module22_26_opt = self.module22_26(opt_relu_637)
        opt_split_642, opt_split_642_1 = self.split_642(module22_26_opt)
        opt_add_643 = opt_split_642 + self.add_643_bias
        opt_add_644 = P.Add()(opt_add_643, opt_split_642_1)
        module20_25_opt = self.module20_25(opt_add_644)
        opt_reshape_654 = self.reshape_654(module20_25_opt, self.reshape_654_shape)
        opt_split_655, opt_split_655_1 = self.split_655(opt_reshape_654)
        opt_mul_656 = P.Mul()(opt_split_655, opt_split_642)
        opt_mul_657 = P.Mul()(opt_split_655_1, opt_split_642_1)
        opt_add_658 = opt_mul_656 + self.add_658_bias
        opt_add_659 = P.Add()(opt_add_658, opt_mul_657)
        opt_conv2d_660 = self.conv2d_660(opt_add_659)
        opt_add_661 = P.Add()(opt_conv2d_660, opt_relu_637)
        opt_relu_662 = self.relu_662(opt_add_661)
        module22_27_opt = self.module22_27(opt_relu_662)
        opt_split_667, opt_split_667_1 = self.split_667(module22_27_opt)
        opt_add_668 = opt_split_667 + self.add_668_bias
        opt_add_669 = P.Add()(opt_add_668, opt_split_667_1)
        module20_26_opt = self.module20_26(opt_add_669)
        opt_reshape_679 = self.reshape_679(module20_26_opt, self.reshape_679_shape)
        opt_split_680, opt_split_680_1 = self.split_680(opt_reshape_679)
        opt_mul_681 = P.Mul()(opt_split_680, opt_split_667)
        opt_mul_682 = P.Mul()(opt_split_680_1, opt_split_667_1)
        opt_add_683 = opt_mul_681 + self.add_683_bias
        opt_add_684 = P.Add()(opt_add_683, opt_mul_682)
        opt_conv2d_685 = self.conv2d_685(opt_add_684)
        opt_add_686 = P.Add()(opt_conv2d_685, opt_relu_662)
        opt_relu_687 = self.relu_687(opt_add_686)
        module22_28_opt = self.module22_28(opt_relu_687)
        opt_split_692, opt_split_692_1 = self.split_692(module22_28_opt)
        opt_add_693 = opt_split_692 + self.add_693_bias
        opt_add_694 = P.Add()(opt_add_693, opt_split_692_1)
        module20_27_opt = self.module20_27(opt_add_694)
        opt_reshape_704 = self.reshape_704(module20_27_opt, self.reshape_704_shape)
        opt_split_705, opt_split_705_1 = self.split_705(opt_reshape_704)
        opt_mul_706 = P.Mul()(opt_split_705, opt_split_692)
        opt_mul_707 = P.Mul()(opt_split_705_1, opt_split_692_1)
        opt_add_708 = opt_mul_706 + self.add_708_bias
        opt_add_709 = P.Add()(opt_add_708, opt_mul_707)
        opt_conv2d_710 = self.conv2d_710(opt_add_709)
        opt_add_711 = P.Add()(opt_conv2d_710, opt_relu_687)
        opt_relu_712 = self.relu_712(opt_add_711)
        module22_29_opt = self.module22_29(opt_relu_712)
        opt_split_717, opt_split_717_1 = self.split_717(module22_29_opt)
        opt_add_718 = opt_split_717 + self.add_718_bias
        opt_add_719 = P.Add()(opt_add_718, opt_split_717_1)
        module20_28_opt = self.module20_28(opt_add_719)
        opt_reshape_729 = self.reshape_729(module20_28_opt, self.reshape_729_shape)
        opt_split_730, opt_split_730_1 = self.split_730(opt_reshape_729)
        opt_mul_731 = P.Mul()(opt_split_730, opt_split_717)
        opt_mul_732 = P.Mul()(opt_split_730_1, opt_split_717_1)
        opt_add_733 = opt_mul_731 + self.add_733_bias
        opt_add_734 = P.Add()(opt_add_733, opt_mul_732)
        opt_conv2d_735 = self.conv2d_735(opt_add_734)
        opt_add_736 = P.Add()(opt_conv2d_735, opt_relu_712)
        opt_relu_737 = self.relu_737(opt_add_736)
        module22_30_opt = self.module22_30(opt_relu_737)
        opt_split_742, opt_split_742_1 = self.split_742(module22_30_opt)
        opt_add_743 = opt_split_742 + self.add_743_bias
        opt_add_744 = P.Add()(opt_add_743, opt_split_742_1)
        module20_29_opt = self.module20_29(opt_add_744)
        opt_reshape_754 = self.reshape_754(module20_29_opt, self.reshape_754_shape)
        opt_split_755, opt_split_755_1 = self.split_755(opt_reshape_754)
        opt_mul_756 = P.Mul()(opt_split_755, opt_split_742)
        opt_mul_757 = P.Mul()(opt_split_755_1, opt_split_742_1)
        opt_add_758 = opt_mul_756 + self.add_758_bias
        opt_add_759 = P.Add()(opt_add_758, opt_mul_757)
        opt_conv2d_760 = self.conv2d_760(opt_add_759)
        opt_add_761 = P.Add()(opt_conv2d_760, opt_relu_737)
        opt_relu_762 = self.relu_762(opt_add_761)
        module22_31_opt = self.module22_31(opt_relu_762)
        opt_split_767, opt_split_767_1 = self.split_767(module22_31_opt)
        opt_add_768 = opt_split_767 + self.add_768_bias
        opt_add_769 = P.Add()(opt_add_768, opt_split_767_1)
        module20_30_opt = self.module20_30(opt_add_769)
        opt_reshape_779 = self.reshape_779(module20_30_opt, self.reshape_779_shape)
        opt_split_780, opt_split_780_1 = self.split_780(opt_reshape_779)
        opt_mul_781 = P.Mul()(opt_split_780, opt_split_767)
        opt_mul_782 = P.Mul()(opt_split_780_1, opt_split_767_1)
        opt_add_783 = opt_mul_781 + self.add_783_bias
        opt_add_784 = P.Add()(opt_add_783, opt_mul_782)
        opt_conv2d_785 = self.conv2d_785(opt_add_784)
        opt_add_786 = P.Add()(opt_conv2d_785, opt_relu_762)
        opt_relu_787 = self.relu_787(opt_add_786)
        module22_32_opt = self.module22_32(opt_relu_787)
        opt_split_792, opt_split_792_1 = self.split_792(module22_32_opt)
        opt_add_793 = opt_split_792 + self.add_793_bias
        opt_add_794 = P.Add()(opt_add_793, opt_split_792_1)
        module20_31_opt = self.module20_31(opt_add_794)
        opt_reshape_804 = self.reshape_804(module20_31_opt, self.reshape_804_shape)
        opt_split_805, opt_split_805_1 = self.split_805(opt_reshape_804)
        opt_mul_806 = P.Mul()(opt_split_805, opt_split_792)
        opt_mul_807 = P.Mul()(opt_split_805_1, opt_split_792_1)
        opt_add_808 = opt_mul_806 + self.add_808_bias
        opt_add_809 = P.Add()(opt_add_808, opt_mul_807)
        opt_conv2d_810 = self.conv2d_810(opt_add_809)
        opt_add_811 = P.Add()(opt_conv2d_810, opt_relu_787)
        opt_relu_812 = self.relu_812(opt_add_811)
        module22_33_opt = self.module22_33(opt_relu_812)
        opt_split_817, opt_split_817_1 = self.split_817(module22_33_opt)
        opt_add_818 = opt_split_817 + self.add_818_bias
        opt_add_819 = P.Add()(opt_add_818, opt_split_817_1)
        module20_32_opt = self.module20_32(opt_add_819)
        opt_reshape_829 = self.reshape_829(module20_32_opt, self.reshape_829_shape)
        opt_split_830, opt_split_830_1 = self.split_830(opt_reshape_829)
        opt_mul_831 = P.Mul()(opt_split_830, opt_split_817)
        opt_mul_832 = P.Mul()(opt_split_830_1, opt_split_817_1)
        opt_add_833 = opt_mul_831 + self.add_833_bias
        opt_add_834 = P.Add()(opt_add_833, opt_mul_832)
        opt_conv2d_835 = self.conv2d_835(opt_add_834)
        opt_add_836 = P.Add()(opt_conv2d_835, opt_relu_812)
        opt_relu_837 = self.relu_837(opt_add_836)
        module22_34_opt = self.module22_34(opt_relu_837)
        opt_split_844, opt_split_844_1 = self.split_844(module22_34_opt)
        opt_add_845 = opt_split_844 + self.add_845_bias
        opt_add_846 = P.Add()(opt_add_845, opt_split_844_1)
        module20_33_opt = self.module20_33(opt_add_846)
        opt_reshape_856 = self.reshape_856(module20_33_opt, self.reshape_856_shape)
        opt_split_857, opt_split_857_1 = self.split_857(opt_reshape_856)
        opt_mul_858 = P.Mul()(opt_split_857, opt_split_844)
        opt_mul_859 = P.Mul()(opt_split_857_1, opt_split_844_1)
        opt_add_860 = opt_mul_858 + self.add_860_bias
        opt_add_861 = P.Add()(opt_add_860, opt_mul_859)
        opt_pad_862 = self.pad_862(opt_add_861)
        opt_avgpool2d_863 = self.pad_avgpool2d_863(opt_pad_862)
        opt_avgpool2d_863 = self.avgpool2d_863(opt_avgpool2d_863)
        opt_conv2d_864 = self.conv2d_864(opt_avgpool2d_863)
        opt_avgpool2d_839 = self.pad_avgpool2d_839(opt_relu_837)
        opt_avgpool2d_839 = self.avgpool2d_839(opt_avgpool2d_839)
        opt_conv2d_841 = self.conv2d_841(opt_avgpool2d_839)
        opt_add_865 = P.Add()(opt_conv2d_864, opt_conv2d_841)
        opt_relu_866 = self.relu_866(opt_add_865)
        module22_35_opt = self.module22_35(opt_relu_866)
        opt_split_871, opt_split_871_1 = self.split_871(module22_35_opt)
        opt_add_872 = opt_split_871 + self.add_872_bias
        opt_add_873 = P.Add()(opt_add_872, opt_split_871_1)
        module20_34_opt = self.module20_34(opt_add_873)
        opt_reshape_883 = self.reshape_883(module20_34_opt, self.reshape_883_shape)
        opt_split_884, opt_split_884_1 = self.split_884(opt_reshape_883)
        opt_mul_885 = P.Mul()(opt_split_884, opt_split_871)
        opt_mul_886 = P.Mul()(opt_split_884_1, opt_split_871_1)
        opt_add_887 = opt_mul_885 + self.add_887_bias
        opt_add_888 = P.Add()(opt_add_887, opt_mul_886)
        opt_conv2d_889 = self.conv2d_889(opt_add_888)
        opt_add_890 = P.Add()(opt_conv2d_889, opt_relu_866)
        opt_relu_891 = self.relu_891(opt_add_890)
        module22_36_opt = self.module22_36(opt_relu_891)
        opt_split_896, opt_split_896_1 = self.split_896(module22_36_opt)
        opt_add_897 = opt_split_896 + self.add_897_bias
        opt_add_898 = P.Add()(opt_add_897, opt_split_896_1)
        module20_35_opt = self.module20_35(opt_add_898)
        opt_reshape_908 = self.reshape_908(module20_35_opt, self.reshape_908_shape)
        opt_split_909, opt_split_909_1 = self.split_909(opt_reshape_908)
        opt_mul_910 = P.Mul()(opt_split_909, opt_split_896)
        opt_mul_911 = P.Mul()(opt_split_909_1, opt_split_896_1)
        opt_add_912 = opt_mul_910 + self.add_912_bias
        opt_add_913 = P.Add()(opt_add_912, opt_mul_911)
        opt_conv2d_914 = self.conv2d_914(opt_add_913)
        opt_add_915 = P.Add()(opt_conv2d_914, opt_relu_891)
        opt_relu_916 = self.relu_916(opt_add_915)
        module22_37_opt = self.module22_37(opt_relu_916)
        opt_split_921, opt_split_921_1 = self.split_921(module22_37_opt)
        opt_add_922 = opt_split_921 + self.add_922_bias
        opt_add_923 = P.Add()(opt_add_922, opt_split_921_1)
        module20_36_opt = self.module20_36(opt_add_923)
        opt_reshape_933 = self.reshape_933(module20_36_opt, self.reshape_933_shape)
        opt_split_934, opt_split_934_1 = self.split_934(opt_reshape_933)
        opt_mul_935 = P.Mul()(opt_split_934, opt_split_921)
        opt_mul_936 = P.Mul()(opt_split_934_1, opt_split_921_1)
        opt_add_937 = opt_mul_935 + self.add_937_bias
        opt_add_938 = P.Add()(opt_add_937, opt_mul_936)
        opt_conv2d_939 = self.conv2d_939(opt_add_938)
        opt_add_940 = P.Add()(opt_conv2d_939, opt_relu_916)
        opt_relu_941 = self.relu_941(opt_add_940)
        module22_38_opt = self.module22_38(opt_relu_941)
        opt_split_946, opt_split_946_1 = self.split_946(module22_38_opt)
        opt_add_947 = opt_split_946 + self.add_947_bias
        opt_add_948 = P.Add()(opt_add_947, opt_split_946_1)
        module20_37_opt = self.module20_37(opt_add_948)
        opt_reshape_958 = self.reshape_958(module20_37_opt, self.reshape_958_shape)
        opt_split_959, opt_split_959_1 = self.split_959(opt_reshape_958)
        opt_mul_960 = P.Mul()(opt_split_959, opt_split_946)
        opt_mul_961 = P.Mul()(opt_split_959_1, opt_split_946_1)
        opt_add_962 = opt_mul_960 + self.add_962_bias
        opt_add_963 = P.Add()(opt_add_962, opt_mul_961)
        opt_conv2d_964 = self.conv2d_964(opt_add_963)
        opt_add_965 = P.Add()(opt_conv2d_964, opt_relu_941)
        opt_relu_966 = self.relu_966(opt_add_965)
        module22_39_opt = self.module22_39(opt_relu_966)
        opt_split_971, opt_split_971_1 = self.split_971(module22_39_opt)
        opt_add_972 = opt_split_971 + self.add_972_bias
        opt_add_973 = P.Add()(opt_add_972, opt_split_971_1)
        module20_38_opt = self.module20_38(opt_add_973)
        opt_reshape_983 = self.reshape_983(module20_38_opt, self.reshape_983_shape)
        opt_split_984, opt_split_984_1 = self.split_984(opt_reshape_983)
        opt_mul_985 = P.Mul()(opt_split_984, opt_split_971)
        opt_mul_986 = P.Mul()(opt_split_984_1, opt_split_971_1)
        opt_add_987 = opt_mul_985 + self.add_987_bias
        opt_add_988 = P.Add()(opt_add_987, opt_mul_986)
        opt_conv2d_989 = self.conv2d_989(opt_add_988)
        opt_add_990 = P.Add()(opt_conv2d_989, opt_relu_966)
        opt_relu_991 = self.relu_991(opt_add_990)
        module22_40_opt = self.module22_40(opt_relu_991)
        opt_split_996, opt_split_996_1 = self.split_996(module22_40_opt)
        opt_add_997 = opt_split_996 + self.add_997_bias
        opt_add_998 = P.Add()(opt_add_997, opt_split_996_1)
        module20_39_opt = self.module20_39(opt_add_998)
        opt_reshape_1008 = self.reshape_1008(module20_39_opt, self.reshape_1008_shape)
        opt_split_1009, opt_split_1009_1 = self.split_1009(opt_reshape_1008)
        opt_mul_1010 = P.Mul()(opt_split_1009, opt_split_996)
        opt_mul_1011 = P.Mul()(opt_split_1009_1, opt_split_996_1)
        opt_add_1012 = opt_mul_1010 + self.add_1012_bias
        opt_add_1013 = P.Add()(opt_add_1012, opt_mul_1011)
        opt_conv2d_1014 = self.conv2d_1014(opt_add_1013)
        opt_add_1015 = P.Add()(opt_conv2d_1014, opt_relu_991)
        opt_relu_1016 = self.relu_1016(opt_add_1015)
        module22_41_opt = self.module22_41(opt_relu_1016)
        opt_split_1021, opt_split_1021_1 = self.split_1021(module22_41_opt)
        opt_add_1022 = opt_split_1021 + self.add_1022_bias
        opt_add_1023 = P.Add()(opt_add_1022, opt_split_1021_1)
        module20_40_opt = self.module20_40(opt_add_1023)
        opt_reshape_1033 = self.reshape_1033(module20_40_opt, self.reshape_1033_shape)
        opt_split_1034, opt_split_1034_1 = self.split_1034(opt_reshape_1033)
        opt_mul_1035 = P.Mul()(opt_split_1034, opt_split_1021)
        opt_mul_1036 = P.Mul()(opt_split_1034_1, opt_split_1021_1)
        opt_add_1037 = opt_mul_1035 + self.add_1037_bias
        opt_add_1038 = P.Add()(opt_add_1037, opt_mul_1036)
        opt_conv2d_1039 = self.conv2d_1039(opt_add_1038)
        opt_add_1040 = P.Add()(opt_conv2d_1039, opt_relu_1016)
        opt_relu_1041 = self.relu_1041(opt_add_1040)
        module22_42_opt = self.module22_42(opt_relu_1041)
        opt_split_1046, opt_split_1046_1 = self.split_1046(module22_42_opt)
        opt_add_1047 = opt_split_1046 + self.add_1047_bias
        opt_add_1048 = P.Add()(opt_add_1047, opt_split_1046_1)
        module20_41_opt = self.module20_41(opt_add_1048)
        opt_reshape_1058 = self.reshape_1058(module20_41_opt, self.reshape_1058_shape)
        opt_split_1059, opt_split_1059_1 = self.split_1059(opt_reshape_1058)
        opt_mul_1060 = P.Mul()(opt_split_1059, opt_split_1046)
        opt_mul_1061 = P.Mul()(opt_split_1059_1, opt_split_1046_1)
        opt_add_1062 = opt_mul_1060 + self.add_1062_bias
        opt_add_1063 = P.Add()(opt_add_1062, opt_mul_1061)
        opt_conv2d_1064 = self.conv2d_1064(opt_add_1063)
        opt_add_1065 = P.Add()(opt_conv2d_1064, opt_relu_1041)
        opt_relu_1066 = self.relu_1066(opt_add_1065)
        module22_43_opt = self.module22_43(opt_relu_1066)
        opt_split_1071, opt_split_1071_1 = self.split_1071(module22_43_opt)
        opt_add_1072 = opt_split_1071 + self.add_1072_bias
        opt_add_1073 = P.Add()(opt_add_1072, opt_split_1071_1)
        module20_42_opt = self.module20_42(opt_add_1073)
        opt_reshape_1083 = self.reshape_1083(module20_42_opt, self.reshape_1083_shape)
        opt_split_1084, opt_split_1084_1 = self.split_1084(opt_reshape_1083)
        opt_mul_1085 = P.Mul()(opt_split_1084, opt_split_1071)
        opt_mul_1086 = P.Mul()(opt_split_1084_1, opt_split_1071_1)
        opt_add_1087 = opt_mul_1085 + self.add_1087_bias
        opt_add_1088 = P.Add()(opt_add_1087, opt_mul_1086)
        opt_conv2d_1089 = self.conv2d_1089(opt_add_1088)
        opt_add_1090 = P.Add()(opt_conv2d_1089, opt_relu_1066)
        opt_relu_1091 = self.relu_1091(opt_add_1090)
        module22_44_opt = self.module22_44(opt_relu_1091)
        opt_split_1096, opt_split_1096_1 = self.split_1096(module22_44_opt)
        opt_add_1097 = opt_split_1096 + self.add_1097_bias
        opt_add_1098 = P.Add()(opt_add_1097, opt_split_1096_1)
        module20_43_opt = self.module20_43(opt_add_1098)
        opt_reshape_1108 = self.reshape_1108(module20_43_opt, self.reshape_1108_shape)
        opt_split_1109, opt_split_1109_1 = self.split_1109(opt_reshape_1108)
        opt_mul_1110 = P.Mul()(opt_split_1109, opt_split_1096)
        opt_mul_1111 = P.Mul()(opt_split_1109_1, opt_split_1096_1)
        opt_add_1112 = opt_mul_1110 + self.add_1112_bias
        opt_add_1113 = P.Add()(opt_add_1112, opt_mul_1111)
        opt_conv2d_1114 = self.conv2d_1114(opt_add_1113)
        opt_add_1115 = P.Add()(opt_conv2d_1114, opt_relu_1091)
        opt_relu_1116 = self.relu_1116(opt_add_1115)
        module22_45_opt = self.module22_45(opt_relu_1116)
        opt_split_1121, opt_split_1121_1 = self.split_1121(module22_45_opt)
        opt_add_1122 = opt_split_1121 + self.add_1122_bias
        opt_add_1123 = P.Add()(opt_add_1122, opt_split_1121_1)
        module20_44_opt = self.module20_44(opt_add_1123)
        opt_reshape_1133 = self.reshape_1133(module20_44_opt, self.reshape_1133_shape)
        opt_split_1134, opt_split_1134_1 = self.split_1134(opt_reshape_1133)
        opt_mul_1135 = P.Mul()(opt_split_1134, opt_split_1121)
        opt_mul_1136 = P.Mul()(opt_split_1134_1, opt_split_1121_1)
        opt_add_1137 = opt_mul_1135 + self.add_1137_bias
        opt_add_1138 = P.Add()(opt_add_1137, opt_mul_1136)
        opt_conv2d_1139 = self.conv2d_1139(opt_add_1138)
        opt_add_1140 = P.Add()(opt_conv2d_1139, opt_relu_1116)
        opt_relu_1141 = self.relu_1141(opt_add_1140)
        module22_46_opt = self.module22_46(opt_relu_1141)
        opt_split_1146, opt_split_1146_1 = self.split_1146(module22_46_opt)
        opt_add_1147 = opt_split_1146 + self.add_1147_bias
        opt_add_1148 = P.Add()(opt_add_1147, opt_split_1146_1)
        module20_45_opt = self.module20_45(opt_add_1148)
        opt_reshape_1158 = self.reshape_1158(module20_45_opt, self.reshape_1158_shape)
        opt_split_1159, opt_split_1159_1 = self.split_1159(opt_reshape_1158)
        opt_mul_1160 = P.Mul()(opt_split_1159, opt_split_1146)
        opt_mul_1161 = P.Mul()(opt_split_1159_1, opt_split_1146_1)
        opt_add_1162 = opt_mul_1160 + self.add_1162_bias
        opt_add_1163 = P.Add()(opt_add_1162, opt_mul_1161)
        opt_conv2d_1164 = self.conv2d_1164(opt_add_1163)
        opt_add_1165 = P.Add()(opt_conv2d_1164, opt_relu_1141)
        opt_relu_1166 = self.relu_1166(opt_add_1165)
        module22_47_opt = self.module22_47(opt_relu_1166)
        opt_split_1171, opt_split_1171_1 = self.split_1171(module22_47_opt)
        opt_add_1172 = opt_split_1171 + self.add_1172_bias
        opt_add_1173 = P.Add()(opt_add_1172, opt_split_1171_1)
        module20_46_opt = self.module20_46(opt_add_1173)
        opt_reshape_1183 = self.reshape_1183(module20_46_opt, self.reshape_1183_shape)
        opt_split_1184, opt_split_1184_1 = self.split_1184(opt_reshape_1183)
        opt_mul_1185 = P.Mul()(opt_split_1184, opt_split_1171)
        opt_mul_1186 = P.Mul()(opt_split_1184_1, opt_split_1171_1)
        opt_add_1187 = opt_mul_1185 + self.add_1187_bias
        opt_add_1188 = P.Add()(opt_add_1187, opt_mul_1186)
        opt_conv2d_1189 = self.conv2d_1189(opt_add_1188)
        opt_add_1190 = P.Add()(opt_conv2d_1189, opt_relu_1166)
        opt_relu_1191 = self.relu_1191(opt_add_1190)
        module22_48_opt = self.module22_48(opt_relu_1191)
        opt_split_1196, opt_split_1196_1 = self.split_1196(module22_48_opt)
        opt_add_1197 = opt_split_1196 + self.add_1197_bias
        opt_add_1198 = P.Add()(opt_add_1197, opt_split_1196_1)
        module20_47_opt = self.module20_47(opt_add_1198)
        opt_reshape_1208 = self.reshape_1208(module20_47_opt, self.reshape_1208_shape)
        opt_split_1209, opt_split_1209_1 = self.split_1209(opt_reshape_1208)
        opt_mul_1210 = P.Mul()(opt_split_1209, opt_split_1196)
        opt_mul_1211 = P.Mul()(opt_split_1209_1, opt_split_1196_1)
        opt_add_1212 = opt_mul_1210 + self.add_1212_bias
        opt_add_1213 = P.Add()(opt_add_1212, opt_mul_1211)
        opt_conv2d_1214 = self.conv2d_1214(opt_add_1213)
        opt_add_1215 = P.Add()(opt_conv2d_1214, opt_relu_1191)
        opt_relu_1216 = self.relu_1216(opt_add_1215)
        module22_49_opt = self.module22_49(opt_relu_1216)
        opt_split_1221, opt_split_1221_1 = self.split_1221(module22_49_opt)
        opt_add_1222 = opt_split_1221 + self.add_1222_bias
        opt_add_1223 = P.Add()(opt_add_1222, opt_split_1221_1)
        module20_48_opt = self.module20_48(opt_add_1223)
        opt_reshape_1233 = self.reshape_1233(module20_48_opt, self.reshape_1233_shape)
        opt_split_1234, opt_split_1234_1 = self.split_1234(opt_reshape_1233)
        opt_mul_1235 = P.Mul()(opt_split_1234, opt_split_1221)
        opt_mul_1236 = P.Mul()(opt_split_1234_1, opt_split_1221_1)
        opt_add_1237 = opt_mul_1235 + self.add_1237_bias
        opt_add_1238 = P.Add()(opt_add_1237, opt_mul_1236)
        opt_conv2d_1239 = self.conv2d_1239(opt_add_1238)
        opt_add_1240 = P.Add()(opt_conv2d_1239, opt_relu_1216)
        opt_relu_1241 = self.relu_1241(opt_add_1240)
        module22_50_opt = self.module22_50(opt_relu_1241)
        opt_split_1246, opt_split_1246_1 = self.split_1246(module22_50_opt)
        opt_add_1247 = opt_split_1246 + self.add_1247_bias
        opt_add_1248 = P.Add()(opt_add_1247, opt_split_1246_1)
        module20_49_opt = self.module20_49(opt_add_1248)
        opt_reshape_1258 = self.reshape_1258(module20_49_opt, self.reshape_1258_shape)
        opt_split_1259, opt_split_1259_1 = self.split_1259(opt_reshape_1258)
        opt_mul_1260 = P.Mul()(opt_split_1259, opt_split_1246)
        opt_mul_1261 = P.Mul()(opt_split_1259_1, opt_split_1246_1)
        opt_add_1262 = opt_mul_1260 + self.add_1262_bias
        opt_add_1263 = P.Add()(opt_add_1262, opt_mul_1261)
        opt_conv2d_1264 = self.conv2d_1264(opt_add_1263)
        opt_add_1265 = P.Add()(opt_conv2d_1264, opt_relu_1241)
        opt_relu_1266 = self.relu_1266(opt_add_1265)
        module22_51_opt = self.module22_51(opt_relu_1266)
        opt_split_1271, opt_split_1271_1 = self.split_1271(module22_51_opt)
        opt_add_1272 = opt_split_1271 + self.add_1272_bias
        opt_add_1273 = P.Add()(opt_add_1272, opt_split_1271_1)
        module20_50_opt = self.module20_50(opt_add_1273)
        opt_reshape_1283 = self.reshape_1283(module20_50_opt, self.reshape_1283_shape)
        opt_split_1284, opt_split_1284_1 = self.split_1284(opt_reshape_1283)
        opt_mul_1285 = P.Mul()(opt_split_1284, opt_split_1271)
        opt_mul_1286 = P.Mul()(opt_split_1284_1, opt_split_1271_1)
        opt_add_1287 = opt_mul_1285 + self.add_1287_bias
        opt_add_1288 = P.Add()(opt_add_1287, opt_mul_1286)
        opt_conv2d_1289 = self.conv2d_1289(opt_add_1288)
        opt_add_1290 = P.Add()(opt_conv2d_1289, opt_relu_1266)
        opt_relu_1291 = self.relu_1291(opt_add_1290)
        module22_52_opt = self.module22_52(opt_relu_1291)
        opt_split_1296, opt_split_1296_1 = self.split_1296(module22_52_opt)
        opt_add_1297 = opt_split_1296 + self.add_1297_bias
        opt_add_1298 = P.Add()(opt_add_1297, opt_split_1296_1)
        module20_51_opt = self.module20_51(opt_add_1298)
        opt_reshape_1308 = self.reshape_1308(module20_51_opt, self.reshape_1308_shape)
        opt_split_1309, opt_split_1309_1 = self.split_1309(opt_reshape_1308)
        opt_mul_1310 = P.Mul()(opt_split_1309, opt_split_1296)
        opt_mul_1311 = P.Mul()(opt_split_1309_1, opt_split_1296_1)
        opt_add_1312 = opt_mul_1310 + self.add_1312_bias
        opt_add_1313 = P.Add()(opt_add_1312, opt_mul_1311)
        opt_conv2d_1314 = self.conv2d_1314(opt_add_1313)
        opt_add_1315 = P.Add()(opt_conv2d_1314, opt_relu_1291)
        opt_relu_1316 = self.relu_1316(opt_add_1315)
        module22_53_opt = self.module22_53(opt_relu_1316)
        opt_split_1321, opt_split_1321_1 = self.split_1321(module22_53_opt)
        opt_add_1322 = opt_split_1321 + self.add_1322_bias
        opt_add_1323 = P.Add()(opt_add_1322, opt_split_1321_1)
        module20_52_opt = self.module20_52(opt_add_1323)
        opt_reshape_1333 = self.reshape_1333(module20_52_opt, self.reshape_1333_shape)
        opt_split_1334, opt_split_1334_1 = self.split_1334(opt_reshape_1333)
        opt_mul_1335 = P.Mul()(opt_split_1334, opt_split_1321)
        opt_mul_1336 = P.Mul()(opt_split_1334_1, opt_split_1321_1)
        opt_add_1337 = opt_mul_1335 + self.add_1337_bias
        opt_add_1338 = P.Add()(opt_add_1337, opt_mul_1336)
        opt_conv2d_1339 = self.conv2d_1339(opt_add_1338)
        opt_add_1340 = P.Add()(opt_conv2d_1339, opt_relu_1316)
        opt_relu_1341 = self.relu_1341(opt_add_1340)
        module22_54_opt = self.module22_54(opt_relu_1341)
        opt_split_1346, opt_split_1346_1 = self.split_1346(module22_54_opt)
        opt_add_1347 = opt_split_1346 + self.add_1347_bias
        opt_add_1348 = P.Add()(opt_add_1347, opt_split_1346_1)
        module20_53_opt = self.module20_53(opt_add_1348)
        opt_reshape_1358 = self.reshape_1358(module20_53_opt, self.reshape_1358_shape)
        opt_split_1359, opt_split_1359_1 = self.split_1359(opt_reshape_1358)
        opt_mul_1360 = P.Mul()(opt_split_1359, opt_split_1346)
        opt_mul_1361 = P.Mul()(opt_split_1359_1, opt_split_1346_1)
        opt_add_1362 = opt_mul_1360 + self.add_1362_bias
        opt_add_1363 = P.Add()(opt_add_1362, opt_mul_1361)
        opt_conv2d_1364 = self.conv2d_1364(opt_add_1363)
        opt_add_1365 = P.Add()(opt_conv2d_1364, opt_relu_1341)
        opt_relu_1366 = self.relu_1366(opt_add_1365)
        module22_55_opt = self.module22_55(opt_relu_1366)
        opt_split_1371, opt_split_1371_1 = self.split_1371(module22_55_opt)
        opt_add_1372 = opt_split_1371 + self.add_1372_bias
        opt_add_1373 = P.Add()(opt_add_1372, opt_split_1371_1)
        module20_54_opt = self.module20_54(opt_add_1373)
        opt_reshape_1383 = self.reshape_1383(module20_54_opt, self.reshape_1383_shape)
        opt_split_1384, opt_split_1384_1 = self.split_1384(opt_reshape_1383)
        opt_mul_1385 = P.Mul()(opt_split_1384, opt_split_1371)
        opt_mul_1386 = P.Mul()(opt_split_1384_1, opt_split_1371_1)
        opt_add_1387 = opt_mul_1385 + self.add_1387_bias
        opt_add_1388 = P.Add()(opt_add_1387, opt_mul_1386)
        opt_conv2d_1389 = self.conv2d_1389(opt_add_1388)
        opt_add_1390 = P.Add()(opt_conv2d_1389, opt_relu_1366)
        opt_relu_1391 = self.relu_1391(opt_add_1390)
        module22_56_opt = self.module22_56(opt_relu_1391)
        opt_split_1396, opt_split_1396_1 = self.split_1396(module22_56_opt)
        opt_add_1397 = opt_split_1396 + self.add_1397_bias
        opt_add_1398 = P.Add()(opt_add_1397, opt_split_1396_1)
        module20_55_opt = self.module20_55(opt_add_1398)
        opt_reshape_1408 = self.reshape_1408(module20_55_opt, self.reshape_1408_shape)
        opt_split_1409, opt_split_1409_1 = self.split_1409(opt_reshape_1408)
        opt_mul_1410 = P.Mul()(opt_split_1409, opt_split_1396)
        opt_mul_1411 = P.Mul()(opt_split_1409_1, opt_split_1396_1)
        opt_add_1412 = opt_mul_1410 + self.add_1412_bias
        opt_add_1413 = P.Add()(opt_add_1412, opt_mul_1411)
        opt_conv2d_1414 = self.conv2d_1414(opt_add_1413)
        opt_add_1415 = P.Add()(opt_conv2d_1414, opt_relu_1391)
        opt_relu_1416 = self.relu_1416(opt_add_1415)
        module22_57_opt = self.module22_57(opt_relu_1416)
        opt_split_1421, opt_split_1421_1 = self.split_1421(module22_57_opt)
        opt_add_1422 = opt_split_1421 + self.add_1422_bias
        opt_add_1423 = P.Add()(opt_add_1422, opt_split_1421_1)
        module20_56_opt = self.module20_56(opt_add_1423)
        opt_reshape_1433 = self.reshape_1433(module20_56_opt, self.reshape_1433_shape)
        opt_split_1434, opt_split_1434_1 = self.split_1434(opt_reshape_1433)
        opt_mul_1435 = P.Mul()(opt_split_1434, opt_split_1421)
        opt_mul_1436 = P.Mul()(opt_split_1434_1, opt_split_1421_1)
        opt_add_1437 = opt_mul_1435 + self.add_1437_bias
        opt_add_1438 = P.Add()(opt_add_1437, opt_mul_1436)
        opt_conv2d_1439 = self.conv2d_1439(opt_add_1438)
        opt_add_1440 = P.Add()(opt_conv2d_1439, opt_relu_1416)
        opt_relu_1441 = self.relu_1441(opt_add_1440)
        module22_58_opt = self.module22_58(opt_relu_1441)
        opt_split_1446, opt_split_1446_1 = self.split_1446(module22_58_opt)
        opt_add_1447 = opt_split_1446 + self.add_1447_bias
        opt_add_1448 = P.Add()(opt_add_1447, opt_split_1446_1)
        module20_57_opt = self.module20_57(opt_add_1448)
        opt_reshape_1458 = self.reshape_1458(module20_57_opt, self.reshape_1458_shape)
        opt_split_1459, opt_split_1459_1 = self.split_1459(opt_reshape_1458)
        opt_mul_1460 = P.Mul()(opt_split_1459, opt_split_1446)
        opt_mul_1461 = P.Mul()(opt_split_1459_1, opt_split_1446_1)
        opt_add_1462 = opt_mul_1460 + self.add_1462_bias
        opt_add_1463 = P.Add()(opt_add_1462, opt_mul_1461)
        opt_conv2d_1464 = self.conv2d_1464(opt_add_1463)
        opt_add_1465 = P.Add()(opt_conv2d_1464, opt_relu_1441)
        opt_relu_1466 = self.relu_1466(opt_add_1465)
        module22_59_opt = self.module22_59(opt_relu_1466)
        opt_split_1471, opt_split_1471_1 = self.split_1471(module22_59_opt)
        opt_add_1472 = opt_split_1471 + self.add_1472_bias
        opt_add_1473 = P.Add()(opt_add_1472, opt_split_1471_1)
        module20_58_opt = self.module20_58(opt_add_1473)
        opt_reshape_1483 = self.reshape_1483(module20_58_opt, self.reshape_1483_shape)
        opt_split_1484, opt_split_1484_1 = self.split_1484(opt_reshape_1483)
        opt_mul_1485 = P.Mul()(opt_split_1484, opt_split_1471)
        opt_mul_1486 = P.Mul()(opt_split_1484_1, opt_split_1471_1)
        opt_add_1487 = opt_mul_1485 + self.add_1487_bias
        opt_add_1488 = P.Add()(opt_add_1487, opt_mul_1486)
        opt_conv2d_1489 = self.conv2d_1489(opt_add_1488)
        opt_add_1490 = P.Add()(opt_conv2d_1489, opt_relu_1466)
        opt_relu_1491 = self.relu_1491(opt_add_1490)
        module22_60_opt = self.module22_60(opt_relu_1491)
        opt_split_1496, opt_split_1496_1 = self.split_1496(module22_60_opt)
        opt_add_1497 = opt_split_1496 + self.add_1497_bias
        opt_add_1498 = P.Add()(opt_add_1497, opt_split_1496_1)
        module20_59_opt = self.module20_59(opt_add_1498)
        opt_reshape_1508 = self.reshape_1508(module20_59_opt, self.reshape_1508_shape)
        opt_split_1509, opt_split_1509_1 = self.split_1509(opt_reshape_1508)
        opt_mul_1510 = P.Mul()(opt_split_1509, opt_split_1496)
        opt_mul_1511 = P.Mul()(opt_split_1509_1, opt_split_1496_1)
        opt_add_1512 = opt_mul_1510 + self.add_1512_bias
        opt_add_1513 = P.Add()(opt_add_1512, opt_mul_1511)
        opt_conv2d_1514 = self.conv2d_1514(opt_add_1513)
        opt_add_1515 = P.Add()(opt_conv2d_1514, opt_relu_1491)
        opt_relu_1516 = self.relu_1516(opt_add_1515)
        module22_61_opt = self.module22_61(opt_relu_1516)
        opt_split_1521, opt_split_1521_1 = self.split_1521(module22_61_opt)
        opt_add_1522 = opt_split_1521 + self.add_1522_bias
        opt_add_1523 = P.Add()(opt_add_1522, opt_split_1521_1)
        module20_60_opt = self.module20_60(opt_add_1523)
        opt_reshape_1533 = self.reshape_1533(module20_60_opt, self.reshape_1533_shape)
        opt_split_1534, opt_split_1534_1 = self.split_1534(opt_reshape_1533)
        opt_mul_1535 = P.Mul()(opt_split_1534, opt_split_1521)
        opt_mul_1536 = P.Mul()(opt_split_1534_1, opt_split_1521_1)
        opt_add_1537 = opt_mul_1535 + self.add_1537_bias
        opt_add_1538 = P.Add()(opt_add_1537, opt_mul_1536)
        opt_conv2d_1539 = self.conv2d_1539(opt_add_1538)
        opt_add_1540 = P.Add()(opt_conv2d_1539, opt_relu_1516)
        opt_relu_1541 = self.relu_1541(opt_add_1540)
        module22_62_opt = self.module22_62(opt_relu_1541)
        opt_split_1546, opt_split_1546_1 = self.split_1546(module22_62_opt)
        opt_add_1547 = opt_split_1546 + self.add_1547_bias
        opt_add_1548 = P.Add()(opt_add_1547, opt_split_1546_1)
        module20_61_opt = self.module20_61(opt_add_1548)
        opt_reshape_1558 = self.reshape_1558(module20_61_opt, self.reshape_1558_shape)
        opt_split_1559, opt_split_1559_1 = self.split_1559(opt_reshape_1558)
        opt_mul_1560 = P.Mul()(opt_split_1559, opt_split_1546)
        opt_mul_1561 = P.Mul()(opt_split_1559_1, opt_split_1546_1)
        opt_add_1562 = opt_mul_1560 + self.add_1562_bias
        opt_add_1563 = P.Add()(opt_add_1562, opt_mul_1561)
        opt_conv2d_1564 = self.conv2d_1564(opt_add_1563)
        opt_add_1565 = P.Add()(opt_conv2d_1564, opt_relu_1541)
        opt_relu_1566 = self.relu_1566(opt_add_1565)
        module22_63_opt = self.module22_63(opt_relu_1566)
        opt_split_1571, opt_split_1571_1 = self.split_1571(module22_63_opt)
        opt_add_1572 = opt_split_1571 + self.add_1572_bias
        opt_add_1573 = P.Add()(opt_add_1572, opt_split_1571_1)
        module20_62_opt = self.module20_62(opt_add_1573)
        opt_reshape_1583 = self.reshape_1583(module20_62_opt, self.reshape_1583_shape)
        opt_split_1584, opt_split_1584_1 = self.split_1584(opt_reshape_1583)
        opt_mul_1585 = P.Mul()(opt_split_1584, opt_split_1571)
        opt_mul_1586 = P.Mul()(opt_split_1584_1, opt_split_1571_1)
        opt_add_1587 = opt_mul_1585 + self.add_1587_bias
        opt_add_1588 = P.Add()(opt_add_1587, opt_mul_1586)
        opt_conv2d_1589 = self.conv2d_1589(opt_add_1588)
        opt_add_1590 = P.Add()(opt_conv2d_1589, opt_relu_1566)
        opt_relu_1591 = self.relu_1591(opt_add_1590)
        module22_64_opt = self.module22_64(opt_relu_1591)
        opt_split_1596, opt_split_1596_1 = self.split_1596(module22_64_opt)
        opt_add_1597 = opt_split_1596 + self.add_1597_bias
        opt_add_1598 = P.Add()(opt_add_1597, opt_split_1596_1)
        module20_63_opt = self.module20_63(opt_add_1598)
        opt_reshape_1608 = self.reshape_1608(module20_63_opt, self.reshape_1608_shape)
        opt_split_1609, opt_split_1609_1 = self.split_1609(opt_reshape_1608)
        opt_mul_1610 = P.Mul()(opt_split_1609, opt_split_1596)
        opt_mul_1611 = P.Mul()(opt_split_1609_1, opt_split_1596_1)
        opt_add_1612 = opt_mul_1610 + self.add_1612_bias
        opt_add_1613 = P.Add()(opt_add_1612, opt_mul_1611)
        opt_conv2d_1614 = self.conv2d_1614(opt_add_1613)
        opt_add_1615 = P.Add()(opt_conv2d_1614, opt_relu_1591)
        opt_relu_1616 = self.relu_1616(opt_add_1615)
        module22_65_opt = self.module22_65(opt_relu_1616)
        opt_split_1621, opt_split_1621_1 = self.split_1621(module22_65_opt)
        opt_add_1622 = opt_split_1621 + self.add_1622_bias
        opt_add_1623 = P.Add()(opt_add_1622, opt_split_1621_1)
        module20_64_opt = self.module20_64(opt_add_1623)
        opt_reshape_1633 = self.reshape_1633(module20_64_opt, self.reshape_1633_shape)
        opt_split_1634, opt_split_1634_1 = self.split_1634(opt_reshape_1633)
        opt_mul_1635 = P.Mul()(opt_split_1634, opt_split_1621)
        opt_mul_1636 = P.Mul()(opt_split_1634_1, opt_split_1621_1)
        opt_add_1637 = opt_mul_1635 + self.add_1637_bias
        opt_add_1638 = P.Add()(opt_add_1637, opt_mul_1636)
        opt_conv2d_1639 = self.conv2d_1639(opt_add_1638)
        opt_add_1640 = P.Add()(opt_conv2d_1639, opt_relu_1616)
        opt_relu_1641 = self.relu_1641(opt_add_1640)
        module22_66_opt = self.module22_66(opt_relu_1641)
        opt_split_1646, opt_split_1646_1 = self.split_1646(module22_66_opt)
        opt_add_1647 = opt_split_1646 + self.add_1647_bias
        opt_add_1648 = P.Add()(opt_add_1647, opt_split_1646_1)
        module20_65_opt = self.module20_65(opt_add_1648)
        opt_reshape_1658 = self.reshape_1658(module20_65_opt, self.reshape_1658_shape)
        opt_split_1659, opt_split_1659_1 = self.split_1659(opt_reshape_1658)
        opt_mul_1660 = P.Mul()(opt_split_1659, opt_split_1646)
        opt_mul_1661 = P.Mul()(opt_split_1659_1, opt_split_1646_1)
        opt_add_1662 = opt_mul_1660 + self.add_1662_bias
        opt_add_1663 = P.Add()(opt_add_1662, opt_mul_1661)
        opt_conv2d_1664 = self.conv2d_1664(opt_add_1663)
        opt_add_1665 = P.Add()(opt_conv2d_1664, opt_relu_1641)
        opt_relu_1666 = self.relu_1666(opt_add_1665)
        module22_67_opt = self.module22_67(opt_relu_1666)
        opt_split_1671, opt_split_1671_1 = self.split_1671(module22_67_opt)
        opt_add_1672 = opt_split_1671 + self.add_1672_bias
        opt_add_1673 = P.Add()(opt_add_1672, opt_split_1671_1)
        module20_66_opt = self.module20_66(opt_add_1673)
        opt_reshape_1683 = self.reshape_1683(module20_66_opt, self.reshape_1683_shape)
        opt_split_1684, opt_split_1684_1 = self.split_1684(opt_reshape_1683)
        opt_mul_1685 = P.Mul()(opt_split_1684, opt_split_1671)
        opt_mul_1686 = P.Mul()(opt_split_1684_1, opt_split_1671_1)
        opt_add_1687 = opt_mul_1685 + self.add_1687_bias
        opt_add_1688 = P.Add()(opt_add_1687, opt_mul_1686)
        opt_conv2d_1689 = self.conv2d_1689(opt_add_1688)
        opt_add_1690 = P.Add()(opt_conv2d_1689, opt_relu_1666)
        opt_relu_1691 = self.relu_1691(opt_add_1690)
        module22_68_opt = self.module22_68(opt_relu_1691)
        opt_split_1696, opt_split_1696_1 = self.split_1696(module22_68_opt)
        opt_add_1697 = opt_split_1696 + self.add_1697_bias
        opt_add_1698 = P.Add()(opt_add_1697, opt_split_1696_1)
        module20_67_opt = self.module20_67(opt_add_1698)
        opt_reshape_1708 = self.reshape_1708(module20_67_opt, self.reshape_1708_shape)
        opt_split_1709, opt_split_1709_1 = self.split_1709(opt_reshape_1708)
        opt_mul_1710 = P.Mul()(opt_split_1709, opt_split_1696)
        opt_mul_1711 = P.Mul()(opt_split_1709_1, opt_split_1696_1)
        opt_add_1712 = opt_mul_1710 + self.add_1712_bias
        opt_add_1713 = P.Add()(opt_add_1712, opt_mul_1711)
        opt_conv2d_1714 = self.conv2d_1714(opt_add_1713)
        opt_add_1715 = P.Add()(opt_conv2d_1714, opt_relu_1691)
        opt_relu_1716 = self.relu_1716(opt_add_1715)
        module22_69_opt = self.module22_69(opt_relu_1716)
        opt_split_1721, opt_split_1721_1 = self.split_1721(module22_69_opt)
        opt_add_1722 = opt_split_1721 + self.add_1722_bias
        opt_add_1723 = P.Add()(opt_add_1722, opt_split_1721_1)
        module20_68_opt = self.module20_68(opt_add_1723)
        opt_reshape_1733 = self.reshape_1733(module20_68_opt, self.reshape_1733_shape)
        opt_split_1734, opt_split_1734_1 = self.split_1734(opt_reshape_1733)
        opt_mul_1735 = P.Mul()(opt_split_1734, opt_split_1721)
        opt_mul_1736 = P.Mul()(opt_split_1734_1, opt_split_1721_1)
        opt_add_1737 = opt_mul_1735 + self.add_1737_bias
        opt_add_1738 = P.Add()(opt_add_1737, opt_mul_1736)
        opt_conv2d_1739 = self.conv2d_1739(opt_add_1738)
        opt_add_1740 = P.Add()(opt_conv2d_1739, opt_relu_1716)
        opt_relu_1741 = self.relu_1741(opt_add_1740)
        module22_70_opt = self.module22_70(opt_relu_1741)
        opt_split_1746, opt_split_1746_1 = self.split_1746(module22_70_opt)
        opt_add_1747 = opt_split_1746 + self.add_1747_bias
        opt_add_1748 = P.Add()(opt_add_1747, opt_split_1746_1)
        module20_69_opt = self.module20_69(opt_add_1748)
        opt_reshape_1758 = self.reshape_1758(module20_69_opt, self.reshape_1758_shape)
        opt_split_1759, opt_split_1759_1 = self.split_1759(opt_reshape_1758)
        opt_mul_1760 = P.Mul()(opt_split_1759, opt_split_1746)
        opt_mul_1761 = P.Mul()(opt_split_1759_1, opt_split_1746_1)
        opt_add_1762 = opt_mul_1760 + self.add_1762_bias
        opt_add_1763 = P.Add()(opt_add_1762, opt_mul_1761)
        opt_conv2d_1764 = self.conv2d_1764(opt_add_1763)
        opt_add_1765 = P.Add()(opt_conv2d_1764, opt_relu_1741)
        opt_relu_1766 = self.relu_1766(opt_add_1765)
        module22_71_opt = self.module22_71(opt_relu_1766)
        opt_split_1771, opt_split_1771_1 = self.split_1771(module22_71_opt)
        opt_add_1772 = opt_split_1771 + self.add_1772_bias
        opt_add_1773 = P.Add()(opt_add_1772, opt_split_1771_1)
        module20_70_opt = self.module20_70(opt_add_1773)
        opt_reshape_1783 = self.reshape_1783(module20_70_opt, self.reshape_1783_shape)
        opt_split_1784, opt_split_1784_1 = self.split_1784(opt_reshape_1783)
        opt_mul_1785 = P.Mul()(opt_split_1784, opt_split_1771)
        opt_mul_1786 = P.Mul()(opt_split_1784_1, opt_split_1771_1)
        opt_add_1787 = opt_mul_1785 + self.add_1787_bias
        opt_add_1788 = P.Add()(opt_add_1787, opt_mul_1786)
        opt_conv2d_1789 = self.conv2d_1789(opt_add_1788)
        opt_add_1790 = P.Add()(opt_conv2d_1789, opt_relu_1766)
        opt_relu_1791 = self.relu_1791(opt_add_1790)
        module22_72_opt = self.module22_72(opt_relu_1791)
        opt_split_1796, opt_split_1796_1 = self.split_1796(module22_72_opt)
        opt_add_1797 = opt_split_1796 + self.add_1797_bias
        opt_add_1798 = P.Add()(opt_add_1797, opt_split_1796_1)
        module20_71_opt = self.module20_71(opt_add_1798)
        opt_reshape_1808 = self.reshape_1808(module20_71_opt, self.reshape_1808_shape)
        opt_split_1809, opt_split_1809_1 = self.split_1809(opt_reshape_1808)
        opt_mul_1810 = P.Mul()(opt_split_1809, opt_split_1796)
        opt_mul_1811 = P.Mul()(opt_split_1809_1, opt_split_1796_1)
        opt_add_1812 = opt_mul_1810 + self.add_1812_bias
        opt_add_1813 = P.Add()(opt_add_1812, opt_mul_1811)
        opt_conv2d_1814 = self.conv2d_1814(opt_add_1813)
        opt_add_1815 = P.Add()(opt_conv2d_1814, opt_relu_1791)
        opt_relu_1816 = self.relu_1816(opt_add_1815)
        module22_73_opt = self.module22_73(opt_relu_1816)
        opt_split_1821, opt_split_1821_1 = self.split_1821(module22_73_opt)
        opt_add_1822 = opt_split_1821 + self.add_1822_bias
        opt_add_1823 = P.Add()(opt_add_1822, opt_split_1821_1)
        module20_72_opt = self.module20_72(opt_add_1823)
        opt_reshape_1833 = self.reshape_1833(module20_72_opt, self.reshape_1833_shape)
        opt_split_1834, opt_split_1834_1 = self.split_1834(opt_reshape_1833)
        opt_mul_1835 = P.Mul()(opt_split_1834, opt_split_1821)
        opt_mul_1836 = P.Mul()(opt_split_1834_1, opt_split_1821_1)
        opt_add_1837 = opt_mul_1835 + self.add_1837_bias
        opt_add_1838 = P.Add()(opt_add_1837, opt_mul_1836)
        opt_conv2d_1839 = self.conv2d_1839(opt_add_1838)
        opt_add_1840 = P.Add()(opt_conv2d_1839, opt_relu_1816)
        opt_relu_1841 = self.relu_1841(opt_add_1840)
        module22_74_opt = self.module22_74(opt_relu_1841)
        opt_split_1846, opt_split_1846_1 = self.split_1846(module22_74_opt)
        opt_add_1847 = opt_split_1846 + self.add_1847_bias
        opt_add_1848 = P.Add()(opt_add_1847, opt_split_1846_1)
        module20_73_opt = self.module20_73(opt_add_1848)
        opt_reshape_1858 = self.reshape_1858(module20_73_opt, self.reshape_1858_shape)
        opt_split_1859, opt_split_1859_1 = self.split_1859(opt_reshape_1858)
        opt_mul_1860 = P.Mul()(opt_split_1859, opt_split_1846)
        opt_mul_1861 = P.Mul()(opt_split_1859_1, opt_split_1846_1)
        opt_add_1862 = opt_mul_1860 + self.add_1862_bias
        opt_add_1863 = P.Add()(opt_add_1862, opt_mul_1861)
        opt_conv2d_1864 = self.conv2d_1864(opt_add_1863)
        opt_add_1865 = P.Add()(opt_conv2d_1864, opt_relu_1841)
        opt_relu_1866 = self.relu_1866(opt_add_1865)
        module22_75_opt = self.module22_75(opt_relu_1866)
        opt_split_1871, opt_split_1871_1 = self.split_1871(module22_75_opt)
        opt_add_1872 = opt_split_1871 + self.add_1872_bias
        opt_add_1873 = P.Add()(opt_add_1872, opt_split_1871_1)
        module20_74_opt = self.module20_74(opt_add_1873)
        opt_reshape_1883 = self.reshape_1883(module20_74_opt, self.reshape_1883_shape)
        opt_split_1884, opt_split_1884_1 = self.split_1884(opt_reshape_1883)
        opt_mul_1885 = P.Mul()(opt_split_1884, opt_split_1871)
        opt_mul_1886 = P.Mul()(opt_split_1884_1, opt_split_1871_1)
        opt_add_1887 = opt_mul_1885 + self.add_1887_bias
        opt_add_1888 = P.Add()(opt_add_1887, opt_mul_1886)
        opt_conv2d_1889 = self.conv2d_1889(opt_add_1888)
        opt_add_1890 = P.Add()(opt_conv2d_1889, opt_relu_1866)
        opt_relu_1891 = self.relu_1891(opt_add_1890)
        module22_76_opt = self.module22_76(opt_relu_1891)
        opt_split_1896, opt_split_1896_1 = self.split_1896(module22_76_opt)
        opt_add_1897 = opt_split_1896 + self.add_1897_bias
        opt_add_1898 = P.Add()(opt_add_1897, opt_split_1896_1)
        module20_75_opt = self.module20_75(opt_add_1898)
        opt_reshape_1908 = self.reshape_1908(module20_75_opt, self.reshape_1908_shape)
        opt_split_1909, opt_split_1909_1 = self.split_1909(opt_reshape_1908)
        opt_mul_1910 = P.Mul()(opt_split_1909, opt_split_1896)
        opt_mul_1911 = P.Mul()(opt_split_1909_1, opt_split_1896_1)
        opt_add_1912 = opt_mul_1910 + self.add_1912_bias
        opt_add_1913 = P.Add()(opt_add_1912, opt_mul_1911)
        opt_conv2d_1914 = self.conv2d_1914(opt_add_1913)
        opt_add_1915 = P.Add()(opt_conv2d_1914, opt_relu_1891)
        opt_relu_1916 = self.relu_1916(opt_add_1915)
        module22_77_opt = self.module22_77(opt_relu_1916)
        opt_split_1921, opt_split_1921_1 = self.split_1921(module22_77_opt)
        opt_add_1922 = opt_split_1921 + self.add_1922_bias
        opt_add_1923 = P.Add()(opt_add_1922, opt_split_1921_1)
        module20_76_opt = self.module20_76(opt_add_1923)
        opt_reshape_1933 = self.reshape_1933(module20_76_opt, self.reshape_1933_shape)
        opt_split_1934, opt_split_1934_1 = self.split_1934(opt_reshape_1933)
        opt_mul_1935 = P.Mul()(opt_split_1934, opt_split_1921)
        opt_mul_1936 = P.Mul()(opt_split_1934_1, opt_split_1921_1)
        opt_add_1937 = opt_mul_1935 + self.add_1937_bias
        opt_add_1938 = P.Add()(opt_add_1937, opt_mul_1936)
        opt_conv2d_1939 = self.conv2d_1939(opt_add_1938)
        opt_add_1940 = P.Add()(opt_conv2d_1939, opt_relu_1916)
        opt_relu_1941 = self.relu_1941(opt_add_1940)
        module22_78_opt = self.module22_78(opt_relu_1941)
        opt_split_1946, opt_split_1946_1 = self.split_1946(module22_78_opt)
        opt_add_1947 = opt_split_1946 + self.add_1947_bias
        opt_add_1948 = P.Add()(opt_add_1947, opt_split_1946_1)
        module20_77_opt = self.module20_77(opt_add_1948)
        opt_reshape_1958 = self.reshape_1958(module20_77_opt, self.reshape_1958_shape)
        opt_split_1959, opt_split_1959_1 = self.split_1959(opt_reshape_1958)
        opt_mul_1960 = P.Mul()(opt_split_1959, opt_split_1946)
        opt_mul_1961 = P.Mul()(opt_split_1959_1, opt_split_1946_1)
        opt_add_1962 = opt_mul_1960 + self.add_1962_bias
        opt_add_1963 = P.Add()(opt_add_1962, opt_mul_1961)
        opt_conv2d_1964 = self.conv2d_1964(opt_add_1963)
        opt_add_1965 = P.Add()(opt_conv2d_1964, opt_relu_1941)
        opt_relu_1966 = self.relu_1966(opt_add_1965)
        module22_79_opt = self.module22_79(opt_relu_1966)
        opt_split_1971, opt_split_1971_1 = self.split_1971(module22_79_opt)
        opt_add_1972 = opt_split_1971 + self.add_1972_bias
        opt_add_1973 = P.Add()(opt_add_1972, opt_split_1971_1)
        module20_78_opt = self.module20_78(opt_add_1973)
        opt_reshape_1983 = self.reshape_1983(module20_78_opt, self.reshape_1983_shape)
        opt_split_1984, opt_split_1984_1 = self.split_1984(opt_reshape_1983)
        opt_mul_1985 = P.Mul()(opt_split_1984, opt_split_1971)
        opt_mul_1986 = P.Mul()(opt_split_1984_1, opt_split_1971_1)
        opt_add_1987 = opt_mul_1985 + self.add_1987_bias
        opt_add_1988 = P.Add()(opt_add_1987, opt_mul_1986)
        opt_conv2d_1989 = self.conv2d_1989(opt_add_1988)
        opt_add_1990 = P.Add()(opt_conv2d_1989, opt_relu_1966)
        opt_relu_1991 = self.relu_1991(opt_add_1990)
        module22_80_opt = self.module22_80(opt_relu_1991)
        opt_split_1996, opt_split_1996_1 = self.split_1996(module22_80_opt)
        opt_add_1997 = opt_split_1996 + self.add_1997_bias
        opt_add_1998 = P.Add()(opt_add_1997, opt_split_1996_1)
        module20_79_opt = self.module20_79(opt_add_1998)
        opt_reshape_2008 = self.reshape_2008(module20_79_opt, self.reshape_2008_shape)
        opt_split_2009, opt_split_2009_1 = self.split_2009(opt_reshape_2008)
        opt_mul_2010 = P.Mul()(opt_split_2009, opt_split_1996)
        opt_mul_2011 = P.Mul()(opt_split_2009_1, opt_split_1996_1)
        opt_add_2012 = opt_mul_2010 + self.add_2012_bias
        opt_add_2013 = P.Add()(opt_add_2012, opt_mul_2011)
        opt_conv2d_2014 = self.conv2d_2014(opt_add_2013)
        opt_add_2015 = P.Add()(opt_conv2d_2014, opt_relu_1991)
        opt_relu_2016 = self.relu_2016(opt_add_2015)
        module22_81_opt = self.module22_81(opt_relu_2016)
        opt_split_2021, opt_split_2021_1 = self.split_2021(module22_81_opt)
        opt_add_2022 = opt_split_2021 + self.add_2022_bias
        opt_add_2023 = P.Add()(opt_add_2022, opt_split_2021_1)
        module20_80_opt = self.module20_80(opt_add_2023)
        opt_reshape_2033 = self.reshape_2033(module20_80_opt, self.reshape_2033_shape)
        opt_split_2034, opt_split_2034_1 = self.split_2034(opt_reshape_2033)
        opt_mul_2035 = P.Mul()(opt_split_2034, opt_split_2021)
        opt_mul_2036 = P.Mul()(opt_split_2034_1, opt_split_2021_1)
        opt_add_2037 = opt_mul_2035 + self.add_2037_bias
        opt_add_2038 = P.Add()(opt_add_2037, opt_mul_2036)
        opt_conv2d_2039 = self.conv2d_2039(opt_add_2038)
        opt_add_2040 = P.Add()(opt_conv2d_2039, opt_relu_2016)
        opt_relu_2041 = self.relu_2041(opt_add_2040)
        module22_82_opt = self.module22_82(opt_relu_2041)
        opt_split_2048, opt_split_2048_1 = self.split_2048(module22_82_opt)
        opt_add_2049 = opt_split_2048 + self.add_2049_bias
        opt_add_2050 = P.Add()(opt_add_2049, opt_split_2048_1)
        module20_81_opt = self.module20_81(opt_add_2050)
        opt_reshape_2060 = self.reshape_2060(module20_81_opt, self.reshape_2060_shape)
        opt_split_2061, opt_split_2061_1 = self.split_2061(opt_reshape_2060)
        opt_mul_2062 = P.Mul()(opt_split_2061, opt_split_2048)
        opt_mul_2063 = P.Mul()(opt_split_2061_1, opt_split_2048_1)
        opt_add_2064 = opt_mul_2062 + self.add_2064_bias
        opt_add_2065 = P.Add()(opt_add_2064, opt_mul_2063)
        opt_pad_2066 = self.pad_2066(opt_add_2065)
        opt_avgpool2d_2067 = self.pad_avgpool2d_2067(opt_pad_2066)
        opt_avgpool2d_2067 = self.avgpool2d_2067(opt_avgpool2d_2067)
        opt_conv2d_2068 = self.conv2d_2068(opt_avgpool2d_2067)
        opt_avgpool2d_2043 = self.pad_avgpool2d_2043(opt_relu_2041)
        opt_avgpool2d_2043 = self.avgpool2d_2043(opt_avgpool2d_2043)
        opt_conv2d_2045 = self.conv2d_2045(opt_avgpool2d_2043)
        opt_add_2069 = P.Add()(opt_conv2d_2068, opt_conv2d_2045)
        opt_relu_2070 = self.relu_2070(opt_add_2069)
        module22_83_opt = self.module22_83(opt_relu_2070)
        opt_split_2075, opt_split_2075_1 = self.split_2075(module22_83_opt)
        opt_add_2076 = opt_split_2075 + self.add_2076_bias
        opt_add_2077 = P.Add()(opt_add_2076, opt_split_2075_1)
        module20_82_opt = self.module20_82(opt_add_2077)
        opt_reshape_2087 = self.reshape_2087(module20_82_opt, self.reshape_2087_shape)
        opt_split_2088, opt_split_2088_1 = self.split_2088(opt_reshape_2087)
        opt_mul_2089 = P.Mul()(opt_split_2088, opt_split_2075)
        opt_mul_2090 = P.Mul()(opt_split_2088_1, opt_split_2075_1)
        opt_add_2091 = opt_mul_2089 + self.add_2091_bias
        opt_add_2092 = P.Add()(opt_add_2091, opt_mul_2090)
        opt_conv2d_2093 = self.conv2d_2093(opt_add_2092)
        opt_add_2094 = P.Add()(opt_conv2d_2093, opt_relu_2070)
        opt_relu_2095 = self.relu_2095(opt_add_2094)
        module22_84_opt = self.module22_84(opt_relu_2095)
        opt_split_2100, opt_split_2100_1 = self.split_2100(module22_84_opt)
        opt_add_2101 = opt_split_2100 + self.add_2101_bias
        opt_add_2102 = P.Add()(opt_add_2101, opt_split_2100_1)
        module20_83_opt = self.module20_83(opt_add_2102)
        opt_reshape_2112 = self.reshape_2112(module20_83_opt, self.reshape_2112_shape)
        opt_split_2113, opt_split_2113_1 = self.split_2113(opt_reshape_2112)
        opt_mul_2114 = P.Mul()(opt_split_2113, opt_split_2100)
        opt_mul_2115 = P.Mul()(opt_split_2113_1, opt_split_2100_1)
        opt_add_2116 = opt_mul_2114 + self.add_2116_bias
        opt_add_2117 = P.Add()(opt_add_2116, opt_mul_2115)
        opt_conv2d_2118 = self.conv2d_2118(opt_add_2117)
        opt_add_2119 = P.Add()(opt_conv2d_2118, opt_relu_2095)
        opt_relu_2120 = self.relu_2120(opt_add_2119)
        module22_85_opt = self.module22_85(opt_relu_2120)
        opt_split_2125, opt_split_2125_1 = self.split_2125(module22_85_opt)
        opt_add_2126 = opt_split_2125 + self.add_2126_bias
        opt_add_2127 = P.Add()(opt_add_2126, opt_split_2125_1)
        module20_84_opt = self.module20_84(opt_add_2127)
        opt_reshape_2137 = self.reshape_2137(module20_84_opt, self.reshape_2137_shape)
        opt_split_2138, opt_split_2138_1 = self.split_2138(opt_reshape_2137)
        opt_mul_2139 = P.Mul()(opt_split_2138, opt_split_2125)
        opt_mul_2140 = P.Mul()(opt_split_2138_1, opt_split_2125_1)
        opt_add_2141 = opt_mul_2139 + self.add_2141_bias
        opt_add_2142 = P.Add()(opt_add_2141, opt_mul_2140)
        opt_conv2d_2143 = self.conv2d_2143(opt_add_2142)
        opt_add_2144 = P.Add()(opt_conv2d_2143, opt_relu_2120)
        opt_relu_2145 = self.relu_2145(opt_add_2144)
        module22_86_opt = self.module22_86(opt_relu_2145)
        opt_split_2150, opt_split_2150_1 = self.split_2150(module22_86_opt)
        opt_add_2151 = opt_split_2150 + self.add_2151_bias
        opt_add_2152 = P.Add()(opt_add_2151, opt_split_2150_1)
        module20_85_opt = self.module20_85(opt_add_2152)
        opt_reshape_2162 = self.reshape_2162(module20_85_opt, self.reshape_2162_shape)
        opt_split_2163, opt_split_2163_1 = self.split_2163(opt_reshape_2162)
        opt_mul_2164 = P.Mul()(opt_split_2163, opt_split_2150)
        opt_mul_2165 = P.Mul()(opt_split_2163_1, opt_split_2150_1)
        opt_add_2166 = opt_mul_2164 + self.add_2166_bias
        opt_add_2167 = P.Add()(opt_add_2166, opt_mul_2165)
        opt_conv2d_2168 = self.conv2d_2168(opt_add_2167)
        opt_add_2169 = P.Add()(opt_conv2d_2168, opt_relu_2145)
        opt_relu_2170 = self.relu_2170(opt_add_2169)
        module22_87_opt = self.module22_87(opt_relu_2170)
        opt_split_2175, opt_split_2175_1 = self.split_2175(module22_87_opt)
        opt_add_2176 = opt_split_2175 + self.add_2176_bias
        opt_add_2177 = P.Add()(opt_add_2176, opt_split_2175_1)
        module20_86_opt = self.module20_86(opt_add_2177)
        opt_reshape_2187 = self.reshape_2187(module20_86_opt, self.reshape_2187_shape)
        opt_split_2188, opt_split_2188_1 = self.split_2188(opt_reshape_2187)
        opt_mul_2189 = P.Mul()(opt_split_2188, opt_split_2175)
        opt_mul_2190 = P.Mul()(opt_split_2188_1, opt_split_2175_1)
        opt_add_2191 = opt_mul_2189 + self.add_2191_bias
        opt_add_2192 = P.Add()(opt_add_2191, opt_mul_2190)
        opt_conv2d_2193 = self.conv2d_2193(opt_add_2192)
        opt_add_2194 = P.Add()(opt_conv2d_2193, opt_relu_2170)
        opt_relu_2195 = self.relu_2195(opt_add_2194)
        module22_88_opt = self.module22_88(opt_relu_2195)
        opt_split_2200, opt_split_2200_1 = self.split_2200(module22_88_opt)
        opt_add_2201 = opt_split_2200 + self.add_2201_bias
        opt_add_2202 = P.Add()(opt_add_2201, opt_split_2200_1)
        module20_87_opt = self.module20_87(opt_add_2202)
        opt_reshape_2212 = self.reshape_2212(module20_87_opt, self.reshape_2212_shape)
        opt_split_2213, opt_split_2213_1 = self.split_2213(opt_reshape_2212)
        opt_mul_2214 = P.Mul()(opt_split_2213, opt_split_2200)
        opt_mul_2215 = P.Mul()(opt_split_2213_1, opt_split_2200_1)
        opt_add_2216 = opt_mul_2214 + self.add_2216_bias
        opt_add_2217 = P.Add()(opt_add_2216, opt_mul_2215)
        opt_conv2d_2218 = self.conv2d_2218(opt_add_2217)
        opt_add_2219 = P.Add()(opt_conv2d_2218, opt_relu_2195)
        opt_relu_2220 = self.relu_2220(opt_add_2219)
        module22_89_opt = self.module22_89(opt_relu_2220)
        opt_split_2225, opt_split_2225_1 = self.split_2225(module22_89_opt)
        opt_add_2226 = opt_split_2225 + self.add_2226_bias
        opt_add_2227 = P.Add()(opt_add_2226, opt_split_2225_1)
        module20_88_opt = self.module20_88(opt_add_2227)
        opt_reshape_2237 = self.reshape_2237(module20_88_opt, self.reshape_2237_shape)
        opt_split_2238, opt_split_2238_1 = self.split_2238(opt_reshape_2237)
        opt_mul_2239 = P.Mul()(opt_split_2238, opt_split_2225)
        opt_mul_2240 = P.Mul()(opt_split_2238_1, opt_split_2225_1)
        opt_add_2241 = opt_mul_2239 + self.add_2241_bias
        opt_add_2242 = P.Add()(opt_add_2241, opt_mul_2240)
        opt_conv2d_2243 = self.conv2d_2243(opt_add_2242)
        opt_add_2244 = P.Add()(opt_conv2d_2243, opt_relu_2220)
        opt_relu_2245 = self.relu_2245(opt_add_2244)
        opt_avgpool2d_2246 = self.avgpool2d_2246(opt_relu_2245)
        opt_reshape_2247 = self.reshape_2247(opt_avgpool2d_2246, self.reshape_2247_shape)
        opt_flatten_2248 = self.flatten_2248(opt_reshape_2247)
        opt_dense_2249 = self.dense_2249(opt_flatten_2248)
        return opt_dense_2249
