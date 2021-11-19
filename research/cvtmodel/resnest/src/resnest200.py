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
                                   module0_0_avgpool2d_0_kernel_size=(80, 80),
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
                                   module0_0_avgpool2d_0_kernel_size=(80, 80),
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
                                   module0_0_avgpool2d_0_kernel_size=(80, 80),
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
                                   module0_0_avgpool2d_0_kernel_size=(80, 80),
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
                                   module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                   module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                   module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                   module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                   module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                   module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                    module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                    module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                    module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                    module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                    module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                    module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                    module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                    module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                    module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                    module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                    module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                    module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                    module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                    module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                    module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                    module0_0_avgpool2d_0_kernel_size=(40, 40),
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
                                    module0_0_avgpool2d_0_kernel_size=(40, 40),
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
        self.split_694 = P.Split(axis=1, output_num=2)
        self.add_695_bias = 0.0
        self.module20_27 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(40, 40),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_706 = P.Reshape()
        self.reshape_706_shape = tuple([1, 512, 1, 1])
        self.split_707 = P.Split(axis=1, output_num=2)
        self.add_710_bias = 0.0
        self.pad_712 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.pad_avgpool2d_713 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_713 = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2d_714 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_689 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_689 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2d_691 = nn.Conv2d(in_channels=512,
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
        self.split_771 = P.Split(axis=1, output_num=2)
        self.add_772_bias = 0.0
        self.module20_30 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_783 = P.Reshape()
        self.reshape_783_shape = tuple([1, 512, 1, 1])
        self.split_784 = P.Split(axis=1, output_num=2)
        self.add_787_bias = 0.0
        self.conv2d_789 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_791 = nn.ReLU()
        self.module22_32 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_796 = P.Split(axis=1, output_num=2)
        self.add_797_bias = 0.0
        self.module20_31 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_808 = P.Reshape()
        self.reshape_808_shape = tuple([1, 512, 1, 1])
        self.split_809 = P.Split(axis=1, output_num=2)
        self.add_812_bias = 0.0
        self.conv2d_814 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_816 = nn.ReLU()
        self.module22_33 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_821 = P.Split(axis=1, output_num=2)
        self.add_822_bias = 0.0
        self.module20_32 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_833 = P.Reshape()
        self.reshape_833_shape = tuple([1, 512, 1, 1])
        self.split_834 = P.Split(axis=1, output_num=2)
        self.add_837_bias = 0.0
        self.conv2d_839 = nn.Conv2d(in_channels=256,
                                    out_channels=1024,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_841 = nn.ReLU()
        self.module22_34 = Module22(module5_0_conv2d_0_in_channels=1024,
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
        self.split_846 = P.Split(axis=1, output_num=2)
        self.add_847_bias = 0.0
        self.module20_33 = Module20(reshape_2_shape=[1, 512],
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
                                    module0_0_conv2d_1_in_channels=256,
                                    module0_0_conv2d_1_out_channels=128,
                                    module0_0_conv2d_3_in_channels=128,
                                    module0_0_conv2d_3_out_channels=512,
                                    module0_0_reshape_4_shape=[1, 1, 2, 256])
        self.reshape_858 = P.Reshape()
        self.reshape_858_shape = tuple([1, 512, 1, 1])
        self.split_859 = P.Split(axis=1, output_num=2)
        self.add_862_bias = 0.0
        self.conv2d_864 = nn.Conv2d(in_channels=256,
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
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
        self.split_1598 = P.Split(axis=1, output_num=2)
        self.add_1599_bias = 0.0
        self.module20_63 = Module20(reshape_2_shape=[1, 1024],
                                    module0_0_avgpool2d_0_kernel_size=(20, 20),
                                    module0_0_conv2d_1_in_channels=512,
                                    module0_0_conv2d_1_out_channels=256,
                                    module0_0_conv2d_3_in_channels=256,
                                    module0_0_conv2d_3_out_channels=1024,
                                    module0_0_reshape_4_shape=[1, 1, 2, 512])
        self.reshape_1610 = P.Reshape()
        self.reshape_1610_shape = tuple([1, 1024, 1, 1])
        self.split_1611 = P.Split(axis=1, output_num=2)
        self.add_1614_bias = 0.0
        self.pad_1616 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.pad_avgpool2d_1617 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_1617 = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2d_1618 = nn.Conv2d(in_channels=512,
                                     out_channels=2048,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.pad_avgpool2d_1593 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_1593 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2d_1595 = nn.Conv2d(in_channels=1024,
                                     out_channels=2048,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1620 = nn.ReLU()
        self.module22_65 = Module22(module5_0_conv2d_0_in_channels=2048,
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
        self.split_1625 = P.Split(axis=1, output_num=2)
        self.add_1626_bias = 0.0
        self.module20_64 = Module20(reshape_2_shape=[1, 1024],
                                    module0_0_avgpool2d_0_kernel_size=(10, 10),
                                    module0_0_conv2d_1_in_channels=512,
                                    module0_0_conv2d_1_out_channels=256,
                                    module0_0_conv2d_3_in_channels=256,
                                    module0_0_conv2d_3_out_channels=1024,
                                    module0_0_reshape_4_shape=[1, 1, 2, 512])
        self.reshape_1637 = P.Reshape()
        self.reshape_1637_shape = tuple([1, 1024, 1, 1])
        self.split_1638 = P.Split(axis=1, output_num=2)
        self.add_1641_bias = 0.0
        self.conv2d_1643 = nn.Conv2d(in_channels=512,
                                     out_channels=2048,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1645 = nn.ReLU()
        self.module22_66 = Module22(module5_0_conv2d_0_in_channels=2048,
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
        self.split_1650 = P.Split(axis=1, output_num=2)
        self.add_1651_bias = 0.0
        self.module20_65 = Module20(reshape_2_shape=[1, 1024],
                                    module0_0_avgpool2d_0_kernel_size=(10, 10),
                                    module0_0_conv2d_1_in_channels=512,
                                    module0_0_conv2d_1_out_channels=256,
                                    module0_0_conv2d_3_in_channels=256,
                                    module0_0_conv2d_3_out_channels=1024,
                                    module0_0_reshape_4_shape=[1, 1, 2, 512])
        self.reshape_1662 = P.Reshape()
        self.reshape_1662_shape = tuple([1, 1024, 1, 1])
        self.split_1663 = P.Split(axis=1, output_num=2)
        self.add_1666_bias = 0.0
        self.conv2d_1668 = nn.Conv2d(in_channels=512,
                                     out_channels=2048,
                                     kernel_size=(1, 1),
                                     stride=(1, 1),
                                     padding=0,
                                     pad_mode="valid",
                                     dilation=(1, 1),
                                     group=1,
                                     has_bias=True)
        self.relu_1670 = nn.ReLU()
        self.avgpool2d_1671 = nn.AvgPool2d(kernel_size=(10, 10))
        self.reshape_1672 = P.Reshape()
        self.reshape_1672_shape = tuple([1, 2048])
        self.flatten_1673 = nn.Flatten()
        self.dense_1674 = nn.Dense(in_channels=2048, out_channels=1000, has_bias=True)

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
        opt_split_694, opt_split_694_1 = self.split_694(module22_28_opt)
        opt_add_695 = opt_split_694 + self.add_695_bias
        opt_add_696 = P.Add()(opt_add_695, opt_split_694_1)
        module20_27_opt = self.module20_27(opt_add_696)
        opt_reshape_706 = self.reshape_706(module20_27_opt, self.reshape_706_shape)
        opt_split_707, opt_split_707_1 = self.split_707(opt_reshape_706)
        opt_mul_708 = P.Mul()(opt_split_707, opt_split_694)
        opt_mul_709 = P.Mul()(opt_split_707_1, opt_split_694_1)
        opt_add_710 = opt_mul_708 + self.add_710_bias
        opt_add_711 = P.Add()(opt_add_710, opt_mul_709)
        opt_pad_712 = self.pad_712(opt_add_711)
        opt_avgpool2d_713 = self.pad_avgpool2d_713(opt_pad_712)
        opt_avgpool2d_713 = self.avgpool2d_713(opt_avgpool2d_713)
        opt_conv2d_714 = self.conv2d_714(opt_avgpool2d_713)
        opt_avgpool2d_689 = self.pad_avgpool2d_689(opt_relu_687)
        opt_avgpool2d_689 = self.avgpool2d_689(opt_avgpool2d_689)
        opt_conv2d_691 = self.conv2d_691(opt_avgpool2d_689)
        opt_add_715 = P.Add()(opt_conv2d_714, opt_conv2d_691)
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
        opt_split_771, opt_split_771_1 = self.split_771(module22_31_opt)
        opt_add_772 = opt_split_771 + self.add_772_bias
        opt_add_773 = P.Add()(opt_add_772, opt_split_771_1)
        module20_30_opt = self.module20_30(opt_add_773)
        opt_reshape_783 = self.reshape_783(module20_30_opt, self.reshape_783_shape)
        opt_split_784, opt_split_784_1 = self.split_784(opt_reshape_783)
        opt_mul_785 = P.Mul()(opt_split_784, opt_split_771)
        opt_mul_786 = P.Mul()(opt_split_784_1, opt_split_771_1)
        opt_add_787 = opt_mul_785 + self.add_787_bias
        opt_add_788 = P.Add()(opt_add_787, opt_mul_786)
        opt_conv2d_789 = self.conv2d_789(opt_add_788)
        opt_add_790 = P.Add()(opt_conv2d_789, opt_relu_766)
        opt_relu_791 = self.relu_791(opt_add_790)
        module22_32_opt = self.module22_32(opt_relu_791)
        opt_split_796, opt_split_796_1 = self.split_796(module22_32_opt)
        opt_add_797 = opt_split_796 + self.add_797_bias
        opt_add_798 = P.Add()(opt_add_797, opt_split_796_1)
        module20_31_opt = self.module20_31(opt_add_798)
        opt_reshape_808 = self.reshape_808(module20_31_opt, self.reshape_808_shape)
        opt_split_809, opt_split_809_1 = self.split_809(opt_reshape_808)
        opt_mul_810 = P.Mul()(opt_split_809, opt_split_796)
        opt_mul_811 = P.Mul()(opt_split_809_1, opt_split_796_1)
        opt_add_812 = opt_mul_810 + self.add_812_bias
        opt_add_813 = P.Add()(opt_add_812, opt_mul_811)
        opt_conv2d_814 = self.conv2d_814(opt_add_813)
        opt_add_815 = P.Add()(opt_conv2d_814, opt_relu_791)
        opt_relu_816 = self.relu_816(opt_add_815)
        module22_33_opt = self.module22_33(opt_relu_816)
        opt_split_821, opt_split_821_1 = self.split_821(module22_33_opt)
        opt_add_822 = opt_split_821 + self.add_822_bias
        opt_add_823 = P.Add()(opt_add_822, opt_split_821_1)
        module20_32_opt = self.module20_32(opt_add_823)
        opt_reshape_833 = self.reshape_833(module20_32_opt, self.reshape_833_shape)
        opt_split_834, opt_split_834_1 = self.split_834(opt_reshape_833)
        opt_mul_835 = P.Mul()(opt_split_834, opt_split_821)
        opt_mul_836 = P.Mul()(opt_split_834_1, opt_split_821_1)
        opt_add_837 = opt_mul_835 + self.add_837_bias
        opt_add_838 = P.Add()(opt_add_837, opt_mul_836)
        opt_conv2d_839 = self.conv2d_839(opt_add_838)
        opt_add_840 = P.Add()(opt_conv2d_839, opt_relu_816)
        opt_relu_841 = self.relu_841(opt_add_840)
        module22_34_opt = self.module22_34(opt_relu_841)
        opt_split_846, opt_split_846_1 = self.split_846(module22_34_opt)
        opt_add_847 = opt_split_846 + self.add_847_bias
        opt_add_848 = P.Add()(opt_add_847, opt_split_846_1)
        module20_33_opt = self.module20_33(opt_add_848)
        opt_reshape_858 = self.reshape_858(module20_33_opt, self.reshape_858_shape)
        opt_split_859, opt_split_859_1 = self.split_859(opt_reshape_858)
        opt_mul_860 = P.Mul()(opt_split_859, opt_split_846)
        opt_mul_861 = P.Mul()(opt_split_859_1, opt_split_846_1)
        opt_add_862 = opt_mul_860 + self.add_862_bias
        opt_add_863 = P.Add()(opt_add_862, opt_mul_861)
        opt_conv2d_864 = self.conv2d_864(opt_add_863)
        opt_add_865 = P.Add()(opt_conv2d_864, opt_relu_841)
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
        opt_split_1598, opt_split_1598_1 = self.split_1598(module22_64_opt)
        opt_add_1599 = opt_split_1598 + self.add_1599_bias
        opt_add_1600 = P.Add()(opt_add_1599, opt_split_1598_1)
        module20_63_opt = self.module20_63(opt_add_1600)
        opt_reshape_1610 = self.reshape_1610(module20_63_opt, self.reshape_1610_shape)
        opt_split_1611, opt_split_1611_1 = self.split_1611(opt_reshape_1610)
        opt_mul_1612 = P.Mul()(opt_split_1611, opt_split_1598)
        opt_mul_1613 = P.Mul()(opt_split_1611_1, opt_split_1598_1)
        opt_add_1614 = opt_mul_1612 + self.add_1614_bias
        opt_add_1615 = P.Add()(opt_add_1614, opt_mul_1613)
        opt_pad_1616 = self.pad_1616(opt_add_1615)
        opt_avgpool2d_1617 = self.pad_avgpool2d_1617(opt_pad_1616)
        opt_avgpool2d_1617 = self.avgpool2d_1617(opt_avgpool2d_1617)
        opt_conv2d_1618 = self.conv2d_1618(opt_avgpool2d_1617)
        opt_avgpool2d_1593 = self.pad_avgpool2d_1593(opt_relu_1591)
        opt_avgpool2d_1593 = self.avgpool2d_1593(opt_avgpool2d_1593)
        opt_conv2d_1595 = self.conv2d_1595(opt_avgpool2d_1593)
        opt_add_1619 = P.Add()(opt_conv2d_1618, opt_conv2d_1595)
        opt_relu_1620 = self.relu_1620(opt_add_1619)
        module22_65_opt = self.module22_65(opt_relu_1620)
        opt_split_1625, opt_split_1625_1 = self.split_1625(module22_65_opt)
        opt_add_1626 = opt_split_1625 + self.add_1626_bias
        opt_add_1627 = P.Add()(opt_add_1626, opt_split_1625_1)
        module20_64_opt = self.module20_64(opt_add_1627)
        opt_reshape_1637 = self.reshape_1637(module20_64_opt, self.reshape_1637_shape)
        opt_split_1638, opt_split_1638_1 = self.split_1638(opt_reshape_1637)
        opt_mul_1639 = P.Mul()(opt_split_1638, opt_split_1625)
        opt_mul_1640 = P.Mul()(opt_split_1638_1, opt_split_1625_1)
        opt_add_1641 = opt_mul_1639 + self.add_1641_bias
        opt_add_1642 = P.Add()(opt_add_1641, opt_mul_1640)
        opt_conv2d_1643 = self.conv2d_1643(opt_add_1642)
        opt_add_1644 = P.Add()(opt_conv2d_1643, opt_relu_1620)
        opt_relu_1645 = self.relu_1645(opt_add_1644)
        module22_66_opt = self.module22_66(opt_relu_1645)
        opt_split_1650, opt_split_1650_1 = self.split_1650(module22_66_opt)
        opt_add_1651 = opt_split_1650 + self.add_1651_bias
        opt_add_1652 = P.Add()(opt_add_1651, opt_split_1650_1)
        module20_65_opt = self.module20_65(opt_add_1652)
        opt_reshape_1662 = self.reshape_1662(module20_65_opt, self.reshape_1662_shape)
        opt_split_1663, opt_split_1663_1 = self.split_1663(opt_reshape_1662)
        opt_mul_1664 = P.Mul()(opt_split_1663, opt_split_1650)
        opt_mul_1665 = P.Mul()(opt_split_1663_1, opt_split_1650_1)
        opt_add_1666 = opt_mul_1664 + self.add_1666_bias
        opt_add_1667 = P.Add()(opt_add_1666, opt_mul_1665)
        opt_conv2d_1668 = self.conv2d_1668(opt_add_1667)
        opt_add_1669 = P.Add()(opt_conv2d_1668, opt_relu_1645)
        opt_relu_1670 = self.relu_1670(opt_add_1669)
        opt_avgpool2d_1671 = self.avgpool2d_1671(opt_relu_1670)
        opt_reshape_1672 = self.reshape_1672(opt_avgpool2d_1671, self.reshape_1672_shape)
        opt_flatten_1673 = self.flatten_1673(opt_reshape_1672)
        opt_dense_1674 = self.dense_1674(opt_flatten_1673)
        return opt_dense_1674
