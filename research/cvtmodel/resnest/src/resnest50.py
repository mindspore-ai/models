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
                                  out_channels=32,
                                  kernel_size=(3, 3),
                                  stride=(2, 2),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.module22_0 = Module22(module5_0_conv2d_0_in_channels=32,
                                   module5_0_conv2d_0_out_channels=32,
                                   module5_0_conv2d_0_kernel_size=(3, 3),
                                   module5_0_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_0_conv2d_0_pad_mode="pad",
                                   module5_0_conv2d_0_group=1,
                                   module5_1_conv2d_0_in_channels=32,
                                   module5_1_conv2d_0_out_channels=64,
                                   module5_1_conv2d_0_kernel_size=(3, 3),
                                   module5_1_conv2d_0_padding=(1, 1, 1, 1),
                                   module5_1_conv2d_0_pad_mode="pad",
                                   module5_1_conv2d_0_group=1)
        self.pad_maxpool2d_6 = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (1, 0)))
        self.maxpool2d_6 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.module22_1 = Module22(module5_0_conv2d_0_in_channels=64,
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
                                   module0_0_avgpool2d_0_kernel_size=(56, 56),
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
        self.conv2d_10 = nn.Conv2d(in_channels=64,
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
                                   module0_0_avgpool2d_0_kernel_size=(56, 56),
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
                                   module0_0_avgpool2d_0_kernel_size=(56, 56),
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
                                   module0_0_avgpool2d_0_kernel_size=(56, 56),
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
                                   module0_0_avgpool2d_0_kernel_size=(28, 28),
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
                                   module0_0_avgpool2d_0_kernel_size=(28, 28),
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
                                   module0_0_avgpool2d_0_kernel_size=(28, 28),
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
                                   module0_0_avgpool2d_0_kernel_size=(28, 28),
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
                                   module0_0_avgpool2d_0_kernel_size=(14, 14),
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
                                   module0_0_avgpool2d_0_kernel_size=(14, 14),
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
                                    module0_0_avgpool2d_0_kernel_size=(14, 14),
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
                                    module0_0_avgpool2d_0_kernel_size=(14, 14),
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
                                    module0_0_avgpool2d_0_kernel_size=(14, 14),
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
        self.split_348 = P.Split(axis=1, output_num=2)
        self.add_349_bias = 0.0
        self.module20_13 = Module20(reshape_2_shape=[1, 1024],
                                    module0_0_avgpool2d_0_kernel_size=(14, 14),
                                    module0_0_conv2d_1_in_channels=512,
                                    module0_0_conv2d_1_out_channels=256,
                                    module0_0_conv2d_3_in_channels=256,
                                    module0_0_conv2d_3_out_channels=1024,
                                    module0_0_reshape_4_shape=[1, 1, 2, 512])
        self.reshape_360 = P.Reshape()
        self.reshape_360_shape = tuple([1, 1024, 1, 1])
        self.split_361 = P.Split(axis=1, output_num=2)
        self.add_364_bias = 0.0
        self.pad_366 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.pad_avgpool2d_367 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_367 = nn.AvgPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.conv2d_368 = nn.Conv2d(in_channels=512,
                                    out_channels=2048,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.pad_avgpool2d_343 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_343 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.conv2d_345 = nn.Conv2d(in_channels=1024,
                                    out_channels=2048,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_370 = nn.ReLU()
        self.module22_15 = Module22(module5_0_conv2d_0_in_channels=2048,
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
        self.split_375 = P.Split(axis=1, output_num=2)
        self.add_376_bias = 0.0
        self.module20_14 = Module20(reshape_2_shape=[1, 1024],
                                    module0_0_avgpool2d_0_kernel_size=(7, 7),
                                    module0_0_conv2d_1_in_channels=512,
                                    module0_0_conv2d_1_out_channels=256,
                                    module0_0_conv2d_3_in_channels=256,
                                    module0_0_conv2d_3_out_channels=1024,
                                    module0_0_reshape_4_shape=[1, 1, 2, 512])
        self.reshape_387 = P.Reshape()
        self.reshape_387_shape = tuple([1, 1024, 1, 1])
        self.split_388 = P.Split(axis=1, output_num=2)
        self.add_391_bias = 0.0
        self.conv2d_393 = nn.Conv2d(in_channels=512,
                                    out_channels=2048,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_395 = nn.ReLU()
        self.module22_16 = Module22(module5_0_conv2d_0_in_channels=2048,
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
        self.split_400 = P.Split(axis=1, output_num=2)
        self.add_401_bias = 0.0
        self.module20_15 = Module20(reshape_2_shape=[1, 1024],
                                    module0_0_avgpool2d_0_kernel_size=(7, 7),
                                    module0_0_conv2d_1_in_channels=512,
                                    module0_0_conv2d_1_out_channels=256,
                                    module0_0_conv2d_3_in_channels=256,
                                    module0_0_conv2d_3_out_channels=1024,
                                    module0_0_reshape_4_shape=[1, 1, 2, 512])
        self.reshape_412 = P.Reshape()
        self.reshape_412_shape = tuple([1, 1024, 1, 1])
        self.split_413 = P.Split(axis=1, output_num=2)
        self.add_416_bias = 0.0
        self.conv2d_418 = nn.Conv2d(in_channels=512,
                                    out_channels=2048,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_420 = nn.ReLU()
        self.avgpool2d_421 = nn.AvgPool2d(kernel_size=(7, 7))
        self.reshape_422 = P.Reshape()
        self.reshape_422_shape = tuple([1, 2048])
        self.flatten_423 = nn.Flatten()
        self.dense_424 = nn.Dense(in_channels=2048, out_channels=1000, has_bias=True)

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
        opt_split_348, opt_split_348_1 = self.split_348(module22_14_opt)
        opt_add_349 = opt_split_348 + self.add_349_bias
        opt_add_350 = P.Add()(opt_add_349, opt_split_348_1)
        module20_13_opt = self.module20_13(opt_add_350)
        opt_reshape_360 = self.reshape_360(module20_13_opt, self.reshape_360_shape)
        opt_split_361, opt_split_361_1 = self.split_361(opt_reshape_360)
        opt_mul_362 = P.Mul()(opt_split_361, opt_split_348)
        opt_mul_363 = P.Mul()(opt_split_361_1, opt_split_348_1)
        opt_add_364 = opt_mul_362 + self.add_364_bias
        opt_add_365 = P.Add()(opt_add_364, opt_mul_363)
        opt_pad_366 = self.pad_366(opt_add_365)
        opt_avgpool2d_367 = self.pad_avgpool2d_367(opt_pad_366)
        opt_avgpool2d_367 = self.avgpool2d_367(opt_avgpool2d_367)
        opt_conv2d_368 = self.conv2d_368(opt_avgpool2d_367)
        opt_avgpool2d_343 = self.pad_avgpool2d_343(opt_relu_341)
        opt_avgpool2d_343 = self.avgpool2d_343(opt_avgpool2d_343)
        opt_conv2d_345 = self.conv2d_345(opt_avgpool2d_343)
        opt_add_369 = P.Add()(opt_conv2d_368, opt_conv2d_345)
        opt_relu_370 = self.relu_370(opt_add_369)
        module22_15_opt = self.module22_15(opt_relu_370)
        opt_split_375, opt_split_375_1 = self.split_375(module22_15_opt)
        opt_add_376 = opt_split_375 + self.add_376_bias
        opt_add_377 = P.Add()(opt_add_376, opt_split_375_1)
        module20_14_opt = self.module20_14(opt_add_377)
        opt_reshape_387 = self.reshape_387(module20_14_opt, self.reshape_387_shape)
        opt_split_388, opt_split_388_1 = self.split_388(opt_reshape_387)
        opt_mul_389 = P.Mul()(opt_split_388, opt_split_375)
        opt_mul_390 = P.Mul()(opt_split_388_1, opt_split_375_1)
        opt_add_391 = opt_mul_389 + self.add_391_bias
        opt_add_392 = P.Add()(opt_add_391, opt_mul_390)
        opt_conv2d_393 = self.conv2d_393(opt_add_392)
        opt_add_394 = P.Add()(opt_conv2d_393, opt_relu_370)
        opt_relu_395 = self.relu_395(opt_add_394)
        module22_16_opt = self.module22_16(opt_relu_395)
        opt_split_400, opt_split_400_1 = self.split_400(module22_16_opt)
        opt_add_401 = opt_split_400 + self.add_401_bias
        opt_add_402 = P.Add()(opt_add_401, opt_split_400_1)
        module20_15_opt = self.module20_15(opt_add_402)
        opt_reshape_412 = self.reshape_412(module20_15_opt, self.reshape_412_shape)
        opt_split_413, opt_split_413_1 = self.split_413(opt_reshape_412)
        opt_mul_414 = P.Mul()(opt_split_413, opt_split_400)
        opt_mul_415 = P.Mul()(opt_split_413_1, opt_split_400_1)
        opt_add_416 = opt_mul_414 + self.add_416_bias
        opt_add_417 = P.Add()(opt_add_416, opt_mul_415)
        opt_conv2d_418 = self.conv2d_418(opt_add_417)
        opt_add_419 = P.Add()(opt_conv2d_418, opt_relu_395)
        opt_relu_420 = self.relu_420(opt_add_419)
        opt_avgpool2d_421 = self.avgpool2d_421(opt_relu_420)
        opt_reshape_422 = self.reshape_422(opt_avgpool2d_421, self.reshape_422_shape)
        opt_flatten_423 = self.flatten_423(opt_reshape_422)
        opt_dense_424 = self.dense_424(opt_flatten_423)
        return opt_dense_424
