import mindspore.ops as P
from mindspore import nn

class Module0(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_2_in_channels, conv2d_2_out_channels,
                 reshape_4_shape, conv2d_7_in_channels, conv2d_7_out_channels, conv2d_9_in_channels,
                 conv2d_9_out_channels, reshape_10_shape, reshape_14_shape, reshape_15_shape, reshape_16_shape):
        super(Module0, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.conv2d_2 = nn.Conv2d(in_channels=conv2d_2_in_channels,
                                  out_channels=conv2d_2_out_channels,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=2,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()
        self.reshape_4 = P.Reshape()
        self.reshape_4_shape = tuple(reshape_4_shape)
        self.reducesum_5 = P.ReduceSum(keep_dims=False)
        self.reducesum_5_axis = 1
        self.reducemean_6 = P.ReduceMean(keep_dims=True)
        self.reducemean_6_axis = (2, 3)
        self.conv2d_7 = nn.Conv2d(in_channels=conv2d_7_in_channels,
                                  out_channels=conv2d_7_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_8 = nn.ReLU()
        self.conv2d_9 = nn.Conv2d(in_channels=conv2d_9_in_channels,
                                  out_channels=conv2d_9_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.reshape_10 = P.Reshape()
        self.reshape_10_shape = tuple(reshape_10_shape)
        self.transpose_11 = P.Transpose()
        self.softmax_12 = nn.Softmax(axis=3)
        self.transpose_13 = P.Transpose()
        self.reshape_14 = P.Reshape()
        self.reshape_14_shape = tuple(reshape_14_shape)
        self.reshape_15 = P.Reshape()
        self.reshape_15_shape = tuple(reshape_15_shape)
        self.reshape_16 = P.Reshape()
        self.reshape_16_shape = tuple(reshape_16_shape)

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        opt_reshape_4 = self.reshape_4(opt_relu_3, self.reshape_4_shape)
        opt_reducesum_5 = self.reducesum_5(opt_reshape_4, self.reducesum_5_axis)
        opt_reducemean_6 = self.reducemean_6(opt_reducesum_5, self.reducemean_6_axis)
        opt_conv2d_7 = self.conv2d_7(opt_reducemean_6)
        opt_relu_8 = self.relu_8(opt_conv2d_7)
        opt_conv2d_9 = self.conv2d_9(opt_relu_8)
        opt_reshape_10 = self.reshape_10(opt_conv2d_9, self.reshape_10_shape)
        opt_transpose_11 = self.transpose_11(opt_reshape_10, (0, 3, 1, 2))
        opt_softmax_12 = self.softmax_12(opt_transpose_11)
        opt_transpose_13 = self.transpose_13(opt_softmax_12, (0, 3, 2, 1))
        opt_reshape_14 = self.reshape_14(opt_transpose_13, self.reshape_14_shape)
        opt_reshape_15 = self.reshape_15(opt_reshape_14, self.reshape_15_shape)
        opt_reshape_16 = self.reshape_16(opt_reshape_15, self.reshape_16_shape)
        opt_mul_17 = P.Mul()(opt_reshape_4, opt_reshape_16)
        return opt_mul_17


class Module4(nn.Cell):
    def __init__(self):
        super(Module4, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=64,
                                 conv2d_2_in_channels=64,
                                 conv2d_2_out_channels=128,
                                 reshape_4_shape=[1, 2, 64, 64, 64],
                                 conv2d_7_in_channels=64,
                                 conv2d_7_out_channels=32,
                                 conv2d_9_in_channels=32,
                                 conv2d_9_out_channels=128,
                                 reshape_10_shape=[1, 1, 2, 64],
                                 reshape_14_shape=[1, 128],
                                 reshape_15_shape=[1, 128, 1, 1],
                                 reshape_16_shape=[1, 2, 64, 1, 1])
        self.reducesum_0 = P.ReduceSum(keep_dims=False)
        self.reducesum_0_axis = 1
        self.conv2d_1 = nn.Conv2d(in_channels=64,
                                  out_channels=256,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_reducesum_0 = self.reducesum_0(module0_0_opt, self.reducesum_0_axis)
        opt_conv2d_1 = self.conv2d_1(opt_reducesum_0)
        return opt_conv2d_1


class Module16(nn.Cell):
    def __init__(self, conv2d_1_in_channels, conv2d_1_out_channels, conv2d_5_in_channels, conv2d_5_out_channels,
                 module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_reshape_4_shape, module0_0_conv2d_7_in_channels,
                 module0_0_conv2d_7_out_channels, module0_0_conv2d_9_in_channels, module0_0_conv2d_9_out_channels,
                 module0_0_reshape_10_shape, module0_0_reshape_14_shape, module0_0_reshape_15_shape,
                 module0_0_reshape_16_shape, module0_1_conv2d_0_in_channels, module0_1_conv2d_0_out_channels,
                 module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels, module0_1_reshape_4_shape,
                 module0_1_conv2d_7_in_channels, module0_1_conv2d_7_out_channels, module0_1_conv2d_9_in_channels,
                 module0_1_conv2d_9_out_channels, module0_1_reshape_10_shape, module0_1_reshape_14_shape,
                 module0_1_reshape_15_shape, module0_1_reshape_16_shape):
        super(Module16, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 reshape_4_shape=module0_0_reshape_4_shape,
                                 conv2d_7_in_channels=module0_0_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_0_conv2d_7_out_channels,
                                 conv2d_9_in_channels=module0_0_conv2d_9_in_channels,
                                 conv2d_9_out_channels=module0_0_conv2d_9_out_channels,
                                 reshape_10_shape=module0_0_reshape_10_shape,
                                 reshape_14_shape=module0_0_reshape_14_shape,
                                 reshape_15_shape=module0_0_reshape_15_shape,
                                 reshape_16_shape=module0_0_reshape_16_shape)
        self.reducesum_0 = P.ReduceSum(keep_dims=False)
        self.reducesum_0_axis = 1
        self.conv2d_1 = nn.Conv2d(in_channels=conv2d_1_in_channels,
                                  out_channels=conv2d_1_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_1_conv2d_2_out_channels,
                                 reshape_4_shape=module0_1_reshape_4_shape,
                                 conv2d_7_in_channels=module0_1_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_1_conv2d_7_out_channels,
                                 conv2d_9_in_channels=module0_1_conv2d_9_in_channels,
                                 conv2d_9_out_channels=module0_1_conv2d_9_out_channels,
                                 reshape_10_shape=module0_1_reshape_10_shape,
                                 reshape_14_shape=module0_1_reshape_14_shape,
                                 reshape_15_shape=module0_1_reshape_15_shape,
                                 reshape_16_shape=module0_1_reshape_16_shape)
        self.reducesum_4 = P.ReduceSum(keep_dims=False)
        self.reducesum_4_axis = 1
        self.conv2d_5 = nn.Conv2d(in_channels=conv2d_5_in_channels,
                                  out_channels=conv2d_5_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_7 = nn.ReLU()

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_reducesum_0 = self.reducesum_0(module0_0_opt, self.reducesum_0_axis)
        opt_conv2d_1 = self.conv2d_1(opt_reducesum_0)
        opt_add_2 = P.Add()(opt_conv2d_1, x)
        opt_relu_3 = self.relu_3(opt_add_2)
        module0_1_opt = self.module0_1(opt_relu_3)
        opt_reducesum_4 = self.reducesum_4(module0_1_opt, self.reducesum_4_axis)
        opt_conv2d_5 = self.conv2d_5(opt_reducesum_4)
        opt_add_6 = P.Add()(opt_conv2d_5, opt_relu_3)
        opt_relu_7 = self.relu_7(opt_add_6)
        return opt_relu_7


class Module10(nn.Cell):
    def __init__(self, avgpool2d_0_kernel_size, conv2d_1_in_channels, conv2d_1_out_channels):
        super(Module10, self).__init__()
        self.pad_avgpool2d_0 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_0 = nn.AvgPool2d(kernel_size=avgpool2d_0_kernel_size, stride=(2, 2))
        self.conv2d_1 = nn.Conv2d(in_channels=conv2d_1_in_channels,
                                  out_channels=conv2d_1_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        opt_avgpool2d_0 = self.pad_avgpool2d_0(x)
        opt_avgpool2d_0 = self.avgpool2d_0(opt_avgpool2d_0)
        opt_conv2d_1 = self.conv2d_1(opt_avgpool2d_0)
        return opt_conv2d_1


class Module15(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_reshape_4_shape, module0_0_conv2d_7_in_channels,
                 module0_0_conv2d_7_out_channels, module0_0_conv2d_9_in_channels, module0_0_conv2d_9_out_channels,
                 module0_0_reshape_10_shape, module0_0_reshape_14_shape, module0_0_reshape_15_shape,
                 module0_0_reshape_16_shape, module10_0_avgpool2d_0_kernel_size, module10_0_conv2d_1_in_channels,
                 module10_0_conv2d_1_out_channels):
        super(Module15, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 reshape_4_shape=module0_0_reshape_4_shape,
                                 conv2d_7_in_channels=module0_0_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_0_conv2d_7_out_channels,
                                 conv2d_9_in_channels=module0_0_conv2d_9_in_channels,
                                 conv2d_9_out_channels=module0_0_conv2d_9_out_channels,
                                 reshape_10_shape=module0_0_reshape_10_shape,
                                 reshape_14_shape=module0_0_reshape_14_shape,
                                 reshape_15_shape=module0_0_reshape_15_shape,
                                 reshape_16_shape=module0_0_reshape_16_shape)
        self.reducesum_0 = P.ReduceSum(keep_dims=False)
        self.reducesum_0_axis = 1
        self.pad_1 = nn.Pad(paddings=((0, 0), (0, 0), (1, 1), (1, 1)), mode="CONSTANT")
        self.module10_0 = Module10(avgpool2d_0_kernel_size=module10_0_avgpool2d_0_kernel_size,
                                   conv2d_1_in_channels=module10_0_conv2d_1_in_channels,
                                   conv2d_1_out_channels=module10_0_conv2d_1_out_channels)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_reducesum_0 = self.reducesum_0(module0_0_opt, self.reducesum_0_axis)
        opt_pad_1 = self.pad_1(opt_reducesum_0)
        module10_0_opt = self.module10_0(opt_pad_1)
        return module10_0_opt


class Module6(nn.Cell):
    def __init__(self):
        super(Module6, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=128,
                                 conv2d_2_in_channels=128,
                                 conv2d_2_out_channels=256,
                                 reshape_4_shape=[1, 2, 128, 32, 32],
                                 conv2d_7_in_channels=128,
                                 conv2d_7_out_channels=64,
                                 conv2d_9_in_channels=64,
                                 conv2d_9_out_channels=256,
                                 reshape_10_shape=[1, 1, 2, 128],
                                 reshape_14_shape=[1, 256],
                                 reshape_15_shape=[1, 256, 1, 1],
                                 reshape_16_shape=[1, 2, 128, 1, 1])
        self.reducesum_0 = P.ReduceSum(keep_dims=False)
        self.reducesum_0_axis = 1
        self.conv2d_1 = nn.Conv2d(in_channels=128,
                                  out_channels=512,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_reducesum_0 = self.reducesum_0(module0_0_opt, self.reducesum_0_axis)
        opt_conv2d_1 = self.conv2d_1(opt_reducesum_0)
        opt_add_2 = P.Add()(opt_conv2d_1, x)
        opt_relu_3 = self.relu_3(opt_add_2)
        return opt_relu_3


class Module44(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_reshape_4_shape, module0_0_conv2d_7_in_channels,
                 module0_0_conv2d_7_out_channels, module0_0_conv2d_9_in_channels, module0_0_conv2d_9_out_channels,
                 module0_0_reshape_10_shape, module0_0_reshape_14_shape, module0_0_reshape_15_shape,
                 module0_0_reshape_16_shape, module0_1_conv2d_0_in_channels, module0_1_conv2d_0_out_channels,
                 module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels, module0_1_reshape_4_shape,
                 module0_1_conv2d_7_in_channels, module0_1_conv2d_7_out_channels, module0_1_conv2d_9_in_channels,
                 module0_1_conv2d_9_out_channels, module0_1_reshape_10_shape, module0_1_reshape_14_shape,
                 module0_1_reshape_15_shape, module0_1_reshape_16_shape, module0_2_conv2d_0_in_channels,
                 module0_2_conv2d_0_out_channels, module0_2_conv2d_2_in_channels, module0_2_conv2d_2_out_channels,
                 module0_2_reshape_4_shape, module0_2_conv2d_7_in_channels, module0_2_conv2d_7_out_channels,
                 module0_2_conv2d_9_in_channels, module0_2_conv2d_9_out_channels, module0_2_reshape_10_shape,
                 module0_2_reshape_14_shape, module0_2_reshape_15_shape, module0_2_reshape_16_shape,
                 module0_3_conv2d_0_in_channels, module0_3_conv2d_0_out_channels, module0_3_conv2d_2_in_channels,
                 module0_3_conv2d_2_out_channels, module0_3_reshape_4_shape, module0_3_conv2d_7_in_channels,
                 module0_3_conv2d_7_out_channels, module0_3_conv2d_9_in_channels, module0_3_conv2d_9_out_channels,
                 module0_3_reshape_10_shape, module0_3_reshape_14_shape, module0_3_reshape_15_shape,
                 module0_3_reshape_16_shape, module0_4_conv2d_0_in_channels, module0_4_conv2d_0_out_channels,
                 module0_4_conv2d_2_in_channels, module0_4_conv2d_2_out_channels, module0_4_reshape_4_shape,
                 module0_4_conv2d_7_in_channels, module0_4_conv2d_7_out_channels, module0_4_conv2d_9_in_channels,
                 module0_4_conv2d_9_out_channels, module0_4_reshape_10_shape, module0_4_reshape_14_shape,
                 module0_4_reshape_15_shape, module0_4_reshape_16_shape, module0_5_conv2d_0_in_channels,
                 module0_5_conv2d_0_out_channels, module0_5_conv2d_2_in_channels, module0_5_conv2d_2_out_channels,
                 module0_5_reshape_4_shape, module0_5_conv2d_7_in_channels, module0_5_conv2d_7_out_channels,
                 module0_5_conv2d_9_in_channels, module0_5_conv2d_9_out_channels, module0_5_reshape_10_shape,
                 module0_5_reshape_14_shape, module0_5_reshape_15_shape, module0_5_reshape_16_shape,
                 module0_6_conv2d_0_in_channels, module0_6_conv2d_0_out_channels, module0_6_conv2d_2_in_channels,
                 module0_6_conv2d_2_out_channels, module0_6_reshape_4_shape, module0_6_conv2d_7_in_channels,
                 module0_6_conv2d_7_out_channels, module0_6_conv2d_9_in_channels, module0_6_conv2d_9_out_channels,
                 module0_6_reshape_10_shape, module0_6_reshape_14_shape, module0_6_reshape_15_shape,
                 module0_6_reshape_16_shape, module0_7_conv2d_0_in_channels, module0_7_conv2d_0_out_channels,
                 module0_7_conv2d_2_in_channels, module0_7_conv2d_2_out_channels, module0_7_reshape_4_shape,
                 module0_7_conv2d_7_in_channels, module0_7_conv2d_7_out_channels, module0_7_conv2d_9_in_channels,
                 module0_7_conv2d_9_out_channels, module0_7_reshape_10_shape, module0_7_reshape_14_shape,
                 module0_7_reshape_15_shape, module0_7_reshape_16_shape):
        super(Module44, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 reshape_4_shape=module0_0_reshape_4_shape,
                                 conv2d_7_in_channels=module0_0_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_0_conv2d_7_out_channels,
                                 conv2d_9_in_channels=module0_0_conv2d_9_in_channels,
                                 conv2d_9_out_channels=module0_0_conv2d_9_out_channels,
                                 reshape_10_shape=module0_0_reshape_10_shape,
                                 reshape_14_shape=module0_0_reshape_14_shape,
                                 reshape_15_shape=module0_0_reshape_15_shape,
                                 reshape_16_shape=module0_0_reshape_16_shape)
        self.reducesum_0 = P.ReduceSum(keep_dims=False)
        self.reducesum_0_axis = 1
        self.conv2d_1 = nn.Conv2d(in_channels=256,
                                  out_channels=1024,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_1_conv2d_2_out_channels,
                                 reshape_4_shape=module0_1_reshape_4_shape,
                                 conv2d_7_in_channels=module0_1_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_1_conv2d_7_out_channels,
                                 conv2d_9_in_channels=module0_1_conv2d_9_in_channels,
                                 conv2d_9_out_channels=module0_1_conv2d_9_out_channels,
                                 reshape_10_shape=module0_1_reshape_10_shape,
                                 reshape_14_shape=module0_1_reshape_14_shape,
                                 reshape_15_shape=module0_1_reshape_15_shape,
                                 reshape_16_shape=module0_1_reshape_16_shape)
        self.reducesum_4 = P.ReduceSum(keep_dims=False)
        self.reducesum_4_axis = 1
        self.conv2d_5 = nn.Conv2d(in_channels=256,
                                  out_channels=1024,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_7 = nn.ReLU()
        self.module0_2 = Module0(conv2d_0_in_channels=module0_2_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_2_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_2_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_2_conv2d_2_out_channels,
                                 reshape_4_shape=module0_2_reshape_4_shape,
                                 conv2d_7_in_channels=module0_2_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_2_conv2d_7_out_channels,
                                 conv2d_9_in_channels=module0_2_conv2d_9_in_channels,
                                 conv2d_9_out_channels=module0_2_conv2d_9_out_channels,
                                 reshape_10_shape=module0_2_reshape_10_shape,
                                 reshape_14_shape=module0_2_reshape_14_shape,
                                 reshape_15_shape=module0_2_reshape_15_shape,
                                 reshape_16_shape=module0_2_reshape_16_shape)
        self.reducesum_8 = P.ReduceSum(keep_dims=False)
        self.reducesum_8_axis = 1
        self.conv2d_9 = nn.Conv2d(in_channels=256,
                                  out_channels=1024,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_11 = nn.ReLU()
        self.module0_3 = Module0(conv2d_0_in_channels=module0_3_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_3_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_3_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_3_conv2d_2_out_channels,
                                 reshape_4_shape=module0_3_reshape_4_shape,
                                 conv2d_7_in_channels=module0_3_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_3_conv2d_7_out_channels,
                                 conv2d_9_in_channels=module0_3_conv2d_9_in_channels,
                                 conv2d_9_out_channels=module0_3_conv2d_9_out_channels,
                                 reshape_10_shape=module0_3_reshape_10_shape,
                                 reshape_14_shape=module0_3_reshape_14_shape,
                                 reshape_15_shape=module0_3_reshape_15_shape,
                                 reshape_16_shape=module0_3_reshape_16_shape)
        self.reducesum_12 = P.ReduceSum(keep_dims=False)
        self.reducesum_12_axis = 1
        self.conv2d_13 = nn.Conv2d(in_channels=256,
                                   out_channels=1024,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_15 = nn.ReLU()
        self.module0_4 = Module0(conv2d_0_in_channels=module0_4_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_4_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_4_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_4_conv2d_2_out_channels,
                                 reshape_4_shape=module0_4_reshape_4_shape,
                                 conv2d_7_in_channels=module0_4_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_4_conv2d_7_out_channels,
                                 conv2d_9_in_channels=module0_4_conv2d_9_in_channels,
                                 conv2d_9_out_channels=module0_4_conv2d_9_out_channels,
                                 reshape_10_shape=module0_4_reshape_10_shape,
                                 reshape_14_shape=module0_4_reshape_14_shape,
                                 reshape_15_shape=module0_4_reshape_15_shape,
                                 reshape_16_shape=module0_4_reshape_16_shape)
        self.reducesum_16 = P.ReduceSum(keep_dims=False)
        self.reducesum_16_axis = 1
        self.conv2d_17 = nn.Conv2d(in_channels=256,
                                   out_channels=1024,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_19 = nn.ReLU()
        self.module0_5 = Module0(conv2d_0_in_channels=module0_5_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_5_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_5_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_5_conv2d_2_out_channels,
                                 reshape_4_shape=module0_5_reshape_4_shape,
                                 conv2d_7_in_channels=module0_5_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_5_conv2d_7_out_channels,
                                 conv2d_9_in_channels=module0_5_conv2d_9_in_channels,
                                 conv2d_9_out_channels=module0_5_conv2d_9_out_channels,
                                 reshape_10_shape=module0_5_reshape_10_shape,
                                 reshape_14_shape=module0_5_reshape_14_shape,
                                 reshape_15_shape=module0_5_reshape_15_shape,
                                 reshape_16_shape=module0_5_reshape_16_shape)
        self.reducesum_20 = P.ReduceSum(keep_dims=False)
        self.reducesum_20_axis = 1
        self.conv2d_21 = nn.Conv2d(in_channels=256,
                                   out_channels=1024,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_23 = nn.ReLU()
        self.module0_6 = Module0(conv2d_0_in_channels=module0_6_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_6_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_6_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_6_conv2d_2_out_channels,
                                 reshape_4_shape=module0_6_reshape_4_shape,
                                 conv2d_7_in_channels=module0_6_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_6_conv2d_7_out_channels,
                                 conv2d_9_in_channels=module0_6_conv2d_9_in_channels,
                                 conv2d_9_out_channels=module0_6_conv2d_9_out_channels,
                                 reshape_10_shape=module0_6_reshape_10_shape,
                                 reshape_14_shape=module0_6_reshape_14_shape,
                                 reshape_15_shape=module0_6_reshape_15_shape,
                                 reshape_16_shape=module0_6_reshape_16_shape)
        self.reducesum_24 = P.ReduceSum(keep_dims=False)
        self.reducesum_24_axis = 1
        self.conv2d_25 = nn.Conv2d(in_channels=256,
                                   out_channels=1024,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_27 = nn.ReLU()
        self.module0_7 = Module0(conv2d_0_in_channels=module0_7_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_7_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_7_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_7_conv2d_2_out_channels,
                                 reshape_4_shape=module0_7_reshape_4_shape,
                                 conv2d_7_in_channels=module0_7_conv2d_7_in_channels,
                                 conv2d_7_out_channels=module0_7_conv2d_7_out_channels,
                                 conv2d_9_in_channels=module0_7_conv2d_9_in_channels,
                                 conv2d_9_out_channels=module0_7_conv2d_9_out_channels,
                                 reshape_10_shape=module0_7_reshape_10_shape,
                                 reshape_14_shape=module0_7_reshape_14_shape,
                                 reshape_15_shape=module0_7_reshape_15_shape,
                                 reshape_16_shape=module0_7_reshape_16_shape)
        self.reducesum_28 = P.ReduceSum(keep_dims=False)
        self.reducesum_28_axis = 1
        self.conv2d_29 = nn.Conv2d(in_channels=256,
                                   out_channels=1024,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_31 = nn.ReLU()

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_reducesum_0 = self.reducesum_0(module0_0_opt, self.reducesum_0_axis)
        opt_conv2d_1 = self.conv2d_1(opt_reducesum_0)
        opt_add_2 = P.Add()(opt_conv2d_1, x)
        opt_relu_3 = self.relu_3(opt_add_2)
        module0_1_opt = self.module0_1(opt_relu_3)
        opt_reducesum_4 = self.reducesum_4(module0_1_opt, self.reducesum_4_axis)
        opt_conv2d_5 = self.conv2d_5(opt_reducesum_4)
        opt_add_6 = P.Add()(opt_conv2d_5, opt_relu_3)
        opt_relu_7 = self.relu_7(opt_add_6)
        module0_2_opt = self.module0_2(opt_relu_7)
        opt_reducesum_8 = self.reducesum_8(module0_2_opt, self.reducesum_8_axis)
        opt_conv2d_9 = self.conv2d_9(opt_reducesum_8)
        opt_add_10 = P.Add()(opt_conv2d_9, opt_relu_7)
        opt_relu_11 = self.relu_11(opt_add_10)
        module0_3_opt = self.module0_3(opt_relu_11)
        opt_reducesum_12 = self.reducesum_12(module0_3_opt, self.reducesum_12_axis)
        opt_conv2d_13 = self.conv2d_13(opt_reducesum_12)
        opt_add_14 = P.Add()(opt_conv2d_13, opt_relu_11)
        opt_relu_15 = self.relu_15(opt_add_14)
        module0_4_opt = self.module0_4(opt_relu_15)
        opt_reducesum_16 = self.reducesum_16(module0_4_opt, self.reducesum_16_axis)
        opt_conv2d_17 = self.conv2d_17(opt_reducesum_16)
        opt_add_18 = P.Add()(opt_conv2d_17, opt_relu_15)
        opt_relu_19 = self.relu_19(opt_add_18)
        module0_5_opt = self.module0_5(opt_relu_19)
        opt_reducesum_20 = self.reducesum_20(module0_5_opt, self.reducesum_20_axis)
        opt_conv2d_21 = self.conv2d_21(opt_reducesum_20)
        opt_add_22 = P.Add()(opt_conv2d_21, opt_relu_19)
        opt_relu_23 = self.relu_23(opt_add_22)
        module0_6_opt = self.module0_6(opt_relu_23)
        opt_reducesum_24 = self.reducesum_24(module0_6_opt, self.reducesum_24_axis)
        opt_conv2d_25 = self.conv2d_25(opt_reducesum_24)
        opt_add_26 = P.Add()(opt_conv2d_25, opt_relu_23)
        opt_relu_27 = self.relu_27(opt_add_26)
        module0_7_opt = self.module0_7(opt_relu_27)
        opt_reducesum_28 = self.reducesum_28(module0_7_opt, self.reducesum_28_axis)
        opt_conv2d_29 = self.conv2d_29(opt_reducesum_28)
        opt_add_30 = P.Add()(opt_conv2d_29, opt_relu_27)
        opt_relu_31 = self.relu_31(opt_add_30)
        return opt_relu_31


class Module43(nn.Cell):
    def __init__(self):
        super(Module43, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=512,
                                 reshape_4_shape=[1, 2, 256, 16, 16],
                                 conv2d_7_in_channels=256,
                                 conv2d_7_out_channels=128,
                                 conv2d_9_in_channels=128,
                                 conv2d_9_out_channels=512,
                                 reshape_10_shape=[1, 1, 2, 256],
                                 reshape_14_shape=[1, 512],
                                 reshape_15_shape=[1, 512, 1, 1],
                                 reshape_16_shape=[1, 2, 256, 1, 1])
        self.reducesum_0 = P.ReduceSum(keep_dims=False)
        self.reducesum_0_axis = 1
        self.conv2d_1 = nn.Conv2d(in_channels=256,
                                  out_channels=1024,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()
        self.module0_1 = Module0(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=512,
                                 reshape_4_shape=[1, 2, 256, 16, 16],
                                 conv2d_7_in_channels=256,
                                 conv2d_7_out_channels=128,
                                 conv2d_9_in_channels=128,
                                 conv2d_9_out_channels=512,
                                 reshape_10_shape=[1, 1, 2, 256],
                                 reshape_14_shape=[1, 512],
                                 reshape_15_shape=[1, 512, 1, 1],
                                 reshape_16_shape=[1, 2, 256, 1, 1])
        self.reducesum_4 = P.ReduceSum(keep_dims=False)
        self.reducesum_4_axis = 1
        self.conv2d_5 = nn.Conv2d(in_channels=256,
                                  out_channels=1024,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_7 = nn.ReLU()
        self.module0_2 = Module0(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=512,
                                 reshape_4_shape=[1, 2, 256, 16, 16],
                                 conv2d_7_in_channels=256,
                                 conv2d_7_out_channels=128,
                                 conv2d_9_in_channels=128,
                                 conv2d_9_out_channels=512,
                                 reshape_10_shape=[1, 1, 2, 256],
                                 reshape_14_shape=[1, 512],
                                 reshape_15_shape=[1, 512, 1, 1],
                                 reshape_16_shape=[1, 2, 256, 1, 1])
        self.reducesum_8 = P.ReduceSum(keep_dims=False)
        self.reducesum_8_axis = 1
        self.conv2d_9 = nn.Conv2d(in_channels=256,
                                  out_channels=1024,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_11 = nn.ReLU()
        self.module0_3 = Module0(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=512,
                                 reshape_4_shape=[1, 2, 256, 16, 16],
                                 conv2d_7_in_channels=256,
                                 conv2d_7_out_channels=128,
                                 conv2d_9_in_channels=128,
                                 conv2d_9_out_channels=512,
                                 reshape_10_shape=[1, 1, 2, 256],
                                 reshape_14_shape=[1, 512],
                                 reshape_15_shape=[1, 512, 1, 1],
                                 reshape_16_shape=[1, 2, 256, 1, 1])
        self.reducesum_12 = P.ReduceSum(keep_dims=False)
        self.reducesum_12_axis = 1
        self.conv2d_13 = nn.Conv2d(in_channels=256,
                                   out_channels=1024,
                                   kernel_size=(1, 1),
                                   stride=(1, 1),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_15 = nn.ReLU()

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_reducesum_0 = self.reducesum_0(module0_0_opt, self.reducesum_0_axis)
        opt_conv2d_1 = self.conv2d_1(opt_reducesum_0)
        opt_add_2 = P.Add()(opt_conv2d_1, x)
        opt_relu_3 = self.relu_3(opt_add_2)
        module0_1_opt = self.module0_1(opt_relu_3)
        opt_reducesum_4 = self.reducesum_4(module0_1_opt, self.reducesum_4_axis)
        opt_conv2d_5 = self.conv2d_5(opt_reducesum_4)
        opt_add_6 = P.Add()(opt_conv2d_5, opt_relu_3)
        opt_relu_7 = self.relu_7(opt_add_6)
        module0_2_opt = self.module0_2(opt_relu_7)
        opt_reducesum_8 = self.reducesum_8(module0_2_opt, self.reducesum_8_axis)
        opt_conv2d_9 = self.conv2d_9(opt_reducesum_8)
        opt_add_10 = P.Add()(opt_conv2d_9, opt_relu_7)
        opt_relu_11 = self.relu_11(opt_add_10)
        module0_3_opt = self.module0_3(opt_relu_11)
        opt_reducesum_12 = self.reducesum_12(module0_3_opt, self.reducesum_12_axis)
        opt_conv2d_13 = self.conv2d_13(opt_reducesum_12)
        opt_add_14 = P.Add()(opt_conv2d_13, opt_relu_11)
        opt_relu_15 = self.relu_15(opt_add_14)
        return opt_relu_15


class MindSporeModel(nn.Cell):
    def __init__(self):
        super(MindSporeModel, self).__init__()
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
        self.conv2d_2 = nn.Conv2d(in_channels=64,
                                  out_channels=64,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()
        self.conv2d_4 = nn.Conv2d(in_channels=64,
                                  out_channels=128,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_5 = nn.ReLU()
        self.pad_maxpool2d_6 = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (1, 0)))
        self.maxpool2d_6 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.module4_0 = Module4()
        self.conv2d_8 = nn.Conv2d(in_channels=128,
                                  out_channels=256,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_29 = nn.ReLU()
        self.module16_0 = Module16(conv2d_1_in_channels=64,
                                   conv2d_1_out_channels=256,
                                   conv2d_5_in_channels=64,
                                   conv2d_5_out_channels=256,
                                   module0_0_conv2d_0_in_channels=256,
                                   module0_0_conv2d_0_out_channels=64,
                                   module0_0_conv2d_2_in_channels=64,
                                   module0_0_conv2d_2_out_channels=128,
                                   module0_0_reshape_4_shape=[1, 2, 64, 64, 64],
                                   module0_0_conv2d_7_in_channels=64,
                                   module0_0_conv2d_7_out_channels=32,
                                   module0_0_conv2d_9_in_channels=32,
                                   module0_0_conv2d_9_out_channels=128,
                                   module0_0_reshape_10_shape=[1, 1, 2, 64],
                                   module0_0_reshape_14_shape=[1, 128],
                                   module0_0_reshape_15_shape=[1, 128, 1, 1],
                                   module0_0_reshape_16_shape=[1, 2, 64, 1, 1],
                                   module0_1_conv2d_0_in_channels=256,
                                   module0_1_conv2d_0_out_channels=64,
                                   module0_1_conv2d_2_in_channels=64,
                                   module0_1_conv2d_2_out_channels=128,
                                   module0_1_reshape_4_shape=[1, 2, 64, 64, 64],
                                   module0_1_conv2d_7_in_channels=64,
                                   module0_1_conv2d_7_out_channels=32,
                                   module0_1_conv2d_9_in_channels=32,
                                   module0_1_conv2d_9_out_channels=128,
                                   module0_1_reshape_10_shape=[1, 1, 2, 64],
                                   module0_1_reshape_14_shape=[1, 128],
                                   module0_1_reshape_15_shape=[1, 128, 1, 1],
                                   module0_1_reshape_16_shape=[1, 2, 64, 1, 1])
        self.module15_0 = Module15(module0_0_conv2d_0_in_channels=256,
                                   module0_0_conv2d_0_out_channels=128,
                                   module0_0_conv2d_2_in_channels=128,
                                   module0_0_conv2d_2_out_channels=256,
                                   module0_0_reshape_4_shape=[1, 2, 128, 64, 64],
                                   module0_0_conv2d_7_in_channels=128,
                                   module0_0_conv2d_7_out_channels=64,
                                   module0_0_conv2d_9_in_channels=64,
                                   module0_0_conv2d_9_out_channels=256,
                                   module0_0_reshape_10_shape=[1, 1, 2, 128],
                                   module0_0_reshape_14_shape=[1, 256],
                                   module0_0_reshape_15_shape=[1, 256, 1, 1],
                                   module0_0_reshape_16_shape=[1, 2, 128, 1, 1],
                                   module10_0_avgpool2d_0_kernel_size=(3, 3),
                                   module10_0_conv2d_1_in_channels=128,
                                   module10_0_conv2d_1_out_channels=512)
        self.module10_0 = Module10(avgpool2d_0_kernel_size=(2, 2), conv2d_1_in_channels=256, conv2d_1_out_channels=512)
        self.relu_99 = nn.ReLU()
        self.module16_1 = Module16(conv2d_1_in_channels=128,
                                   conv2d_1_out_channels=512,
                                   conv2d_5_in_channels=128,
                                   conv2d_5_out_channels=512,
                                   module0_0_conv2d_0_in_channels=512,
                                   module0_0_conv2d_0_out_channels=128,
                                   module0_0_conv2d_2_in_channels=128,
                                   module0_0_conv2d_2_out_channels=256,
                                   module0_0_reshape_4_shape=[1, 2, 128, 32, 32],
                                   module0_0_conv2d_7_in_channels=128,
                                   module0_0_conv2d_7_out_channels=64,
                                   module0_0_conv2d_9_in_channels=64,
                                   module0_0_conv2d_9_out_channels=256,
                                   module0_0_reshape_10_shape=[1, 1, 2, 128],
                                   module0_0_reshape_14_shape=[1, 256],
                                   module0_0_reshape_15_shape=[1, 256, 1, 1],
                                   module0_0_reshape_16_shape=[1, 2, 128, 1, 1],
                                   module0_1_conv2d_0_in_channels=512,
                                   module0_1_conv2d_0_out_channels=128,
                                   module0_1_conv2d_2_in_channels=128,
                                   module0_1_conv2d_2_out_channels=256,
                                   module0_1_reshape_4_shape=[1, 2, 128, 32, 32],
                                   module0_1_conv2d_7_in_channels=128,
                                   module0_1_conv2d_7_out_channels=64,
                                   module0_1_conv2d_9_in_channels=64,
                                   module0_1_conv2d_9_out_channels=256,
                                   module0_1_reshape_10_shape=[1, 1, 2, 128],
                                   module0_1_reshape_14_shape=[1, 256],
                                   module0_1_reshape_15_shape=[1, 256, 1, 1],
                                   module0_1_reshape_16_shape=[1, 2, 128, 1, 1])
        self.module6_0 = Module6()
        self.module15_1 = Module15(module0_0_conv2d_0_in_channels=512,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_2_in_channels=256,
                                   module0_0_conv2d_2_out_channels=512,
                                   module0_0_reshape_4_shape=[1, 2, 256, 32, 32],
                                   module0_0_conv2d_7_in_channels=256,
                                   module0_0_conv2d_7_out_channels=128,
                                   module0_0_conv2d_9_in_channels=128,
                                   module0_0_conv2d_9_out_channels=512,
                                   module0_0_reshape_10_shape=[1, 1, 2, 256],
                                   module0_0_reshape_14_shape=[1, 512],
                                   module0_0_reshape_15_shape=[1, 512, 1, 1],
                                   module0_0_reshape_16_shape=[1, 2, 256, 1, 1],
                                   module10_0_avgpool2d_0_kernel_size=(3, 3),
                                   module10_0_conv2d_1_in_channels=256,
                                   module10_0_conv2d_1_out_channels=1024)
        self.module10_1 = Module10(avgpool2d_0_kernel_size=(2, 2), conv2d_1_in_channels=512, conv2d_1_out_channels=1024)
        self.relu_191 = nn.ReLU()
        self.module44_0 = Module44(module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_2_in_channels=256,
                                   module0_0_conv2d_2_out_channels=512,
                                   module0_0_reshape_4_shape=[1, 2, 256, 16, 16],
                                   module0_0_conv2d_7_in_channels=256,
                                   module0_0_conv2d_7_out_channels=128,
                                   module0_0_conv2d_9_in_channels=128,
                                   module0_0_conv2d_9_out_channels=512,
                                   module0_0_reshape_10_shape=[1, 1, 2, 256],
                                   module0_0_reshape_14_shape=[1, 512],
                                   module0_0_reshape_15_shape=[1, 512, 1, 1],
                                   module0_0_reshape_16_shape=[1, 2, 256, 1, 1],
                                   module0_1_conv2d_0_in_channels=1024,
                                   module0_1_conv2d_0_out_channels=256,
                                   module0_1_conv2d_2_in_channels=256,
                                   module0_1_conv2d_2_out_channels=512,
                                   module0_1_reshape_4_shape=[1, 2, 256, 16, 16],
                                   module0_1_conv2d_7_in_channels=256,
                                   module0_1_conv2d_7_out_channels=128,
                                   module0_1_conv2d_9_in_channels=128,
                                   module0_1_conv2d_9_out_channels=512,
                                   module0_1_reshape_10_shape=[1, 1, 2, 256],
                                   module0_1_reshape_14_shape=[1, 512],
                                   module0_1_reshape_15_shape=[1, 512, 1, 1],
                                   module0_1_reshape_16_shape=[1, 2, 256, 1, 1],
                                   module0_2_conv2d_0_in_channels=1024,
                                   module0_2_conv2d_0_out_channels=256,
                                   module0_2_conv2d_2_in_channels=256,
                                   module0_2_conv2d_2_out_channels=512,
                                   module0_2_reshape_4_shape=[1, 2, 256, 16, 16],
                                   module0_2_conv2d_7_in_channels=256,
                                   module0_2_conv2d_7_out_channels=128,
                                   module0_2_conv2d_9_in_channels=128,
                                   module0_2_conv2d_9_out_channels=512,
                                   module0_2_reshape_10_shape=[1, 1, 2, 256],
                                   module0_2_reshape_14_shape=[1, 512],
                                   module0_2_reshape_15_shape=[1, 512, 1, 1],
                                   module0_2_reshape_16_shape=[1, 2, 256, 1, 1],
                                   module0_3_conv2d_0_in_channels=1024,
                                   module0_3_conv2d_0_out_channels=256,
                                   module0_3_conv2d_2_in_channels=256,
                                   module0_3_conv2d_2_out_channels=512,
                                   module0_3_reshape_4_shape=[1, 2, 256, 16, 16],
                                   module0_3_conv2d_7_in_channels=256,
                                   module0_3_conv2d_7_out_channels=128,
                                   module0_3_conv2d_9_in_channels=128,
                                   module0_3_conv2d_9_out_channels=512,
                                   module0_3_reshape_10_shape=[1, 1, 2, 256],
                                   module0_3_reshape_14_shape=[1, 512],
                                   module0_3_reshape_15_shape=[1, 512, 1, 1],
                                   module0_3_reshape_16_shape=[1, 2, 256, 1, 1],
                                   module0_4_conv2d_0_in_channels=1024,
                                   module0_4_conv2d_0_out_channels=256,
                                   module0_4_conv2d_2_in_channels=256,
                                   module0_4_conv2d_2_out_channels=512,
                                   module0_4_reshape_4_shape=[1, 2, 256, 16, 16],
                                   module0_4_conv2d_7_in_channels=256,
                                   module0_4_conv2d_7_out_channels=128,
                                   module0_4_conv2d_9_in_channels=128,
                                   module0_4_conv2d_9_out_channels=512,
                                   module0_4_reshape_10_shape=[1, 1, 2, 256],
                                   module0_4_reshape_14_shape=[1, 512],
                                   module0_4_reshape_15_shape=[1, 512, 1, 1],
                                   module0_4_reshape_16_shape=[1, 2, 256, 1, 1],
                                   module0_5_conv2d_0_in_channels=1024,
                                   module0_5_conv2d_0_out_channels=256,
                                   module0_5_conv2d_2_in_channels=256,
                                   module0_5_conv2d_2_out_channels=512,
                                   module0_5_reshape_4_shape=[1, 2, 256, 16, 16],
                                   module0_5_conv2d_7_in_channels=256,
                                   module0_5_conv2d_7_out_channels=128,
                                   module0_5_conv2d_9_in_channels=128,
                                   module0_5_conv2d_9_out_channels=512,
                                   module0_5_reshape_10_shape=[1, 1, 2, 256],
                                   module0_5_reshape_14_shape=[1, 512],
                                   module0_5_reshape_15_shape=[1, 512, 1, 1],
                                   module0_5_reshape_16_shape=[1, 2, 256, 1, 1],
                                   module0_6_conv2d_0_in_channels=1024,
                                   module0_6_conv2d_0_out_channels=256,
                                   module0_6_conv2d_2_in_channels=256,
                                   module0_6_conv2d_2_out_channels=512,
                                   module0_6_reshape_4_shape=[1, 2, 256, 16, 16],
                                   module0_6_conv2d_7_in_channels=256,
                                   module0_6_conv2d_7_out_channels=128,
                                   module0_6_conv2d_9_in_channels=128,
                                   module0_6_conv2d_9_out_channels=512,
                                   module0_6_reshape_10_shape=[1, 1, 2, 256],
                                   module0_6_reshape_14_shape=[1, 512],
                                   module0_6_reshape_15_shape=[1, 512, 1, 1],
                                   module0_6_reshape_16_shape=[1, 2, 256, 1, 1],
                                   module0_7_conv2d_0_in_channels=1024,
                                   module0_7_conv2d_0_out_channels=256,
                                   module0_7_conv2d_2_in_channels=256,
                                   module0_7_conv2d_2_out_channels=512,
                                   module0_7_reshape_4_shape=[1, 2, 256, 16, 16],
                                   module0_7_conv2d_7_in_channels=256,
                                   module0_7_conv2d_7_out_channels=128,
                                   module0_7_conv2d_9_in_channels=128,
                                   module0_7_conv2d_9_out_channels=512,
                                   module0_7_reshape_10_shape=[1, 1, 2, 256],
                                   module0_7_reshape_14_shape=[1, 512],
                                   module0_7_reshape_15_shape=[1, 512, 1, 1],
                                   module0_7_reshape_16_shape=[1, 2, 256, 1, 1])
        self.module44_1 = Module44(module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_2_in_channels=256,
                                   module0_0_conv2d_2_out_channels=512,
                                   module0_0_reshape_4_shape=[1, 2, 256, 16, 16],
                                   module0_0_conv2d_7_in_channels=256,
                                   module0_0_conv2d_7_out_channels=128,
                                   module0_0_conv2d_9_in_channels=128,
                                   module0_0_conv2d_9_out_channels=512,
                                   module0_0_reshape_10_shape=[1, 1, 2, 256],
                                   module0_0_reshape_14_shape=[1, 512],
                                   module0_0_reshape_15_shape=[1, 512, 1, 1],
                                   module0_0_reshape_16_shape=[1, 2, 256, 1, 1],
                                   module0_1_conv2d_0_in_channels=1024,
                                   module0_1_conv2d_0_out_channels=256,
                                   module0_1_conv2d_2_in_channels=256,
                                   module0_1_conv2d_2_out_channels=512,
                                   module0_1_reshape_4_shape=[1, 2, 256, 16, 16],
                                   module0_1_conv2d_7_in_channels=256,
                                   module0_1_conv2d_7_out_channels=128,
                                   module0_1_conv2d_9_in_channels=128,
                                   module0_1_conv2d_9_out_channels=512,
                                   module0_1_reshape_10_shape=[1, 1, 2, 256],
                                   module0_1_reshape_14_shape=[1, 512],
                                   module0_1_reshape_15_shape=[1, 512, 1, 1],
                                   module0_1_reshape_16_shape=[1, 2, 256, 1, 1],
                                   module0_2_conv2d_0_in_channels=1024,
                                   module0_2_conv2d_0_out_channels=256,
                                   module0_2_conv2d_2_in_channels=256,
                                   module0_2_conv2d_2_out_channels=512,
                                   module0_2_reshape_4_shape=[1, 2, 256, 16, 16],
                                   module0_2_conv2d_7_in_channels=256,
                                   module0_2_conv2d_7_out_channels=128,
                                   module0_2_conv2d_9_in_channels=128,
                                   module0_2_conv2d_9_out_channels=512,
                                   module0_2_reshape_10_shape=[1, 1, 2, 256],
                                   module0_2_reshape_14_shape=[1, 512],
                                   module0_2_reshape_15_shape=[1, 512, 1, 1],
                                   module0_2_reshape_16_shape=[1, 2, 256, 1, 1],
                                   module0_3_conv2d_0_in_channels=1024,
                                   module0_3_conv2d_0_out_channels=256,
                                   module0_3_conv2d_2_in_channels=256,
                                   module0_3_conv2d_2_out_channels=512,
                                   module0_3_reshape_4_shape=[1, 2, 256, 16, 16],
                                   module0_3_conv2d_7_in_channels=256,
                                   module0_3_conv2d_7_out_channels=128,
                                   module0_3_conv2d_9_in_channels=128,
                                   module0_3_conv2d_9_out_channels=512,
                                   module0_3_reshape_10_shape=[1, 1, 2, 256],
                                   module0_3_reshape_14_shape=[1, 512],
                                   module0_3_reshape_15_shape=[1, 512, 1, 1],
                                   module0_3_reshape_16_shape=[1, 2, 256, 1, 1],
                                   module0_4_conv2d_0_in_channels=1024,
                                   module0_4_conv2d_0_out_channels=256,
                                   module0_4_conv2d_2_in_channels=256,
                                   module0_4_conv2d_2_out_channels=512,
                                   module0_4_reshape_4_shape=[1, 2, 256, 16, 16],
                                   module0_4_conv2d_7_in_channels=256,
                                   module0_4_conv2d_7_out_channels=128,
                                   module0_4_conv2d_9_in_channels=128,
                                   module0_4_conv2d_9_out_channels=512,
                                   module0_4_reshape_10_shape=[1, 1, 2, 256],
                                   module0_4_reshape_14_shape=[1, 512],
                                   module0_4_reshape_15_shape=[1, 512, 1, 1],
                                   module0_4_reshape_16_shape=[1, 2, 256, 1, 1],
                                   module0_5_conv2d_0_in_channels=1024,
                                   module0_5_conv2d_0_out_channels=256,
                                   module0_5_conv2d_2_in_channels=256,
                                   module0_5_conv2d_2_out_channels=512,
                                   module0_5_reshape_4_shape=[1, 2, 256, 16, 16],
                                   module0_5_conv2d_7_in_channels=256,
                                   module0_5_conv2d_7_out_channels=128,
                                   module0_5_conv2d_9_in_channels=128,
                                   module0_5_conv2d_9_out_channels=512,
                                   module0_5_reshape_10_shape=[1, 1, 2, 256],
                                   module0_5_reshape_14_shape=[1, 512],
                                   module0_5_reshape_15_shape=[1, 512, 1, 1],
                                   module0_5_reshape_16_shape=[1, 2, 256, 1, 1],
                                   module0_6_conv2d_0_in_channels=1024,
                                   module0_6_conv2d_0_out_channels=256,
                                   module0_6_conv2d_2_in_channels=256,
                                   module0_6_conv2d_2_out_channels=512,
                                   module0_6_reshape_4_shape=[1, 2, 256, 16, 16],
                                   module0_6_conv2d_7_in_channels=256,
                                   module0_6_conv2d_7_out_channels=128,
                                   module0_6_conv2d_9_in_channels=128,
                                   module0_6_conv2d_9_out_channels=512,
                                   module0_6_reshape_10_shape=[1, 1, 2, 256],
                                   module0_6_reshape_14_shape=[1, 512],
                                   module0_6_reshape_15_shape=[1, 512, 1, 1],
                                   module0_6_reshape_16_shape=[1, 2, 256, 1, 1],
                                   module0_7_conv2d_0_in_channels=1024,
                                   module0_7_conv2d_0_out_channels=256,
                                   module0_7_conv2d_2_in_channels=256,
                                   module0_7_conv2d_2_out_channels=512,
                                   module0_7_reshape_4_shape=[1, 2, 256, 16, 16],
                                   module0_7_conv2d_7_in_channels=256,
                                   module0_7_conv2d_7_out_channels=128,
                                   module0_7_conv2d_9_in_channels=128,
                                   module0_7_conv2d_9_out_channels=512,
                                   module0_7_reshape_10_shape=[1, 1, 2, 256],
                                   module0_7_reshape_14_shape=[1, 512],
                                   module0_7_reshape_15_shape=[1, 512, 1, 1],
                                   module0_7_reshape_16_shape=[1, 2, 256, 1, 1])
        self.module43_0 = Module43()
        self.module16_2 = Module16(conv2d_1_in_channels=256,
                                   conv2d_1_out_channels=1024,
                                   conv2d_5_in_channels=256,
                                   conv2d_5_out_channels=1024,
                                   module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_2_in_channels=256,
                                   module0_0_conv2d_2_out_channels=512,
                                   module0_0_reshape_4_shape=[1, 2, 256, 16, 16],
                                   module0_0_conv2d_7_in_channels=256,
                                   module0_0_conv2d_7_out_channels=128,
                                   module0_0_conv2d_9_in_channels=128,
                                   module0_0_conv2d_9_out_channels=512,
                                   module0_0_reshape_10_shape=[1, 1, 2, 256],
                                   module0_0_reshape_14_shape=[1, 512],
                                   module0_0_reshape_15_shape=[1, 512, 1, 1],
                                   module0_0_reshape_16_shape=[1, 2, 256, 1, 1],
                                   module0_1_conv2d_0_in_channels=1024,
                                   module0_1_conv2d_0_out_channels=256,
                                   module0_1_conv2d_2_in_channels=256,
                                   module0_1_conv2d_2_out_channels=512,
                                   module0_1_reshape_4_shape=[1, 2, 256, 16, 16],
                                   module0_1_conv2d_7_in_channels=256,
                                   module0_1_conv2d_7_out_channels=128,
                                   module0_1_conv2d_9_in_channels=128,
                                   module0_1_conv2d_9_out_channels=512,
                                   module0_1_reshape_10_shape=[1, 1, 2, 256],
                                   module0_1_reshape_14_shape=[1, 512],
                                   module0_1_reshape_15_shape=[1, 512, 1, 1],
                                   module0_1_reshape_16_shape=[1, 2, 256, 1, 1])
        self.module15_2 = Module15(module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=512,
                                   module0_0_conv2d_2_in_channels=512,
                                   module0_0_conv2d_2_out_channels=1024,
                                   module0_0_reshape_4_shape=[1, 2, 512, 16, 16],
                                   module0_0_conv2d_7_in_channels=512,
                                   module0_0_conv2d_7_out_channels=256,
                                   module0_0_conv2d_9_in_channels=256,
                                   module0_0_conv2d_9_out_channels=1024,
                                   module0_0_reshape_10_shape=[1, 1, 2, 512],
                                   module0_0_reshape_14_shape=[1, 1024],
                                   module0_0_reshape_15_shape=[1, 1024, 1, 1],
                                   module0_0_reshape_16_shape=[1, 2, 512, 1, 1],
                                   module10_0_avgpool2d_0_kernel_size=(3, 3),
                                   module10_0_conv2d_1_in_channels=512,
                                   module10_0_conv2d_1_out_channels=2048)
        self.module10_2 = Module10(avgpool2d_0_kernel_size=(2, 2),
                                   conv2d_1_in_channels=1024,
                                   conv2d_1_out_channels=2048)
        self.relu_701 = nn.ReLU()
        self.module16_3 = Module16(conv2d_1_in_channels=512,
                                   conv2d_1_out_channels=2048,
                                   conv2d_5_in_channels=512,
                                   conv2d_5_out_channels=2048,
                                   module0_0_conv2d_0_in_channels=2048,
                                   module0_0_conv2d_0_out_channels=512,
                                   module0_0_conv2d_2_in_channels=512,
                                   module0_0_conv2d_2_out_channels=1024,
                                   module0_0_reshape_4_shape=[1, 2, 512, 8, 8],
                                   module0_0_conv2d_7_in_channels=512,
                                   module0_0_conv2d_7_out_channels=256,
                                   module0_0_conv2d_9_in_channels=256,
                                   module0_0_conv2d_9_out_channels=1024,
                                   module0_0_reshape_10_shape=[1, 1, 2, 512],
                                   module0_0_reshape_14_shape=[1, 1024],
                                   module0_0_reshape_15_shape=[1, 1024, 1, 1],
                                   module0_0_reshape_16_shape=[1, 2, 512, 1, 1],
                                   module0_1_conv2d_0_in_channels=2048,
                                   module0_1_conv2d_0_out_channels=512,
                                   module0_1_conv2d_2_in_channels=512,
                                   module0_1_conv2d_2_out_channels=1024,
                                   module0_1_reshape_4_shape=[1, 2, 512, 8, 8],
                                   module0_1_conv2d_7_in_channels=512,
                                   module0_1_conv2d_7_out_channels=256,
                                   module0_1_conv2d_9_in_channels=256,
                                   module0_1_conv2d_9_out_channels=1024,
                                   module0_1_reshape_10_shape=[1, 1, 2, 512],
                                   module0_1_reshape_14_shape=[1, 1024],
                                   module0_1_reshape_15_shape=[1, 1024, 1, 1],
                                   module0_1_reshape_16_shape=[1, 2, 512, 1, 1])
        self.avgpool2d_746 = nn.AvgPool2d(kernel_size=(8, 8))
        self.flatten_747 = nn.Flatten()
        self.dense_748 = nn.Dense(in_channels=2048, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_relu_3)
        opt_relu_5 = self.relu_5(opt_conv2d_4)
        opt_maxpool2d_6 = self.pad_maxpool2d_6(opt_relu_5)
        opt_maxpool2d_6 = self.maxpool2d_6(opt_maxpool2d_6)
        module4_0_opt = self.module4_0(opt_maxpool2d_6)
        opt_conv2d_8 = self.conv2d_8(opt_maxpool2d_6)
        opt_add_28 = P.Add()(module4_0_opt, opt_conv2d_8)
        opt_relu_29 = self.relu_29(opt_add_28)
        module16_0_opt = self.module16_0(opt_relu_29)
        module15_0_opt = self.module15_0(module16_0_opt)
        module10_0_opt = self.module10_0(module16_0_opt)
        opt_add_98 = P.Add()(module15_0_opt, module10_0_opt)
        opt_relu_99 = self.relu_99(opt_add_98)
        module16_1_opt = self.module16_1(opt_relu_99)
        module6_0_opt = self.module6_0(module16_1_opt)
        module15_1_opt = self.module15_1(module6_0_opt)
        module10_1_opt = self.module10_1(module6_0_opt)
        opt_add_190 = P.Add()(module15_1_opt, module10_1_opt)
        opt_relu_191 = self.relu_191(opt_add_190)
        module44_0_opt = self.module44_0(opt_relu_191)
        module44_1_opt = self.module44_1(module44_0_opt)
        module43_0_opt = self.module43_0(module44_1_opt)
        module16_2_opt = self.module16_2(module43_0_opt)
        module15_2_opt = self.module15_2(module16_2_opt)
        module10_2_opt = self.module10_2(module16_2_opt)
        opt_add_700 = P.Add()(module15_2_opt, module10_2_opt)
        opt_relu_701 = self.relu_701(opt_add_700)
        module16_3_opt = self.module16_3(opt_relu_701)
        opt_avgpool2d_746 = self.avgpool2d_746(module16_3_opt)
        opt_flatten_747 = self.flatten_747(opt_avgpool2d_746)
        opt_dense_748 = self.dense_748(opt_flatten_747)
        return opt_dense_748
