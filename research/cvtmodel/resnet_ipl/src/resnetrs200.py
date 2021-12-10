import mindspore.ops as P
from mindspore import nn


class Module11(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_stride):
        super(Module11, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(3, 3),
                                  stride=conv2d_0_stride,
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        return opt_relu_1


class Module0(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_2_in_channels, conv2d_2_out_channels,
                 conv2d_2_stride, conv2d_4_in_channels, conv2d_4_out_channels, conv2d_6_in_channels,
                 conv2d_6_out_channels, conv2d_8_in_channels, conv2d_8_out_channels):
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
                                  stride=conv2d_2_stride,
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()
        self.conv2d_4 = nn.Conv2d(in_channels=conv2d_4_in_channels,
                                  out_channels=conv2d_4_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.reducemean_5 = P.ReduceMean(keep_dims=True)
        self.reducemean_5_axis = (2, 3)
        self.conv2d_6 = nn.Conv2d(in_channels=conv2d_6_in_channels,
                                  out_channels=conv2d_6_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_7 = nn.ReLU()
        self.conv2d_8 = nn.Conv2d(in_channels=conv2d_8_in_channels,
                                  out_channels=conv2d_8_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.sigmoid_9 = nn.Sigmoid()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_relu_3)
        opt_reducemean_5 = self.reducemean_5(opt_conv2d_4, self.reducemean_5_axis)
        opt_conv2d_6 = self.conv2d_6(opt_reducemean_5)
        opt_relu_7 = self.relu_7(opt_conv2d_6)
        opt_conv2d_8 = self.conv2d_8(opt_relu_7)
        opt_sigmoid_9 = self.sigmoid_9(opt_conv2d_8)
        opt_mul_10 = P.Mul()(opt_conv2d_4, opt_sigmoid_9)
        return opt_mul_10


class Module2(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_2_stride, module0_0_conv2d_4_in_channels,
                 module0_0_conv2d_4_out_channels, module0_0_conv2d_6_in_channels, module0_0_conv2d_6_out_channels,
                 module0_0_conv2d_8_in_channels, module0_0_conv2d_8_out_channels, module0_1_conv2d_0_in_channels,
                 module0_1_conv2d_0_out_channels, module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels,
                 module0_1_conv2d_2_stride, module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels,
                 module0_1_conv2d_6_in_channels, module0_1_conv2d_6_out_channels, module0_1_conv2d_8_in_channels,
                 module0_1_conv2d_8_out_channels):
        super(Module2, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_0_conv2d_2_stride,
                                 conv2d_4_in_channels=module0_0_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_0_conv2d_4_out_channels,
                                 conv2d_6_in_channels=module0_0_conv2d_6_in_channels,
                                 conv2d_6_out_channels=module0_0_conv2d_6_out_channels,
                                 conv2d_8_in_channels=module0_0_conv2d_8_in_channels,
                                 conv2d_8_out_channels=module0_0_conv2d_8_out_channels)
        self.relu_1 = nn.ReLU()
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_1_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_1_conv2d_2_stride,
                                 conv2d_4_in_channels=module0_1_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_1_conv2d_4_out_channels,
                                 conv2d_6_in_channels=module0_1_conv2d_6_in_channels,
                                 conv2d_6_out_channels=module0_1_conv2d_6_out_channels,
                                 conv2d_8_in_channels=module0_1_conv2d_8_in_channels,
                                 conv2d_8_out_channels=module0_1_conv2d_8_out_channels)
        self.relu_3 = nn.ReLU()

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_add_0 = P.Add()(module0_0_opt, x)
        opt_relu_1 = self.relu_1(opt_add_0)
        module0_1_opt = self.module0_1(opt_relu_1)
        opt_add_2 = P.Add()(module0_1_opt, opt_relu_1)
        opt_relu_3 = self.relu_3(opt_add_2)
        return opt_relu_3


class Module6(nn.Cell):
    def __init__(self, conv2d_1_in_channels, conv2d_1_out_channels):
        super(Module6, self).__init__()
        self.pad_avgpool2d_0 = nn.Pad(paddings=((0, 0), (0, 0), (0, 0), (0, 0)))
        self.avgpool2d_0 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
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


class Module40(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_2_stride, module0_0_conv2d_4_in_channels,
                 module0_0_conv2d_4_out_channels, module0_0_conv2d_6_in_channels, module0_0_conv2d_6_out_channels,
                 module0_0_conv2d_8_in_channels, module0_0_conv2d_8_out_channels, module0_1_conv2d_0_in_channels,
                 module0_1_conv2d_0_out_channels, module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels,
                 module0_1_conv2d_2_stride, module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels,
                 module0_1_conv2d_6_in_channels, module0_1_conv2d_6_out_channels, module0_1_conv2d_8_in_channels,
                 module0_1_conv2d_8_out_channels, module0_2_conv2d_0_in_channels, module0_2_conv2d_0_out_channels,
                 module0_2_conv2d_2_in_channels, module0_2_conv2d_2_out_channels, module0_2_conv2d_2_stride,
                 module0_2_conv2d_4_in_channels, module0_2_conv2d_4_out_channels, module0_2_conv2d_6_in_channels,
                 module0_2_conv2d_6_out_channels, module0_2_conv2d_8_in_channels, module0_2_conv2d_8_out_channels,
                 module0_3_conv2d_0_in_channels, module0_3_conv2d_0_out_channels, module0_3_conv2d_2_in_channels,
                 module0_3_conv2d_2_out_channels, module0_3_conv2d_2_stride, module0_3_conv2d_4_in_channels,
                 module0_3_conv2d_4_out_channels, module0_3_conv2d_6_in_channels, module0_3_conv2d_6_out_channels,
                 module0_3_conv2d_8_in_channels, module0_3_conv2d_8_out_channels, module0_4_conv2d_0_in_channels,
                 module0_4_conv2d_0_out_channels, module0_4_conv2d_2_in_channels, module0_4_conv2d_2_out_channels,
                 module0_4_conv2d_2_stride, module0_4_conv2d_4_in_channels, module0_4_conv2d_4_out_channels,
                 module0_4_conv2d_6_in_channels, module0_4_conv2d_6_out_channels, module0_4_conv2d_8_in_channels,
                 module0_4_conv2d_8_out_channels, module0_5_conv2d_0_in_channels, module0_5_conv2d_0_out_channels,
                 module0_5_conv2d_2_in_channels, module0_5_conv2d_2_out_channels, module0_5_conv2d_2_stride,
                 module0_5_conv2d_4_in_channels, module0_5_conv2d_4_out_channels, module0_5_conv2d_6_in_channels,
                 module0_5_conv2d_6_out_channels, module0_5_conv2d_8_in_channels, module0_5_conv2d_8_out_channels,
                 module0_6_conv2d_0_in_channels, module0_6_conv2d_0_out_channels, module0_6_conv2d_2_in_channels,
                 module0_6_conv2d_2_out_channels, module0_6_conv2d_2_stride, module0_6_conv2d_4_in_channels,
                 module0_6_conv2d_4_out_channels, module0_6_conv2d_6_in_channels, module0_6_conv2d_6_out_channels,
                 module0_6_conv2d_8_in_channels, module0_6_conv2d_8_out_channels, module0_7_conv2d_0_in_channels,
                 module0_7_conv2d_0_out_channels, module0_7_conv2d_2_in_channels, module0_7_conv2d_2_out_channels,
                 module0_7_conv2d_2_stride, module0_7_conv2d_4_in_channels, module0_7_conv2d_4_out_channels,
                 module0_7_conv2d_6_in_channels, module0_7_conv2d_6_out_channels, module0_7_conv2d_8_in_channels,
                 module0_7_conv2d_8_out_channels, module0_8_conv2d_0_in_channels, module0_8_conv2d_0_out_channels,
                 module0_8_conv2d_2_in_channels, module0_8_conv2d_2_out_channels, module0_8_conv2d_2_stride,
                 module0_8_conv2d_4_in_channels, module0_8_conv2d_4_out_channels, module0_8_conv2d_6_in_channels,
                 module0_8_conv2d_6_out_channels, module0_8_conv2d_8_in_channels, module0_8_conv2d_8_out_channels,
                 module0_9_conv2d_0_in_channels, module0_9_conv2d_0_out_channels, module0_9_conv2d_2_in_channels,
                 module0_9_conv2d_2_out_channels, module0_9_conv2d_2_stride, module0_9_conv2d_4_in_channels,
                 module0_9_conv2d_4_out_channels, module0_9_conv2d_6_in_channels, module0_9_conv2d_6_out_channels,
                 module0_9_conv2d_8_in_channels, module0_9_conv2d_8_out_channels, module0_10_conv2d_0_in_channels,
                 module0_10_conv2d_0_out_channels, module0_10_conv2d_2_in_channels, module0_10_conv2d_2_out_channels,
                 module0_10_conv2d_2_stride, module0_10_conv2d_4_in_channels, module0_10_conv2d_4_out_channels,
                 module0_10_conv2d_6_in_channels, module0_10_conv2d_6_out_channels, module0_10_conv2d_8_in_channels,
                 module0_10_conv2d_8_out_channels, module0_11_conv2d_0_in_channels, module0_11_conv2d_0_out_channels,
                 module0_11_conv2d_2_in_channels, module0_11_conv2d_2_out_channels, module0_11_conv2d_2_stride,
                 module0_11_conv2d_4_in_channels, module0_11_conv2d_4_out_channels, module0_11_conv2d_6_in_channels,
                 module0_11_conv2d_6_out_channels, module0_11_conv2d_8_in_channels, module0_11_conv2d_8_out_channels,
                 module0_12_conv2d_0_in_channels, module0_12_conv2d_0_out_channels, module0_12_conv2d_2_in_channels,
                 module0_12_conv2d_2_out_channels, module0_12_conv2d_2_stride, module0_12_conv2d_4_in_channels,
                 module0_12_conv2d_4_out_channels, module0_12_conv2d_6_in_channels, module0_12_conv2d_6_out_channels,
                 module0_12_conv2d_8_in_channels, module0_12_conv2d_8_out_channels, module0_13_conv2d_0_in_channels,
                 module0_13_conv2d_0_out_channels, module0_13_conv2d_2_in_channels, module0_13_conv2d_2_out_channels,
                 module0_13_conv2d_2_stride, module0_13_conv2d_4_in_channels, module0_13_conv2d_4_out_channels,
                 module0_13_conv2d_6_in_channels, module0_13_conv2d_6_out_channels, module0_13_conv2d_8_in_channels,
                 module0_13_conv2d_8_out_channels, module0_14_conv2d_0_in_channels, module0_14_conv2d_0_out_channels,
                 module0_14_conv2d_2_in_channels, module0_14_conv2d_2_out_channels, module0_14_conv2d_2_stride,
                 module0_14_conv2d_4_in_channels, module0_14_conv2d_4_out_channels, module0_14_conv2d_6_in_channels,
                 module0_14_conv2d_6_out_channels, module0_14_conv2d_8_in_channels, module0_14_conv2d_8_out_channels,
                 module0_15_conv2d_0_in_channels, module0_15_conv2d_0_out_channels, module0_15_conv2d_2_in_channels,
                 module0_15_conv2d_2_out_channels, module0_15_conv2d_2_stride, module0_15_conv2d_4_in_channels,
                 module0_15_conv2d_4_out_channels, module0_15_conv2d_6_in_channels, module0_15_conv2d_6_out_channels,
                 module0_15_conv2d_8_in_channels, module0_15_conv2d_8_out_channels):
        super(Module40, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_0_conv2d_2_stride,
                                 conv2d_4_in_channels=module0_0_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_0_conv2d_4_out_channels,
                                 conv2d_6_in_channels=module0_0_conv2d_6_in_channels,
                                 conv2d_6_out_channels=module0_0_conv2d_6_out_channels,
                                 conv2d_8_in_channels=module0_0_conv2d_8_in_channels,
                                 conv2d_8_out_channels=module0_0_conv2d_8_out_channels)
        self.relu_1 = nn.ReLU()
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_1_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_1_conv2d_2_stride,
                                 conv2d_4_in_channels=module0_1_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_1_conv2d_4_out_channels,
                                 conv2d_6_in_channels=module0_1_conv2d_6_in_channels,
                                 conv2d_6_out_channels=module0_1_conv2d_6_out_channels,
                                 conv2d_8_in_channels=module0_1_conv2d_8_in_channels,
                                 conv2d_8_out_channels=module0_1_conv2d_8_out_channels)
        self.relu_3 = nn.ReLU()
        self.module0_2 = Module0(conv2d_0_in_channels=module0_2_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_2_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_2_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_2_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_2_conv2d_2_stride,
                                 conv2d_4_in_channels=module0_2_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_2_conv2d_4_out_channels,
                                 conv2d_6_in_channels=module0_2_conv2d_6_in_channels,
                                 conv2d_6_out_channels=module0_2_conv2d_6_out_channels,
                                 conv2d_8_in_channels=module0_2_conv2d_8_in_channels,
                                 conv2d_8_out_channels=module0_2_conv2d_8_out_channels)
        self.relu_5 = nn.ReLU()
        self.module0_3 = Module0(conv2d_0_in_channels=module0_3_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_3_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_3_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_3_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_3_conv2d_2_stride,
                                 conv2d_4_in_channels=module0_3_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_3_conv2d_4_out_channels,
                                 conv2d_6_in_channels=module0_3_conv2d_6_in_channels,
                                 conv2d_6_out_channels=module0_3_conv2d_6_out_channels,
                                 conv2d_8_in_channels=module0_3_conv2d_8_in_channels,
                                 conv2d_8_out_channels=module0_3_conv2d_8_out_channels)
        self.relu_7 = nn.ReLU()
        self.module0_4 = Module0(conv2d_0_in_channels=module0_4_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_4_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_4_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_4_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_4_conv2d_2_stride,
                                 conv2d_4_in_channels=module0_4_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_4_conv2d_4_out_channels,
                                 conv2d_6_in_channels=module0_4_conv2d_6_in_channels,
                                 conv2d_6_out_channels=module0_4_conv2d_6_out_channels,
                                 conv2d_8_in_channels=module0_4_conv2d_8_in_channels,
                                 conv2d_8_out_channels=module0_4_conv2d_8_out_channels)
        self.relu_9 = nn.ReLU()
        self.module0_5 = Module0(conv2d_0_in_channels=module0_5_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_5_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_5_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_5_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_5_conv2d_2_stride,
                                 conv2d_4_in_channels=module0_5_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_5_conv2d_4_out_channels,
                                 conv2d_6_in_channels=module0_5_conv2d_6_in_channels,
                                 conv2d_6_out_channels=module0_5_conv2d_6_out_channels,
                                 conv2d_8_in_channels=module0_5_conv2d_8_in_channels,
                                 conv2d_8_out_channels=module0_5_conv2d_8_out_channels)
        self.relu_11 = nn.ReLU()
        self.module0_6 = Module0(conv2d_0_in_channels=module0_6_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_6_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_6_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_6_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_6_conv2d_2_stride,
                                 conv2d_4_in_channels=module0_6_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_6_conv2d_4_out_channels,
                                 conv2d_6_in_channels=module0_6_conv2d_6_in_channels,
                                 conv2d_6_out_channels=module0_6_conv2d_6_out_channels,
                                 conv2d_8_in_channels=module0_6_conv2d_8_in_channels,
                                 conv2d_8_out_channels=module0_6_conv2d_8_out_channels)
        self.relu_13 = nn.ReLU()
        self.module0_7 = Module0(conv2d_0_in_channels=module0_7_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_7_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_7_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_7_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_7_conv2d_2_stride,
                                 conv2d_4_in_channels=module0_7_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_7_conv2d_4_out_channels,
                                 conv2d_6_in_channels=module0_7_conv2d_6_in_channels,
                                 conv2d_6_out_channels=module0_7_conv2d_6_out_channels,
                                 conv2d_8_in_channels=module0_7_conv2d_8_in_channels,
                                 conv2d_8_out_channels=module0_7_conv2d_8_out_channels)
        self.relu_15 = nn.ReLU()
        self.module0_8 = Module0(conv2d_0_in_channels=module0_8_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_8_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_8_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_8_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_8_conv2d_2_stride,
                                 conv2d_4_in_channels=module0_8_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_8_conv2d_4_out_channels,
                                 conv2d_6_in_channels=module0_8_conv2d_6_in_channels,
                                 conv2d_6_out_channels=module0_8_conv2d_6_out_channels,
                                 conv2d_8_in_channels=module0_8_conv2d_8_in_channels,
                                 conv2d_8_out_channels=module0_8_conv2d_8_out_channels)
        self.relu_17 = nn.ReLU()
        self.module0_9 = Module0(conv2d_0_in_channels=module0_9_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_9_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_9_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_9_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_9_conv2d_2_stride,
                                 conv2d_4_in_channels=module0_9_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_9_conv2d_4_out_channels,
                                 conv2d_6_in_channels=module0_9_conv2d_6_in_channels,
                                 conv2d_6_out_channels=module0_9_conv2d_6_out_channels,
                                 conv2d_8_in_channels=module0_9_conv2d_8_in_channels,
                                 conv2d_8_out_channels=module0_9_conv2d_8_out_channels)
        self.relu_19 = nn.ReLU()
        self.module0_10 = Module0(conv2d_0_in_channels=module0_10_conv2d_0_in_channels,
                                  conv2d_0_out_channels=module0_10_conv2d_0_out_channels,
                                  conv2d_2_in_channels=module0_10_conv2d_2_in_channels,
                                  conv2d_2_out_channels=module0_10_conv2d_2_out_channels,
                                  conv2d_2_stride=module0_10_conv2d_2_stride,
                                  conv2d_4_in_channels=module0_10_conv2d_4_in_channels,
                                  conv2d_4_out_channels=module0_10_conv2d_4_out_channels,
                                  conv2d_6_in_channels=module0_10_conv2d_6_in_channels,
                                  conv2d_6_out_channels=module0_10_conv2d_6_out_channels,
                                  conv2d_8_in_channels=module0_10_conv2d_8_in_channels,
                                  conv2d_8_out_channels=module0_10_conv2d_8_out_channels)
        self.relu_21 = nn.ReLU()
        self.module0_11 = Module0(conv2d_0_in_channels=module0_11_conv2d_0_in_channels,
                                  conv2d_0_out_channels=module0_11_conv2d_0_out_channels,
                                  conv2d_2_in_channels=module0_11_conv2d_2_in_channels,
                                  conv2d_2_out_channels=module0_11_conv2d_2_out_channels,
                                  conv2d_2_stride=module0_11_conv2d_2_stride,
                                  conv2d_4_in_channels=module0_11_conv2d_4_in_channels,
                                  conv2d_4_out_channels=module0_11_conv2d_4_out_channels,
                                  conv2d_6_in_channels=module0_11_conv2d_6_in_channels,
                                  conv2d_6_out_channels=module0_11_conv2d_6_out_channels,
                                  conv2d_8_in_channels=module0_11_conv2d_8_in_channels,
                                  conv2d_8_out_channels=module0_11_conv2d_8_out_channels)
        self.relu_23 = nn.ReLU()
        self.module0_12 = Module0(conv2d_0_in_channels=module0_12_conv2d_0_in_channels,
                                  conv2d_0_out_channels=module0_12_conv2d_0_out_channels,
                                  conv2d_2_in_channels=module0_12_conv2d_2_in_channels,
                                  conv2d_2_out_channels=module0_12_conv2d_2_out_channels,
                                  conv2d_2_stride=module0_12_conv2d_2_stride,
                                  conv2d_4_in_channels=module0_12_conv2d_4_in_channels,
                                  conv2d_4_out_channels=module0_12_conv2d_4_out_channels,
                                  conv2d_6_in_channels=module0_12_conv2d_6_in_channels,
                                  conv2d_6_out_channels=module0_12_conv2d_6_out_channels,
                                  conv2d_8_in_channels=module0_12_conv2d_8_in_channels,
                                  conv2d_8_out_channels=module0_12_conv2d_8_out_channels)
        self.relu_25 = nn.ReLU()
        self.module0_13 = Module0(conv2d_0_in_channels=module0_13_conv2d_0_in_channels,
                                  conv2d_0_out_channels=module0_13_conv2d_0_out_channels,
                                  conv2d_2_in_channels=module0_13_conv2d_2_in_channels,
                                  conv2d_2_out_channels=module0_13_conv2d_2_out_channels,
                                  conv2d_2_stride=module0_13_conv2d_2_stride,
                                  conv2d_4_in_channels=module0_13_conv2d_4_in_channels,
                                  conv2d_4_out_channels=module0_13_conv2d_4_out_channels,
                                  conv2d_6_in_channels=module0_13_conv2d_6_in_channels,
                                  conv2d_6_out_channels=module0_13_conv2d_6_out_channels,
                                  conv2d_8_in_channels=module0_13_conv2d_8_in_channels,
                                  conv2d_8_out_channels=module0_13_conv2d_8_out_channels)
        self.relu_27 = nn.ReLU()
        self.module0_14 = Module0(conv2d_0_in_channels=module0_14_conv2d_0_in_channels,
                                  conv2d_0_out_channels=module0_14_conv2d_0_out_channels,
                                  conv2d_2_in_channels=module0_14_conv2d_2_in_channels,
                                  conv2d_2_out_channels=module0_14_conv2d_2_out_channels,
                                  conv2d_2_stride=module0_14_conv2d_2_stride,
                                  conv2d_4_in_channels=module0_14_conv2d_4_in_channels,
                                  conv2d_4_out_channels=module0_14_conv2d_4_out_channels,
                                  conv2d_6_in_channels=module0_14_conv2d_6_in_channels,
                                  conv2d_6_out_channels=module0_14_conv2d_6_out_channels,
                                  conv2d_8_in_channels=module0_14_conv2d_8_in_channels,
                                  conv2d_8_out_channels=module0_14_conv2d_8_out_channels)
        self.relu_29 = nn.ReLU()
        self.module0_15 = Module0(conv2d_0_in_channels=module0_15_conv2d_0_in_channels,
                                  conv2d_0_out_channels=module0_15_conv2d_0_out_channels,
                                  conv2d_2_in_channels=module0_15_conv2d_2_in_channels,
                                  conv2d_2_out_channels=module0_15_conv2d_2_out_channels,
                                  conv2d_2_stride=module0_15_conv2d_2_stride,
                                  conv2d_4_in_channels=module0_15_conv2d_4_in_channels,
                                  conv2d_4_out_channels=module0_15_conv2d_4_out_channels,
                                  conv2d_6_in_channels=module0_15_conv2d_6_in_channels,
                                  conv2d_6_out_channels=module0_15_conv2d_6_out_channels,
                                  conv2d_8_in_channels=module0_15_conv2d_8_in_channels,
                                  conv2d_8_out_channels=module0_15_conv2d_8_out_channels)
        self.relu_31 = nn.ReLU()

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_add_0 = P.Add()(module0_0_opt, x)
        opt_relu_1 = self.relu_1(opt_add_0)
        module0_1_opt = self.module0_1(opt_relu_1)
        opt_add_2 = P.Add()(module0_1_opt, opt_relu_1)
        opt_relu_3 = self.relu_3(opt_add_2)
        module0_2_opt = self.module0_2(opt_relu_3)
        opt_add_4 = P.Add()(module0_2_opt, opt_relu_3)
        opt_relu_5 = self.relu_5(opt_add_4)
        module0_3_opt = self.module0_3(opt_relu_5)
        opt_add_6 = P.Add()(module0_3_opt, opt_relu_5)
        opt_relu_7 = self.relu_7(opt_add_6)
        module0_4_opt = self.module0_4(opt_relu_7)
        opt_add_8 = P.Add()(module0_4_opt, opt_relu_7)
        opt_relu_9 = self.relu_9(opt_add_8)
        module0_5_opt = self.module0_5(opt_relu_9)
        opt_add_10 = P.Add()(module0_5_opt, opt_relu_9)
        opt_relu_11 = self.relu_11(opt_add_10)
        module0_6_opt = self.module0_6(opt_relu_11)
        opt_add_12 = P.Add()(module0_6_opt, opt_relu_11)
        opt_relu_13 = self.relu_13(opt_add_12)
        module0_7_opt = self.module0_7(opt_relu_13)
        opt_add_14 = P.Add()(module0_7_opt, opt_relu_13)
        opt_relu_15 = self.relu_15(opt_add_14)
        module0_8_opt = self.module0_8(opt_relu_15)
        opt_add_16 = P.Add()(module0_8_opt, opt_relu_15)
        opt_relu_17 = self.relu_17(opt_add_16)
        module0_9_opt = self.module0_9(opt_relu_17)
        opt_add_18 = P.Add()(module0_9_opt, opt_relu_17)
        opt_relu_19 = self.relu_19(opt_add_18)
        module0_10_opt = self.module0_10(opt_relu_19)
        opt_add_20 = P.Add()(module0_10_opt, opt_relu_19)
        opt_relu_21 = self.relu_21(opt_add_20)
        module0_11_opt = self.module0_11(opt_relu_21)
        opt_add_22 = P.Add()(module0_11_opt, opt_relu_21)
        opt_relu_23 = self.relu_23(opt_add_22)
        module0_12_opt = self.module0_12(opt_relu_23)
        opt_add_24 = P.Add()(module0_12_opt, opt_relu_23)
        opt_relu_25 = self.relu_25(opt_add_24)
        module0_13_opt = self.module0_13(opt_relu_25)
        opt_add_26 = P.Add()(module0_13_opt, opt_relu_25)
        opt_relu_27 = self.relu_27(opt_add_26)
        module0_14_opt = self.module0_14(opt_relu_27)
        opt_add_28 = P.Add()(module0_14_opt, opt_relu_27)
        opt_relu_29 = self.relu_29(opt_add_28)
        module0_15_opt = self.module0_15(opt_relu_29)
        opt_add_30 = P.Add()(module0_15_opt, opt_relu_29)
        opt_relu_31 = self.relu_31(opt_add_30)
        return opt_relu_31


class Module3(nn.Cell):
    def __init__(self):
        super(Module3, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=128,
                                 conv2d_2_in_channels=128,
                                 conv2d_2_out_channels=128,
                                 conv2d_2_stride=(1, 1),
                                 conv2d_4_in_channels=128,
                                 conv2d_4_out_channels=512,
                                 conv2d_6_in_channels=512,
                                 conv2d_6_out_channels=128,
                                 conv2d_8_in_channels=128,
                                 conv2d_8_out_channels=512)
        self.relu_1 = nn.ReLU()
        self.module0_1 = Module0(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=128,
                                 conv2d_2_in_channels=128,
                                 conv2d_2_out_channels=128,
                                 conv2d_2_stride=(1, 1),
                                 conv2d_4_in_channels=128,
                                 conv2d_4_out_channels=512,
                                 conv2d_6_in_channels=512,
                                 conv2d_6_out_channels=128,
                                 conv2d_8_in_channels=128,
                                 conv2d_8_out_channels=512)
        self.relu_3 = nn.ReLU()
        self.module0_2 = Module0(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=128,
                                 conv2d_2_in_channels=128,
                                 conv2d_2_out_channels=128,
                                 conv2d_2_stride=(1, 1),
                                 conv2d_4_in_channels=128,
                                 conv2d_4_out_channels=512,
                                 conv2d_6_in_channels=512,
                                 conv2d_6_out_channels=128,
                                 conv2d_8_in_channels=128,
                                 conv2d_8_out_channels=512)
        self.relu_5 = nn.ReLU()
        self.module0_3 = Module0(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=128,
                                 conv2d_2_in_channels=128,
                                 conv2d_2_out_channels=128,
                                 conv2d_2_stride=(1, 1),
                                 conv2d_4_in_channels=128,
                                 conv2d_4_out_channels=512,
                                 conv2d_6_in_channels=512,
                                 conv2d_6_out_channels=128,
                                 conv2d_8_in_channels=128,
                                 conv2d_8_out_channels=512)
        self.relu_7 = nn.ReLU()

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_add_0 = P.Add()(module0_0_opt, x)
        opt_relu_1 = self.relu_1(opt_add_0)
        module0_1_opt = self.module0_1(opt_relu_1)
        opt_add_2 = P.Add()(module0_1_opt, opt_relu_1)
        opt_relu_3 = self.relu_3(opt_add_2)
        module0_2_opt = self.module0_2(opt_relu_3)
        opt_add_4 = P.Add()(module0_2_opt, opt_relu_3)
        opt_relu_5 = self.relu_5(opt_add_4)
        module0_3_opt = self.module0_3(opt_relu_5)
        opt_add_6 = P.Add()(module0_3_opt, opt_relu_5)
        opt_relu_7 = self.relu_7(opt_add_6)
        return opt_relu_7


class Module16(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_2_stride, module0_0_conv2d_4_in_channels,
                 module0_0_conv2d_4_out_channels, module0_0_conv2d_6_in_channels, module0_0_conv2d_6_out_channels,
                 module0_0_conv2d_8_in_channels, module0_0_conv2d_8_out_channels, module0_1_conv2d_0_in_channels,
                 module0_1_conv2d_0_out_channels, module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels,
                 module0_1_conv2d_2_stride, module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels,
                 module0_1_conv2d_6_in_channels, module0_1_conv2d_6_out_channels, module0_1_conv2d_8_in_channels,
                 module0_1_conv2d_8_out_channels, module0_2_conv2d_0_in_channels, module0_2_conv2d_0_out_channels,
                 module0_2_conv2d_2_in_channels, module0_2_conv2d_2_out_channels, module0_2_conv2d_2_stride,
                 module0_2_conv2d_4_in_channels, module0_2_conv2d_4_out_channels, module0_2_conv2d_6_in_channels,
                 module0_2_conv2d_6_out_channels, module0_2_conv2d_8_in_channels, module0_2_conv2d_8_out_channels):
        super(Module16, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_0_conv2d_2_stride,
                                 conv2d_4_in_channels=module0_0_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_0_conv2d_4_out_channels,
                                 conv2d_6_in_channels=module0_0_conv2d_6_in_channels,
                                 conv2d_6_out_channels=module0_0_conv2d_6_out_channels,
                                 conv2d_8_in_channels=module0_0_conv2d_8_in_channels,
                                 conv2d_8_out_channels=module0_0_conv2d_8_out_channels)
        self.relu_1 = nn.ReLU()
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_1_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_1_conv2d_2_stride,
                                 conv2d_4_in_channels=module0_1_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_1_conv2d_4_out_channels,
                                 conv2d_6_in_channels=module0_1_conv2d_6_in_channels,
                                 conv2d_6_out_channels=module0_1_conv2d_6_out_channels,
                                 conv2d_8_in_channels=module0_1_conv2d_8_in_channels,
                                 conv2d_8_out_channels=module0_1_conv2d_8_out_channels)
        self.relu_3 = nn.ReLU()
        self.module0_2 = Module0(conv2d_0_in_channels=module0_2_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_2_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_2_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_2_conv2d_2_out_channels,
                                 conv2d_2_stride=module0_2_conv2d_2_stride,
                                 conv2d_4_in_channels=module0_2_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_2_conv2d_4_out_channels,
                                 conv2d_6_in_channels=module0_2_conv2d_6_in_channels,
                                 conv2d_6_out_channels=module0_2_conv2d_6_out_channels,
                                 conv2d_8_in_channels=module0_2_conv2d_8_in_channels,
                                 conv2d_8_out_channels=module0_2_conv2d_8_out_channels)
        self.relu_5 = nn.ReLU()

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        opt_add_0 = P.Add()(module0_0_opt, x)
        opt_relu_1 = self.relu_1(opt_add_0)
        module0_1_opt = self.module0_1(opt_relu_1)
        opt_add_2 = P.Add()(module0_1_opt, opt_relu_1)
        opt_relu_3 = self.relu_3(opt_add_2)
        module0_2_opt = self.module0_2(opt_relu_3)
        opt_add_4 = P.Add()(module0_2_opt, opt_relu_3)
        opt_relu_5 = self.relu_5(opt_add_4)
        return opt_relu_5


class MindSporeModel(nn.Cell):
    def __init__(self):
        super(MindSporeModel, self).__init__()
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
        self.module11_0 = Module11(conv2d_0_in_channels=32, conv2d_0_out_channels=32, conv2d_0_stride=(1, 1))
        self.module11_1 = Module11(conv2d_0_in_channels=32, conv2d_0_out_channels=64, conv2d_0_stride=(1, 1))
        self.module11_2 = Module11(conv2d_0_in_channels=64, conv2d_0_out_channels=64, conv2d_0_stride=(2, 2))
        self.module0_0 = Module0(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=64,
                                 conv2d_2_in_channels=64,
                                 conv2d_2_out_channels=64,
                                 conv2d_2_stride=(1, 1),
                                 conv2d_4_in_channels=64,
                                 conv2d_4_out_channels=256,
                                 conv2d_6_in_channels=256,
                                 conv2d_6_out_channels=64,
                                 conv2d_8_in_channels=64,
                                 conv2d_8_out_channels=256)
        self.conv2d_9 = nn.Conv2d(in_channels=64,
                                  out_channels=256,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_21 = nn.ReLU()
        self.module2_0 = Module2(module0_0_conv2d_0_in_channels=256,
                                 module0_0_conv2d_0_out_channels=64,
                                 module0_0_conv2d_2_in_channels=64,
                                 module0_0_conv2d_2_out_channels=64,
                                 module0_0_conv2d_2_stride=(1, 1),
                                 module0_0_conv2d_4_in_channels=64,
                                 module0_0_conv2d_4_out_channels=256,
                                 module0_0_conv2d_6_in_channels=256,
                                 module0_0_conv2d_6_out_channels=64,
                                 module0_0_conv2d_8_in_channels=64,
                                 module0_0_conv2d_8_out_channels=256,
                                 module0_1_conv2d_0_in_channels=256,
                                 module0_1_conv2d_0_out_channels=64,
                                 module0_1_conv2d_2_in_channels=64,
                                 module0_1_conv2d_2_out_channels=64,
                                 module0_1_conv2d_2_stride=(1, 1),
                                 module0_1_conv2d_4_in_channels=64,
                                 module0_1_conv2d_4_out_channels=256,
                                 module0_1_conv2d_6_in_channels=256,
                                 module0_1_conv2d_6_out_channels=64,
                                 module0_1_conv2d_8_in_channels=64,
                                 module0_1_conv2d_8_out_channels=256)
        self.module0_1 = Module0(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=128,
                                 conv2d_2_in_channels=128,
                                 conv2d_2_out_channels=128,
                                 conv2d_2_stride=(2, 2),
                                 conv2d_4_in_channels=128,
                                 conv2d_4_out_channels=512,
                                 conv2d_6_in_channels=512,
                                 conv2d_6_out_channels=128,
                                 conv2d_8_in_channels=128,
                                 conv2d_8_out_channels=512)
        self.module6_0 = Module6(conv2d_1_in_channels=256, conv2d_1_out_channels=512)
        self.relu_62 = nn.ReLU()
        self.module40_0 = Module40(module0_0_conv2d_0_in_channels=512,
                                   module0_0_conv2d_0_out_channels=128,
                                   module0_0_conv2d_2_in_channels=128,
                                   module0_0_conv2d_2_out_channels=128,
                                   module0_0_conv2d_2_stride=(1, 1),
                                   module0_0_conv2d_4_in_channels=128,
                                   module0_0_conv2d_4_out_channels=512,
                                   module0_0_conv2d_6_in_channels=512,
                                   module0_0_conv2d_6_out_channels=128,
                                   module0_0_conv2d_8_in_channels=128,
                                   module0_0_conv2d_8_out_channels=512,
                                   module0_1_conv2d_0_in_channels=512,
                                   module0_1_conv2d_0_out_channels=128,
                                   module0_1_conv2d_2_in_channels=128,
                                   module0_1_conv2d_2_out_channels=128,
                                   module0_1_conv2d_2_stride=(1, 1),
                                   module0_1_conv2d_4_in_channels=128,
                                   module0_1_conv2d_4_out_channels=512,
                                   module0_1_conv2d_6_in_channels=512,
                                   module0_1_conv2d_6_out_channels=128,
                                   module0_1_conv2d_8_in_channels=128,
                                   module0_1_conv2d_8_out_channels=512,
                                   module0_2_conv2d_0_in_channels=512,
                                   module0_2_conv2d_0_out_channels=128,
                                   module0_2_conv2d_2_in_channels=128,
                                   module0_2_conv2d_2_out_channels=128,
                                   module0_2_conv2d_2_stride=(1, 1),
                                   module0_2_conv2d_4_in_channels=128,
                                   module0_2_conv2d_4_out_channels=512,
                                   module0_2_conv2d_6_in_channels=512,
                                   module0_2_conv2d_6_out_channels=128,
                                   module0_2_conv2d_8_in_channels=128,
                                   module0_2_conv2d_8_out_channels=512,
                                   module0_3_conv2d_0_in_channels=512,
                                   module0_3_conv2d_0_out_channels=128,
                                   module0_3_conv2d_2_in_channels=128,
                                   module0_3_conv2d_2_out_channels=128,
                                   module0_3_conv2d_2_stride=(1, 1),
                                   module0_3_conv2d_4_in_channels=128,
                                   module0_3_conv2d_4_out_channels=512,
                                   module0_3_conv2d_6_in_channels=512,
                                   module0_3_conv2d_6_out_channels=128,
                                   module0_3_conv2d_8_in_channels=128,
                                   module0_3_conv2d_8_out_channels=512,
                                   module0_4_conv2d_0_in_channels=512,
                                   module0_4_conv2d_0_out_channels=128,
                                   module0_4_conv2d_2_in_channels=128,
                                   module0_4_conv2d_2_out_channels=128,
                                   module0_4_conv2d_2_stride=(1, 1),
                                   module0_4_conv2d_4_in_channels=128,
                                   module0_4_conv2d_4_out_channels=512,
                                   module0_4_conv2d_6_in_channels=512,
                                   module0_4_conv2d_6_out_channels=128,
                                   module0_4_conv2d_8_in_channels=128,
                                   module0_4_conv2d_8_out_channels=512,
                                   module0_5_conv2d_0_in_channels=512,
                                   module0_5_conv2d_0_out_channels=128,
                                   module0_5_conv2d_2_in_channels=128,
                                   module0_5_conv2d_2_out_channels=128,
                                   module0_5_conv2d_2_stride=(1, 1),
                                   module0_5_conv2d_4_in_channels=128,
                                   module0_5_conv2d_4_out_channels=512,
                                   module0_5_conv2d_6_in_channels=512,
                                   module0_5_conv2d_6_out_channels=128,
                                   module0_5_conv2d_8_in_channels=128,
                                   module0_5_conv2d_8_out_channels=512,
                                   module0_6_conv2d_0_in_channels=512,
                                   module0_6_conv2d_0_out_channels=128,
                                   module0_6_conv2d_2_in_channels=128,
                                   module0_6_conv2d_2_out_channels=128,
                                   module0_6_conv2d_2_stride=(1, 1),
                                   module0_6_conv2d_4_in_channels=128,
                                   module0_6_conv2d_4_out_channels=512,
                                   module0_6_conv2d_6_in_channels=512,
                                   module0_6_conv2d_6_out_channels=128,
                                   module0_6_conv2d_8_in_channels=128,
                                   module0_6_conv2d_8_out_channels=512,
                                   module0_7_conv2d_0_in_channels=512,
                                   module0_7_conv2d_0_out_channels=128,
                                   module0_7_conv2d_2_in_channels=128,
                                   module0_7_conv2d_2_out_channels=128,
                                   module0_7_conv2d_2_stride=(1, 1),
                                   module0_7_conv2d_4_in_channels=128,
                                   module0_7_conv2d_4_out_channels=512,
                                   module0_7_conv2d_6_in_channels=512,
                                   module0_7_conv2d_6_out_channels=128,
                                   module0_7_conv2d_8_in_channels=128,
                                   module0_7_conv2d_8_out_channels=512,
                                   module0_8_conv2d_0_in_channels=512,
                                   module0_8_conv2d_0_out_channels=128,
                                   module0_8_conv2d_2_in_channels=128,
                                   module0_8_conv2d_2_out_channels=128,
                                   module0_8_conv2d_2_stride=(1, 1),
                                   module0_8_conv2d_4_in_channels=128,
                                   module0_8_conv2d_4_out_channels=512,
                                   module0_8_conv2d_6_in_channels=512,
                                   module0_8_conv2d_6_out_channels=128,
                                   module0_8_conv2d_8_in_channels=128,
                                   module0_8_conv2d_8_out_channels=512,
                                   module0_9_conv2d_0_in_channels=512,
                                   module0_9_conv2d_0_out_channels=128,
                                   module0_9_conv2d_2_in_channels=128,
                                   module0_9_conv2d_2_out_channels=128,
                                   module0_9_conv2d_2_stride=(1, 1),
                                   module0_9_conv2d_4_in_channels=128,
                                   module0_9_conv2d_4_out_channels=512,
                                   module0_9_conv2d_6_in_channels=512,
                                   module0_9_conv2d_6_out_channels=128,
                                   module0_9_conv2d_8_in_channels=128,
                                   module0_9_conv2d_8_out_channels=512,
                                   module0_10_conv2d_0_in_channels=512,
                                   module0_10_conv2d_0_out_channels=128,
                                   module0_10_conv2d_2_in_channels=128,
                                   module0_10_conv2d_2_out_channels=128,
                                   module0_10_conv2d_2_stride=(1, 1),
                                   module0_10_conv2d_4_in_channels=128,
                                   module0_10_conv2d_4_out_channels=512,
                                   module0_10_conv2d_6_in_channels=512,
                                   module0_10_conv2d_6_out_channels=128,
                                   module0_10_conv2d_8_in_channels=128,
                                   module0_10_conv2d_8_out_channels=512,
                                   module0_11_conv2d_0_in_channels=512,
                                   module0_11_conv2d_0_out_channels=128,
                                   module0_11_conv2d_2_in_channels=128,
                                   module0_11_conv2d_2_out_channels=128,
                                   module0_11_conv2d_2_stride=(1, 1),
                                   module0_11_conv2d_4_in_channels=128,
                                   module0_11_conv2d_4_out_channels=512,
                                   module0_11_conv2d_6_in_channels=512,
                                   module0_11_conv2d_6_out_channels=128,
                                   module0_11_conv2d_8_in_channels=128,
                                   module0_11_conv2d_8_out_channels=512,
                                   module0_12_conv2d_0_in_channels=512,
                                   module0_12_conv2d_0_out_channels=128,
                                   module0_12_conv2d_2_in_channels=128,
                                   module0_12_conv2d_2_out_channels=128,
                                   module0_12_conv2d_2_stride=(1, 1),
                                   module0_12_conv2d_4_in_channels=128,
                                   module0_12_conv2d_4_out_channels=512,
                                   module0_12_conv2d_6_in_channels=512,
                                   module0_12_conv2d_6_out_channels=128,
                                   module0_12_conv2d_8_in_channels=128,
                                   module0_12_conv2d_8_out_channels=512,
                                   module0_13_conv2d_0_in_channels=512,
                                   module0_13_conv2d_0_out_channels=128,
                                   module0_13_conv2d_2_in_channels=128,
                                   module0_13_conv2d_2_out_channels=128,
                                   module0_13_conv2d_2_stride=(1, 1),
                                   module0_13_conv2d_4_in_channels=128,
                                   module0_13_conv2d_4_out_channels=512,
                                   module0_13_conv2d_6_in_channels=512,
                                   module0_13_conv2d_6_out_channels=128,
                                   module0_13_conv2d_8_in_channels=128,
                                   module0_13_conv2d_8_out_channels=512,
                                   module0_14_conv2d_0_in_channels=512,
                                   module0_14_conv2d_0_out_channels=128,
                                   module0_14_conv2d_2_in_channels=128,
                                   module0_14_conv2d_2_out_channels=128,
                                   module0_14_conv2d_2_stride=(1, 1),
                                   module0_14_conv2d_4_in_channels=128,
                                   module0_14_conv2d_4_out_channels=512,
                                   module0_14_conv2d_6_in_channels=512,
                                   module0_14_conv2d_6_out_channels=128,
                                   module0_14_conv2d_8_in_channels=128,
                                   module0_14_conv2d_8_out_channels=512,
                                   module0_15_conv2d_0_in_channels=512,
                                   module0_15_conv2d_0_out_channels=128,
                                   module0_15_conv2d_2_in_channels=128,
                                   module0_15_conv2d_2_out_channels=128,
                                   module0_15_conv2d_2_stride=(1, 1),
                                   module0_15_conv2d_4_in_channels=128,
                                   module0_15_conv2d_4_out_channels=512,
                                   module0_15_conv2d_6_in_channels=512,
                                   module0_15_conv2d_6_out_channels=128,
                                   module0_15_conv2d_8_in_channels=128,
                                   module0_15_conv2d_8_out_channels=512)
        self.module3_0 = Module3()
        self.module16_0 = Module16(module0_0_conv2d_0_in_channels=512,
                                   module0_0_conv2d_0_out_channels=128,
                                   module0_0_conv2d_2_in_channels=128,
                                   module0_0_conv2d_2_out_channels=128,
                                   module0_0_conv2d_2_stride=(1, 1),
                                   module0_0_conv2d_4_in_channels=128,
                                   module0_0_conv2d_4_out_channels=512,
                                   module0_0_conv2d_6_in_channels=512,
                                   module0_0_conv2d_6_out_channels=128,
                                   module0_0_conv2d_8_in_channels=128,
                                   module0_0_conv2d_8_out_channels=512,
                                   module0_1_conv2d_0_in_channels=512,
                                   module0_1_conv2d_0_out_channels=128,
                                   module0_1_conv2d_2_in_channels=128,
                                   module0_1_conv2d_2_out_channels=128,
                                   module0_1_conv2d_2_stride=(1, 1),
                                   module0_1_conv2d_4_in_channels=128,
                                   module0_1_conv2d_4_out_channels=512,
                                   module0_1_conv2d_6_in_channels=512,
                                   module0_1_conv2d_6_out_channels=128,
                                   module0_1_conv2d_8_in_channels=128,
                                   module0_1_conv2d_8_out_channels=512,
                                   module0_2_conv2d_0_in_channels=512,
                                   module0_2_conv2d_0_out_channels=128,
                                   module0_2_conv2d_2_in_channels=128,
                                   module0_2_conv2d_2_out_channels=128,
                                   module0_2_conv2d_2_stride=(1, 1),
                                   module0_2_conv2d_4_in_channels=128,
                                   module0_2_conv2d_4_out_channels=512,
                                   module0_2_conv2d_6_in_channels=512,
                                   module0_2_conv2d_6_out_channels=128,
                                   module0_2_conv2d_8_in_channels=128,
                                   module0_2_conv2d_8_out_channels=512)
        self.module0_2 = Module0(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_2_stride=(2, 2),
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024,
                                 conv2d_6_in_channels=1024,
                                 conv2d_6_out_channels=256,
                                 conv2d_8_in_channels=256,
                                 conv2d_8_out_channels=1024)
        self.module6_1 = Module6(conv2d_1_in_channels=512, conv2d_1_out_channels=1024)
        self.relu_376 = nn.ReLU()
        self.module40_1 = Module40(module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_2_in_channels=256,
                                   module0_0_conv2d_2_out_channels=256,
                                   module0_0_conv2d_2_stride=(1, 1),
                                   module0_0_conv2d_4_in_channels=256,
                                   module0_0_conv2d_4_out_channels=1024,
                                   module0_0_conv2d_6_in_channels=1024,
                                   module0_0_conv2d_6_out_channels=256,
                                   module0_0_conv2d_8_in_channels=256,
                                   module0_0_conv2d_8_out_channels=1024,
                                   module0_1_conv2d_0_in_channels=1024,
                                   module0_1_conv2d_0_out_channels=256,
                                   module0_1_conv2d_2_in_channels=256,
                                   module0_1_conv2d_2_out_channels=256,
                                   module0_1_conv2d_2_stride=(1, 1),
                                   module0_1_conv2d_4_in_channels=256,
                                   module0_1_conv2d_4_out_channels=1024,
                                   module0_1_conv2d_6_in_channels=1024,
                                   module0_1_conv2d_6_out_channels=256,
                                   module0_1_conv2d_8_in_channels=256,
                                   module0_1_conv2d_8_out_channels=1024,
                                   module0_2_conv2d_0_in_channels=1024,
                                   module0_2_conv2d_0_out_channels=256,
                                   module0_2_conv2d_2_in_channels=256,
                                   module0_2_conv2d_2_out_channels=256,
                                   module0_2_conv2d_2_stride=(1, 1),
                                   module0_2_conv2d_4_in_channels=256,
                                   module0_2_conv2d_4_out_channels=1024,
                                   module0_2_conv2d_6_in_channels=1024,
                                   module0_2_conv2d_6_out_channels=256,
                                   module0_2_conv2d_8_in_channels=256,
                                   module0_2_conv2d_8_out_channels=1024,
                                   module0_3_conv2d_0_in_channels=1024,
                                   module0_3_conv2d_0_out_channels=256,
                                   module0_3_conv2d_2_in_channels=256,
                                   module0_3_conv2d_2_out_channels=256,
                                   module0_3_conv2d_2_stride=(1, 1),
                                   module0_3_conv2d_4_in_channels=256,
                                   module0_3_conv2d_4_out_channels=1024,
                                   module0_3_conv2d_6_in_channels=1024,
                                   module0_3_conv2d_6_out_channels=256,
                                   module0_3_conv2d_8_in_channels=256,
                                   module0_3_conv2d_8_out_channels=1024,
                                   module0_4_conv2d_0_in_channels=1024,
                                   module0_4_conv2d_0_out_channels=256,
                                   module0_4_conv2d_2_in_channels=256,
                                   module0_4_conv2d_2_out_channels=256,
                                   module0_4_conv2d_2_stride=(1, 1),
                                   module0_4_conv2d_4_in_channels=256,
                                   module0_4_conv2d_4_out_channels=1024,
                                   module0_4_conv2d_6_in_channels=1024,
                                   module0_4_conv2d_6_out_channels=256,
                                   module0_4_conv2d_8_in_channels=256,
                                   module0_4_conv2d_8_out_channels=1024,
                                   module0_5_conv2d_0_in_channels=1024,
                                   module0_5_conv2d_0_out_channels=256,
                                   module0_5_conv2d_2_in_channels=256,
                                   module0_5_conv2d_2_out_channels=256,
                                   module0_5_conv2d_2_stride=(1, 1),
                                   module0_5_conv2d_4_in_channels=256,
                                   module0_5_conv2d_4_out_channels=1024,
                                   module0_5_conv2d_6_in_channels=1024,
                                   module0_5_conv2d_6_out_channels=256,
                                   module0_5_conv2d_8_in_channels=256,
                                   module0_5_conv2d_8_out_channels=1024,
                                   module0_6_conv2d_0_in_channels=1024,
                                   module0_6_conv2d_0_out_channels=256,
                                   module0_6_conv2d_2_in_channels=256,
                                   module0_6_conv2d_2_out_channels=256,
                                   module0_6_conv2d_2_stride=(1, 1),
                                   module0_6_conv2d_4_in_channels=256,
                                   module0_6_conv2d_4_out_channels=1024,
                                   module0_6_conv2d_6_in_channels=1024,
                                   module0_6_conv2d_6_out_channels=256,
                                   module0_6_conv2d_8_in_channels=256,
                                   module0_6_conv2d_8_out_channels=1024,
                                   module0_7_conv2d_0_in_channels=1024,
                                   module0_7_conv2d_0_out_channels=256,
                                   module0_7_conv2d_2_in_channels=256,
                                   module0_7_conv2d_2_out_channels=256,
                                   module0_7_conv2d_2_stride=(1, 1),
                                   module0_7_conv2d_4_in_channels=256,
                                   module0_7_conv2d_4_out_channels=1024,
                                   module0_7_conv2d_6_in_channels=1024,
                                   module0_7_conv2d_6_out_channels=256,
                                   module0_7_conv2d_8_in_channels=256,
                                   module0_7_conv2d_8_out_channels=1024,
                                   module0_8_conv2d_0_in_channels=1024,
                                   module0_8_conv2d_0_out_channels=256,
                                   module0_8_conv2d_2_in_channels=256,
                                   module0_8_conv2d_2_out_channels=256,
                                   module0_8_conv2d_2_stride=(1, 1),
                                   module0_8_conv2d_4_in_channels=256,
                                   module0_8_conv2d_4_out_channels=1024,
                                   module0_8_conv2d_6_in_channels=1024,
                                   module0_8_conv2d_6_out_channels=256,
                                   module0_8_conv2d_8_in_channels=256,
                                   module0_8_conv2d_8_out_channels=1024,
                                   module0_9_conv2d_0_in_channels=1024,
                                   module0_9_conv2d_0_out_channels=256,
                                   module0_9_conv2d_2_in_channels=256,
                                   module0_9_conv2d_2_out_channels=256,
                                   module0_9_conv2d_2_stride=(1, 1),
                                   module0_9_conv2d_4_in_channels=256,
                                   module0_9_conv2d_4_out_channels=1024,
                                   module0_9_conv2d_6_in_channels=1024,
                                   module0_9_conv2d_6_out_channels=256,
                                   module0_9_conv2d_8_in_channels=256,
                                   module0_9_conv2d_8_out_channels=1024,
                                   module0_10_conv2d_0_in_channels=1024,
                                   module0_10_conv2d_0_out_channels=256,
                                   module0_10_conv2d_2_in_channels=256,
                                   module0_10_conv2d_2_out_channels=256,
                                   module0_10_conv2d_2_stride=(1, 1),
                                   module0_10_conv2d_4_in_channels=256,
                                   module0_10_conv2d_4_out_channels=1024,
                                   module0_10_conv2d_6_in_channels=1024,
                                   module0_10_conv2d_6_out_channels=256,
                                   module0_10_conv2d_8_in_channels=256,
                                   module0_10_conv2d_8_out_channels=1024,
                                   module0_11_conv2d_0_in_channels=1024,
                                   module0_11_conv2d_0_out_channels=256,
                                   module0_11_conv2d_2_in_channels=256,
                                   module0_11_conv2d_2_out_channels=256,
                                   module0_11_conv2d_2_stride=(1, 1),
                                   module0_11_conv2d_4_in_channels=256,
                                   module0_11_conv2d_4_out_channels=1024,
                                   module0_11_conv2d_6_in_channels=1024,
                                   module0_11_conv2d_6_out_channels=256,
                                   module0_11_conv2d_8_in_channels=256,
                                   module0_11_conv2d_8_out_channels=1024,
                                   module0_12_conv2d_0_in_channels=1024,
                                   module0_12_conv2d_0_out_channels=256,
                                   module0_12_conv2d_2_in_channels=256,
                                   module0_12_conv2d_2_out_channels=256,
                                   module0_12_conv2d_2_stride=(1, 1),
                                   module0_12_conv2d_4_in_channels=256,
                                   module0_12_conv2d_4_out_channels=1024,
                                   module0_12_conv2d_6_in_channels=1024,
                                   module0_12_conv2d_6_out_channels=256,
                                   module0_12_conv2d_8_in_channels=256,
                                   module0_12_conv2d_8_out_channels=1024,
                                   module0_13_conv2d_0_in_channels=1024,
                                   module0_13_conv2d_0_out_channels=256,
                                   module0_13_conv2d_2_in_channels=256,
                                   module0_13_conv2d_2_out_channels=256,
                                   module0_13_conv2d_2_stride=(1, 1),
                                   module0_13_conv2d_4_in_channels=256,
                                   module0_13_conv2d_4_out_channels=1024,
                                   module0_13_conv2d_6_in_channels=1024,
                                   module0_13_conv2d_6_out_channels=256,
                                   module0_13_conv2d_8_in_channels=256,
                                   module0_13_conv2d_8_out_channels=1024,
                                   module0_14_conv2d_0_in_channels=1024,
                                   module0_14_conv2d_0_out_channels=256,
                                   module0_14_conv2d_2_in_channels=256,
                                   module0_14_conv2d_2_out_channels=256,
                                   module0_14_conv2d_2_stride=(1, 1),
                                   module0_14_conv2d_4_in_channels=256,
                                   module0_14_conv2d_4_out_channels=1024,
                                   module0_14_conv2d_6_in_channels=1024,
                                   module0_14_conv2d_6_out_channels=256,
                                   module0_14_conv2d_8_in_channels=256,
                                   module0_14_conv2d_8_out_channels=1024,
                                   module0_15_conv2d_0_in_channels=1024,
                                   module0_15_conv2d_0_out_channels=256,
                                   module0_15_conv2d_2_in_channels=256,
                                   module0_15_conv2d_2_out_channels=256,
                                   module0_15_conv2d_2_stride=(1, 1),
                                   module0_15_conv2d_4_in_channels=256,
                                   module0_15_conv2d_4_out_channels=1024,
                                   module0_15_conv2d_6_in_channels=1024,
                                   module0_15_conv2d_6_out_channels=256,
                                   module0_15_conv2d_8_in_channels=256,
                                   module0_15_conv2d_8_out_channels=1024)
        self.module40_2 = Module40(module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_2_in_channels=256,
                                   module0_0_conv2d_2_out_channels=256,
                                   module0_0_conv2d_2_stride=(1, 1),
                                   module0_0_conv2d_4_in_channels=256,
                                   module0_0_conv2d_4_out_channels=1024,
                                   module0_0_conv2d_6_in_channels=1024,
                                   module0_0_conv2d_6_out_channels=256,
                                   module0_0_conv2d_8_in_channels=256,
                                   module0_0_conv2d_8_out_channels=1024,
                                   module0_1_conv2d_0_in_channels=1024,
                                   module0_1_conv2d_0_out_channels=256,
                                   module0_1_conv2d_2_in_channels=256,
                                   module0_1_conv2d_2_out_channels=256,
                                   module0_1_conv2d_2_stride=(1, 1),
                                   module0_1_conv2d_4_in_channels=256,
                                   module0_1_conv2d_4_out_channels=1024,
                                   module0_1_conv2d_6_in_channels=1024,
                                   module0_1_conv2d_6_out_channels=256,
                                   module0_1_conv2d_8_in_channels=256,
                                   module0_1_conv2d_8_out_channels=1024,
                                   module0_2_conv2d_0_in_channels=1024,
                                   module0_2_conv2d_0_out_channels=256,
                                   module0_2_conv2d_2_in_channels=256,
                                   module0_2_conv2d_2_out_channels=256,
                                   module0_2_conv2d_2_stride=(1, 1),
                                   module0_2_conv2d_4_in_channels=256,
                                   module0_2_conv2d_4_out_channels=1024,
                                   module0_2_conv2d_6_in_channels=1024,
                                   module0_2_conv2d_6_out_channels=256,
                                   module0_2_conv2d_8_in_channels=256,
                                   module0_2_conv2d_8_out_channels=1024,
                                   module0_3_conv2d_0_in_channels=1024,
                                   module0_3_conv2d_0_out_channels=256,
                                   module0_3_conv2d_2_in_channels=256,
                                   module0_3_conv2d_2_out_channels=256,
                                   module0_3_conv2d_2_stride=(1, 1),
                                   module0_3_conv2d_4_in_channels=256,
                                   module0_3_conv2d_4_out_channels=1024,
                                   module0_3_conv2d_6_in_channels=1024,
                                   module0_3_conv2d_6_out_channels=256,
                                   module0_3_conv2d_8_in_channels=256,
                                   module0_3_conv2d_8_out_channels=1024,
                                   module0_4_conv2d_0_in_channels=1024,
                                   module0_4_conv2d_0_out_channels=256,
                                   module0_4_conv2d_2_in_channels=256,
                                   module0_4_conv2d_2_out_channels=256,
                                   module0_4_conv2d_2_stride=(1, 1),
                                   module0_4_conv2d_4_in_channels=256,
                                   module0_4_conv2d_4_out_channels=1024,
                                   module0_4_conv2d_6_in_channels=1024,
                                   module0_4_conv2d_6_out_channels=256,
                                   module0_4_conv2d_8_in_channels=256,
                                   module0_4_conv2d_8_out_channels=1024,
                                   module0_5_conv2d_0_in_channels=1024,
                                   module0_5_conv2d_0_out_channels=256,
                                   module0_5_conv2d_2_in_channels=256,
                                   module0_5_conv2d_2_out_channels=256,
                                   module0_5_conv2d_2_stride=(1, 1),
                                   module0_5_conv2d_4_in_channels=256,
                                   module0_5_conv2d_4_out_channels=1024,
                                   module0_5_conv2d_6_in_channels=1024,
                                   module0_5_conv2d_6_out_channels=256,
                                   module0_5_conv2d_8_in_channels=256,
                                   module0_5_conv2d_8_out_channels=1024,
                                   module0_6_conv2d_0_in_channels=1024,
                                   module0_6_conv2d_0_out_channels=256,
                                   module0_6_conv2d_2_in_channels=256,
                                   module0_6_conv2d_2_out_channels=256,
                                   module0_6_conv2d_2_stride=(1, 1),
                                   module0_6_conv2d_4_in_channels=256,
                                   module0_6_conv2d_4_out_channels=1024,
                                   module0_6_conv2d_6_in_channels=1024,
                                   module0_6_conv2d_6_out_channels=256,
                                   module0_6_conv2d_8_in_channels=256,
                                   module0_6_conv2d_8_out_channels=1024,
                                   module0_7_conv2d_0_in_channels=1024,
                                   module0_7_conv2d_0_out_channels=256,
                                   module0_7_conv2d_2_in_channels=256,
                                   module0_7_conv2d_2_out_channels=256,
                                   module0_7_conv2d_2_stride=(1, 1),
                                   module0_7_conv2d_4_in_channels=256,
                                   module0_7_conv2d_4_out_channels=1024,
                                   module0_7_conv2d_6_in_channels=1024,
                                   module0_7_conv2d_6_out_channels=256,
                                   module0_7_conv2d_8_in_channels=256,
                                   module0_7_conv2d_8_out_channels=1024,
                                   module0_8_conv2d_0_in_channels=1024,
                                   module0_8_conv2d_0_out_channels=256,
                                   module0_8_conv2d_2_in_channels=256,
                                   module0_8_conv2d_2_out_channels=256,
                                   module0_8_conv2d_2_stride=(1, 1),
                                   module0_8_conv2d_4_in_channels=256,
                                   module0_8_conv2d_4_out_channels=1024,
                                   module0_8_conv2d_6_in_channels=1024,
                                   module0_8_conv2d_6_out_channels=256,
                                   module0_8_conv2d_8_in_channels=256,
                                   module0_8_conv2d_8_out_channels=1024,
                                   module0_9_conv2d_0_in_channels=1024,
                                   module0_9_conv2d_0_out_channels=256,
                                   module0_9_conv2d_2_in_channels=256,
                                   module0_9_conv2d_2_out_channels=256,
                                   module0_9_conv2d_2_stride=(1, 1),
                                   module0_9_conv2d_4_in_channels=256,
                                   module0_9_conv2d_4_out_channels=1024,
                                   module0_9_conv2d_6_in_channels=1024,
                                   module0_9_conv2d_6_out_channels=256,
                                   module0_9_conv2d_8_in_channels=256,
                                   module0_9_conv2d_8_out_channels=1024,
                                   module0_10_conv2d_0_in_channels=1024,
                                   module0_10_conv2d_0_out_channels=256,
                                   module0_10_conv2d_2_in_channels=256,
                                   module0_10_conv2d_2_out_channels=256,
                                   module0_10_conv2d_2_stride=(1, 1),
                                   module0_10_conv2d_4_in_channels=256,
                                   module0_10_conv2d_4_out_channels=1024,
                                   module0_10_conv2d_6_in_channels=1024,
                                   module0_10_conv2d_6_out_channels=256,
                                   module0_10_conv2d_8_in_channels=256,
                                   module0_10_conv2d_8_out_channels=1024,
                                   module0_11_conv2d_0_in_channels=1024,
                                   module0_11_conv2d_0_out_channels=256,
                                   module0_11_conv2d_2_in_channels=256,
                                   module0_11_conv2d_2_out_channels=256,
                                   module0_11_conv2d_2_stride=(1, 1),
                                   module0_11_conv2d_4_in_channels=256,
                                   module0_11_conv2d_4_out_channels=1024,
                                   module0_11_conv2d_6_in_channels=1024,
                                   module0_11_conv2d_6_out_channels=256,
                                   module0_11_conv2d_8_in_channels=256,
                                   module0_11_conv2d_8_out_channels=1024,
                                   module0_12_conv2d_0_in_channels=1024,
                                   module0_12_conv2d_0_out_channels=256,
                                   module0_12_conv2d_2_in_channels=256,
                                   module0_12_conv2d_2_out_channels=256,
                                   module0_12_conv2d_2_stride=(1, 1),
                                   module0_12_conv2d_4_in_channels=256,
                                   module0_12_conv2d_4_out_channels=1024,
                                   module0_12_conv2d_6_in_channels=1024,
                                   module0_12_conv2d_6_out_channels=256,
                                   module0_12_conv2d_8_in_channels=256,
                                   module0_12_conv2d_8_out_channels=1024,
                                   module0_13_conv2d_0_in_channels=1024,
                                   module0_13_conv2d_0_out_channels=256,
                                   module0_13_conv2d_2_in_channels=256,
                                   module0_13_conv2d_2_out_channels=256,
                                   module0_13_conv2d_2_stride=(1, 1),
                                   module0_13_conv2d_4_in_channels=256,
                                   module0_13_conv2d_4_out_channels=1024,
                                   module0_13_conv2d_6_in_channels=1024,
                                   module0_13_conv2d_6_out_channels=256,
                                   module0_13_conv2d_8_in_channels=256,
                                   module0_13_conv2d_8_out_channels=1024,
                                   module0_14_conv2d_0_in_channels=1024,
                                   module0_14_conv2d_0_out_channels=256,
                                   module0_14_conv2d_2_in_channels=256,
                                   module0_14_conv2d_2_out_channels=256,
                                   module0_14_conv2d_2_stride=(1, 1),
                                   module0_14_conv2d_4_in_channels=256,
                                   module0_14_conv2d_4_out_channels=1024,
                                   module0_14_conv2d_6_in_channels=1024,
                                   module0_14_conv2d_6_out_channels=256,
                                   module0_14_conv2d_8_in_channels=256,
                                   module0_14_conv2d_8_out_channels=1024,
                                   module0_15_conv2d_0_in_channels=1024,
                                   module0_15_conv2d_0_out_channels=256,
                                   module0_15_conv2d_2_in_channels=256,
                                   module0_15_conv2d_2_out_channels=256,
                                   module0_15_conv2d_2_stride=(1, 1),
                                   module0_15_conv2d_4_in_channels=256,
                                   module0_15_conv2d_4_out_channels=1024,
                                   module0_15_conv2d_6_in_channels=1024,
                                   module0_15_conv2d_6_out_channels=256,
                                   module0_15_conv2d_8_in_channels=256,
                                   module0_15_conv2d_8_out_channels=1024)
        self.module16_1 = Module16(module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_2_in_channels=256,
                                   module0_0_conv2d_2_out_channels=256,
                                   module0_0_conv2d_2_stride=(1, 1),
                                   module0_0_conv2d_4_in_channels=256,
                                   module0_0_conv2d_4_out_channels=1024,
                                   module0_0_conv2d_6_in_channels=1024,
                                   module0_0_conv2d_6_out_channels=256,
                                   module0_0_conv2d_8_in_channels=256,
                                   module0_0_conv2d_8_out_channels=1024,
                                   module0_1_conv2d_0_in_channels=1024,
                                   module0_1_conv2d_0_out_channels=256,
                                   module0_1_conv2d_2_in_channels=256,
                                   module0_1_conv2d_2_out_channels=256,
                                   module0_1_conv2d_2_stride=(1, 1),
                                   module0_1_conv2d_4_in_channels=256,
                                   module0_1_conv2d_4_out_channels=1024,
                                   module0_1_conv2d_6_in_channels=1024,
                                   module0_1_conv2d_6_out_channels=256,
                                   module0_1_conv2d_8_in_channels=256,
                                   module0_1_conv2d_8_out_channels=1024,
                                   module0_2_conv2d_0_in_channels=1024,
                                   module0_2_conv2d_0_out_channels=256,
                                   module0_2_conv2d_2_in_channels=256,
                                   module0_2_conv2d_2_out_channels=256,
                                   module0_2_conv2d_2_stride=(1, 1),
                                   module0_2_conv2d_4_in_channels=256,
                                   module0_2_conv2d_4_out_channels=1024,
                                   module0_2_conv2d_6_in_channels=1024,
                                   module0_2_conv2d_6_out_channels=256,
                                   module0_2_conv2d_8_in_channels=256,
                                   module0_2_conv2d_8_out_channels=1024)
        self.module0_3 = Module0(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=512,
                                 conv2d_2_in_channels=512,
                                 conv2d_2_out_channels=512,
                                 conv2d_2_stride=(2, 2),
                                 conv2d_4_in_channels=512,
                                 conv2d_4_out_channels=2048,
                                 conv2d_6_in_channels=2048,
                                 conv2d_6_out_channels=512,
                                 conv2d_8_in_channels=512,
                                 conv2d_8_out_channels=2048)
        self.module6_2 = Module6(conv2d_1_in_channels=1024, conv2d_1_out_channels=2048)
        self.relu_846 = nn.ReLU()
        self.module2_1 = Module2(module0_0_conv2d_0_in_channels=2048,
                                 module0_0_conv2d_0_out_channels=512,
                                 module0_0_conv2d_2_in_channels=512,
                                 module0_0_conv2d_2_out_channels=512,
                                 module0_0_conv2d_2_stride=(1, 1),
                                 module0_0_conv2d_4_in_channels=512,
                                 module0_0_conv2d_4_out_channels=2048,
                                 module0_0_conv2d_6_in_channels=2048,
                                 module0_0_conv2d_6_out_channels=512,
                                 module0_0_conv2d_8_in_channels=512,
                                 module0_0_conv2d_8_out_channels=2048,
                                 module0_1_conv2d_0_in_channels=2048,
                                 module0_1_conv2d_0_out_channels=512,
                                 module0_1_conv2d_2_in_channels=512,
                                 module0_1_conv2d_2_out_channels=512,
                                 module0_1_conv2d_2_stride=(1, 1),
                                 module0_1_conv2d_4_in_channels=512,
                                 module0_1_conv2d_4_out_channels=2048,
                                 module0_1_conv2d_6_in_channels=2048,
                                 module0_1_conv2d_6_out_channels=512,
                                 module0_1_conv2d_8_in_channels=512,
                                 module0_1_conv2d_8_out_channels=2048)
        self.avgpool2d_873 = nn.AvgPool2d(kernel_size=(8, 8))
        self.flatten_874 = nn.Flatten()
        self.dense_875 = nn.Dense(in_channels=2048, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        module11_0_opt = self.module11_0(opt_relu_1)
        module11_1_opt = self.module11_1(module11_0_opt)
        module11_2_opt = self.module11_2(module11_1_opt)
        module0_0_opt = self.module0_0(module11_2_opt)
        opt_conv2d_9 = self.conv2d_9(module11_2_opt)
        opt_add_20 = P.Add()(module0_0_opt, opt_conv2d_9)
        opt_relu_21 = self.relu_21(opt_add_20)
        module2_0_opt = self.module2_0(opt_relu_21)
        module0_1_opt = self.module0_1(module2_0_opt)
        module6_0_opt = self.module6_0(module2_0_opt)
        opt_add_61 = P.Add()(module0_1_opt, module6_0_opt)
        opt_relu_62 = self.relu_62(opt_add_61)
        module40_0_opt = self.module40_0(opt_relu_62)
        module3_0_opt = self.module3_0(module40_0_opt)
        module16_0_opt = self.module16_0(module3_0_opt)
        module0_2_opt = self.module0_2(module16_0_opt)
        module6_1_opt = self.module6_1(module16_0_opt)
        opt_add_375 = P.Add()(module0_2_opt, module6_1_opt)
        opt_relu_376 = self.relu_376(opt_add_375)
        module40_1_opt = self.module40_1(opt_relu_376)
        module40_2_opt = self.module40_2(module40_1_opt)
        module16_1_opt = self.module16_1(module40_2_opt)
        module0_3_opt = self.module0_3(module16_1_opt)
        module6_2_opt = self.module6_2(module16_1_opt)
        opt_add_845 = P.Add()(module0_3_opt, module6_2_opt)
        opt_relu_846 = self.relu_846(opt_add_845)
        module2_1_opt = self.module2_1(opt_relu_846)
        opt_avgpool2d_873 = self.avgpool2d_873(module2_1_opt)
        opt_flatten_874 = self.flatten_874(opt_avgpool2d_873)
        opt_dense_875 = self.dense_875(opt_flatten_874)
        return opt_dense_875
