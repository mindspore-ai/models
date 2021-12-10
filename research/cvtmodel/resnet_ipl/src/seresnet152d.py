import mindspore.ops as P
from mindspore import nn


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


class Module4(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_2_stride, module0_0_conv2d_4_in_channels,
                 module0_0_conv2d_4_out_channels, module0_0_conv2d_6_in_channels, module0_0_conv2d_6_out_channels,
                 module0_0_conv2d_8_in_channels, module0_0_conv2d_8_out_channels, module0_1_conv2d_0_in_channels,
                 module0_1_conv2d_0_out_channels, module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels,
                 module0_1_conv2d_2_stride, module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels,
                 module0_1_conv2d_6_in_channels, module0_1_conv2d_6_out_channels, module0_1_conv2d_8_in_channels,
                 module0_1_conv2d_8_out_channels):
        super(Module4, self).__init__()
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


class Module3(nn.Cell):
    def __init__(self, conv2d_1_in_channels, conv2d_1_out_channels):
        super(Module3, self).__init__()
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
                 module0_6_conv2d_8_in_channels, module0_6_conv2d_8_out_channels):
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
        return opt_relu_13


class Module60(nn.Cell):
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
                 module0_7_conv2d_8_out_channels):
        super(Module60, self).__init__()
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
        return opt_relu_15


class Module5(nn.Cell):
    def __init__(self):
        super(Module5, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_2_stride=(1, 1),
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024,
                                 conv2d_6_in_channels=1024,
                                 conv2d_6_out_channels=64,
                                 conv2d_8_in_channels=64,
                                 conv2d_8_out_channels=1024)
        self.relu_1 = nn.ReLU()
        self.module0_1 = Module0(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_2_stride=(1, 1),
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024,
                                 conv2d_6_in_channels=1024,
                                 conv2d_6_out_channels=64,
                                 conv2d_8_in_channels=64,
                                 conv2d_8_out_channels=1024)
        self.relu_3 = nn.ReLU()
        self.module0_2 = Module0(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_2_stride=(1, 1),
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024,
                                 conv2d_6_in_channels=1024,
                                 conv2d_6_out_channels=64,
                                 conv2d_8_in_channels=64,
                                 conv2d_8_out_channels=1024)
        self.relu_5 = nn.ReLU()
        self.module0_3 = Module0(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_2_stride=(1, 1),
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024,
                                 conv2d_6_in_channels=1024,
                                 conv2d_6_out_channels=64,
                                 conv2d_8_in_channels=64,
                                 conv2d_8_out_channels=1024)
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
        self.conv2d_2 = nn.Conv2d(in_channels=32,
                                  out_channels=32,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_3 = nn.ReLU()
        self.conv2d_4 = nn.Conv2d(in_channels=32,
                                  out_channels=64,
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
        self.module0_0 = Module0(conv2d_0_in_channels=64,
                                 conv2d_0_out_channels=64,
                                 conv2d_2_in_channels=64,
                                 conv2d_2_out_channels=64,
                                 conv2d_2_stride=(1, 1),
                                 conv2d_4_in_channels=64,
                                 conv2d_4_out_channels=256,
                                 conv2d_6_in_channels=256,
                                 conv2d_6_out_channels=16,
                                 conv2d_8_in_channels=16,
                                 conv2d_8_out_channels=256)
        self.conv2d_8 = nn.Conv2d(in_channels=64,
                                  out_channels=256,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_20 = nn.ReLU()
        self.module4_0 = Module4(module0_0_conv2d_0_in_channels=256,
                                 module0_0_conv2d_0_out_channels=64,
                                 module0_0_conv2d_2_in_channels=64,
                                 module0_0_conv2d_2_out_channels=64,
                                 module0_0_conv2d_2_stride=(1, 1),
                                 module0_0_conv2d_4_in_channels=64,
                                 module0_0_conv2d_4_out_channels=256,
                                 module0_0_conv2d_6_in_channels=256,
                                 module0_0_conv2d_6_out_channels=16,
                                 module0_0_conv2d_8_in_channels=16,
                                 module0_0_conv2d_8_out_channels=256,
                                 module0_1_conv2d_0_in_channels=256,
                                 module0_1_conv2d_0_out_channels=64,
                                 module0_1_conv2d_2_in_channels=64,
                                 module0_1_conv2d_2_out_channels=64,
                                 module0_1_conv2d_2_stride=(1, 1),
                                 module0_1_conv2d_4_in_channels=64,
                                 module0_1_conv2d_4_out_channels=256,
                                 module0_1_conv2d_6_in_channels=256,
                                 module0_1_conv2d_6_out_channels=16,
                                 module0_1_conv2d_8_in_channels=16,
                                 module0_1_conv2d_8_out_channels=256)
        self.module0_1 = Module0(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=128,
                                 conv2d_2_in_channels=128,
                                 conv2d_2_out_channels=128,
                                 conv2d_2_stride=(2, 2),
                                 conv2d_4_in_channels=128,
                                 conv2d_4_out_channels=512,
                                 conv2d_6_in_channels=512,
                                 conv2d_6_out_channels=32,
                                 conv2d_8_in_channels=32,
                                 conv2d_8_out_channels=512)
        self.module3_0 = Module3(conv2d_1_in_channels=256, conv2d_1_out_channels=512)
        self.relu_61 = nn.ReLU()
        self.module16_0 = Module16(module0_0_conv2d_0_in_channels=512,
                                   module0_0_conv2d_0_out_channels=128,
                                   module0_0_conv2d_2_in_channels=128,
                                   module0_0_conv2d_2_out_channels=128,
                                   module0_0_conv2d_2_stride=(1, 1),
                                   module0_0_conv2d_4_in_channels=128,
                                   module0_0_conv2d_4_out_channels=512,
                                   module0_0_conv2d_6_in_channels=512,
                                   module0_0_conv2d_6_out_channels=32,
                                   module0_0_conv2d_8_in_channels=32,
                                   module0_0_conv2d_8_out_channels=512,
                                   module0_1_conv2d_0_in_channels=512,
                                   module0_1_conv2d_0_out_channels=128,
                                   module0_1_conv2d_2_in_channels=128,
                                   module0_1_conv2d_2_out_channels=128,
                                   module0_1_conv2d_2_stride=(1, 1),
                                   module0_1_conv2d_4_in_channels=128,
                                   module0_1_conv2d_4_out_channels=512,
                                   module0_1_conv2d_6_in_channels=512,
                                   module0_1_conv2d_6_out_channels=32,
                                   module0_1_conv2d_8_in_channels=32,
                                   module0_1_conv2d_8_out_channels=512,
                                   module0_2_conv2d_0_in_channels=512,
                                   module0_2_conv2d_0_out_channels=128,
                                   module0_2_conv2d_2_in_channels=128,
                                   module0_2_conv2d_2_out_channels=128,
                                   module0_2_conv2d_2_stride=(1, 1),
                                   module0_2_conv2d_4_in_channels=128,
                                   module0_2_conv2d_4_out_channels=512,
                                   module0_2_conv2d_6_in_channels=512,
                                   module0_2_conv2d_6_out_channels=32,
                                   module0_2_conv2d_8_in_channels=32,
                                   module0_2_conv2d_8_out_channels=512,
                                   module0_3_conv2d_0_in_channels=512,
                                   module0_3_conv2d_0_out_channels=128,
                                   module0_3_conv2d_2_in_channels=128,
                                   module0_3_conv2d_2_out_channels=128,
                                   module0_3_conv2d_2_stride=(1, 1),
                                   module0_3_conv2d_4_in_channels=128,
                                   module0_3_conv2d_4_out_channels=512,
                                   module0_3_conv2d_6_in_channels=512,
                                   module0_3_conv2d_6_out_channels=32,
                                   module0_3_conv2d_8_in_channels=32,
                                   module0_3_conv2d_8_out_channels=512,
                                   module0_4_conv2d_0_in_channels=512,
                                   module0_4_conv2d_0_out_channels=128,
                                   module0_4_conv2d_2_in_channels=128,
                                   module0_4_conv2d_2_out_channels=128,
                                   module0_4_conv2d_2_stride=(1, 1),
                                   module0_4_conv2d_4_in_channels=128,
                                   module0_4_conv2d_4_out_channels=512,
                                   module0_4_conv2d_6_in_channels=512,
                                   module0_4_conv2d_6_out_channels=32,
                                   module0_4_conv2d_8_in_channels=32,
                                   module0_4_conv2d_8_out_channels=512,
                                   module0_5_conv2d_0_in_channels=512,
                                   module0_5_conv2d_0_out_channels=128,
                                   module0_5_conv2d_2_in_channels=128,
                                   module0_5_conv2d_2_out_channels=128,
                                   module0_5_conv2d_2_stride=(1, 1),
                                   module0_5_conv2d_4_in_channels=128,
                                   module0_5_conv2d_4_out_channels=512,
                                   module0_5_conv2d_6_in_channels=512,
                                   module0_5_conv2d_6_out_channels=32,
                                   module0_5_conv2d_8_in_channels=32,
                                   module0_5_conv2d_8_out_channels=512,
                                   module0_6_conv2d_0_in_channels=512,
                                   module0_6_conv2d_0_out_channels=128,
                                   module0_6_conv2d_2_in_channels=128,
                                   module0_6_conv2d_2_out_channels=128,
                                   module0_6_conv2d_2_stride=(1, 1),
                                   module0_6_conv2d_4_in_channels=128,
                                   module0_6_conv2d_4_out_channels=512,
                                   module0_6_conv2d_6_in_channels=512,
                                   module0_6_conv2d_6_out_channels=32,
                                   module0_6_conv2d_8_in_channels=32,
                                   module0_6_conv2d_8_out_channels=512)
        self.module0_2 = Module0(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_2_stride=(2, 2),
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=1024,
                                 conv2d_6_in_channels=1024,
                                 conv2d_6_out_channels=64,
                                 conv2d_8_in_channels=64,
                                 conv2d_8_out_channels=1024)
        self.module3_1 = Module3(conv2d_1_in_channels=512, conv2d_1_out_channels=1024)
        self.relu_167 = nn.ReLU()
        self.module60_0 = Module60(module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_2_in_channels=256,
                                   module0_0_conv2d_2_out_channels=256,
                                   module0_0_conv2d_2_stride=(1, 1),
                                   module0_0_conv2d_4_in_channels=256,
                                   module0_0_conv2d_4_out_channels=1024,
                                   module0_0_conv2d_6_in_channels=1024,
                                   module0_0_conv2d_6_out_channels=64,
                                   module0_0_conv2d_8_in_channels=64,
                                   module0_0_conv2d_8_out_channels=1024,
                                   module0_1_conv2d_0_in_channels=1024,
                                   module0_1_conv2d_0_out_channels=256,
                                   module0_1_conv2d_2_in_channels=256,
                                   module0_1_conv2d_2_out_channels=256,
                                   module0_1_conv2d_2_stride=(1, 1),
                                   module0_1_conv2d_4_in_channels=256,
                                   module0_1_conv2d_4_out_channels=1024,
                                   module0_1_conv2d_6_in_channels=1024,
                                   module0_1_conv2d_6_out_channels=64,
                                   module0_1_conv2d_8_in_channels=64,
                                   module0_1_conv2d_8_out_channels=1024,
                                   module0_2_conv2d_0_in_channels=1024,
                                   module0_2_conv2d_0_out_channels=256,
                                   module0_2_conv2d_2_in_channels=256,
                                   module0_2_conv2d_2_out_channels=256,
                                   module0_2_conv2d_2_stride=(1, 1),
                                   module0_2_conv2d_4_in_channels=256,
                                   module0_2_conv2d_4_out_channels=1024,
                                   module0_2_conv2d_6_in_channels=1024,
                                   module0_2_conv2d_6_out_channels=64,
                                   module0_2_conv2d_8_in_channels=64,
                                   module0_2_conv2d_8_out_channels=1024,
                                   module0_3_conv2d_0_in_channels=1024,
                                   module0_3_conv2d_0_out_channels=256,
                                   module0_3_conv2d_2_in_channels=256,
                                   module0_3_conv2d_2_out_channels=256,
                                   module0_3_conv2d_2_stride=(1, 1),
                                   module0_3_conv2d_4_in_channels=256,
                                   module0_3_conv2d_4_out_channels=1024,
                                   module0_3_conv2d_6_in_channels=1024,
                                   module0_3_conv2d_6_out_channels=64,
                                   module0_3_conv2d_8_in_channels=64,
                                   module0_3_conv2d_8_out_channels=1024,
                                   module0_4_conv2d_0_in_channels=1024,
                                   module0_4_conv2d_0_out_channels=256,
                                   module0_4_conv2d_2_in_channels=256,
                                   module0_4_conv2d_2_out_channels=256,
                                   module0_4_conv2d_2_stride=(1, 1),
                                   module0_4_conv2d_4_in_channels=256,
                                   module0_4_conv2d_4_out_channels=1024,
                                   module0_4_conv2d_6_in_channels=1024,
                                   module0_4_conv2d_6_out_channels=64,
                                   module0_4_conv2d_8_in_channels=64,
                                   module0_4_conv2d_8_out_channels=1024,
                                   module0_5_conv2d_0_in_channels=1024,
                                   module0_5_conv2d_0_out_channels=256,
                                   module0_5_conv2d_2_in_channels=256,
                                   module0_5_conv2d_2_out_channels=256,
                                   module0_5_conv2d_2_stride=(1, 1),
                                   module0_5_conv2d_4_in_channels=256,
                                   module0_5_conv2d_4_out_channels=1024,
                                   module0_5_conv2d_6_in_channels=1024,
                                   module0_5_conv2d_6_out_channels=64,
                                   module0_5_conv2d_8_in_channels=64,
                                   module0_5_conv2d_8_out_channels=1024,
                                   module0_6_conv2d_0_in_channels=1024,
                                   module0_6_conv2d_0_out_channels=256,
                                   module0_6_conv2d_2_in_channels=256,
                                   module0_6_conv2d_2_out_channels=256,
                                   module0_6_conv2d_2_stride=(1, 1),
                                   module0_6_conv2d_4_in_channels=256,
                                   module0_6_conv2d_4_out_channels=1024,
                                   module0_6_conv2d_6_in_channels=1024,
                                   module0_6_conv2d_6_out_channels=64,
                                   module0_6_conv2d_8_in_channels=64,
                                   module0_6_conv2d_8_out_channels=1024,
                                   module0_7_conv2d_0_in_channels=1024,
                                   module0_7_conv2d_0_out_channels=256,
                                   module0_7_conv2d_2_in_channels=256,
                                   module0_7_conv2d_2_out_channels=256,
                                   module0_7_conv2d_2_stride=(1, 1),
                                   module0_7_conv2d_4_in_channels=256,
                                   module0_7_conv2d_4_out_channels=1024,
                                   module0_7_conv2d_6_in_channels=1024,
                                   module0_7_conv2d_6_out_channels=64,
                                   module0_7_conv2d_8_in_channels=64,
                                   module0_7_conv2d_8_out_channels=1024)
        self.module60_1 = Module60(module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_2_in_channels=256,
                                   module0_0_conv2d_2_out_channels=256,
                                   module0_0_conv2d_2_stride=(1, 1),
                                   module0_0_conv2d_4_in_channels=256,
                                   module0_0_conv2d_4_out_channels=1024,
                                   module0_0_conv2d_6_in_channels=1024,
                                   module0_0_conv2d_6_out_channels=64,
                                   module0_0_conv2d_8_in_channels=64,
                                   module0_0_conv2d_8_out_channels=1024,
                                   module0_1_conv2d_0_in_channels=1024,
                                   module0_1_conv2d_0_out_channels=256,
                                   module0_1_conv2d_2_in_channels=256,
                                   module0_1_conv2d_2_out_channels=256,
                                   module0_1_conv2d_2_stride=(1, 1),
                                   module0_1_conv2d_4_in_channels=256,
                                   module0_1_conv2d_4_out_channels=1024,
                                   module0_1_conv2d_6_in_channels=1024,
                                   module0_1_conv2d_6_out_channels=64,
                                   module0_1_conv2d_8_in_channels=64,
                                   module0_1_conv2d_8_out_channels=1024,
                                   module0_2_conv2d_0_in_channels=1024,
                                   module0_2_conv2d_0_out_channels=256,
                                   module0_2_conv2d_2_in_channels=256,
                                   module0_2_conv2d_2_out_channels=256,
                                   module0_2_conv2d_2_stride=(1, 1),
                                   module0_2_conv2d_4_in_channels=256,
                                   module0_2_conv2d_4_out_channels=1024,
                                   module0_2_conv2d_6_in_channels=1024,
                                   module0_2_conv2d_6_out_channels=64,
                                   module0_2_conv2d_8_in_channels=64,
                                   module0_2_conv2d_8_out_channels=1024,
                                   module0_3_conv2d_0_in_channels=1024,
                                   module0_3_conv2d_0_out_channels=256,
                                   module0_3_conv2d_2_in_channels=256,
                                   module0_3_conv2d_2_out_channels=256,
                                   module0_3_conv2d_2_stride=(1, 1),
                                   module0_3_conv2d_4_in_channels=256,
                                   module0_3_conv2d_4_out_channels=1024,
                                   module0_3_conv2d_6_in_channels=1024,
                                   module0_3_conv2d_6_out_channels=64,
                                   module0_3_conv2d_8_in_channels=64,
                                   module0_3_conv2d_8_out_channels=1024,
                                   module0_4_conv2d_0_in_channels=1024,
                                   module0_4_conv2d_0_out_channels=256,
                                   module0_4_conv2d_2_in_channels=256,
                                   module0_4_conv2d_2_out_channels=256,
                                   module0_4_conv2d_2_stride=(1, 1),
                                   module0_4_conv2d_4_in_channels=256,
                                   module0_4_conv2d_4_out_channels=1024,
                                   module0_4_conv2d_6_in_channels=1024,
                                   module0_4_conv2d_6_out_channels=64,
                                   module0_4_conv2d_8_in_channels=64,
                                   module0_4_conv2d_8_out_channels=1024,
                                   module0_5_conv2d_0_in_channels=1024,
                                   module0_5_conv2d_0_out_channels=256,
                                   module0_5_conv2d_2_in_channels=256,
                                   module0_5_conv2d_2_out_channels=256,
                                   module0_5_conv2d_2_stride=(1, 1),
                                   module0_5_conv2d_4_in_channels=256,
                                   module0_5_conv2d_4_out_channels=1024,
                                   module0_5_conv2d_6_in_channels=1024,
                                   module0_5_conv2d_6_out_channels=64,
                                   module0_5_conv2d_8_in_channels=64,
                                   module0_5_conv2d_8_out_channels=1024,
                                   module0_6_conv2d_0_in_channels=1024,
                                   module0_6_conv2d_0_out_channels=256,
                                   module0_6_conv2d_2_in_channels=256,
                                   module0_6_conv2d_2_out_channels=256,
                                   module0_6_conv2d_2_stride=(1, 1),
                                   module0_6_conv2d_4_in_channels=256,
                                   module0_6_conv2d_4_out_channels=1024,
                                   module0_6_conv2d_6_in_channels=1024,
                                   module0_6_conv2d_6_out_channels=64,
                                   module0_6_conv2d_8_in_channels=64,
                                   module0_6_conv2d_8_out_channels=1024,
                                   module0_7_conv2d_0_in_channels=1024,
                                   module0_7_conv2d_0_out_channels=256,
                                   module0_7_conv2d_2_in_channels=256,
                                   module0_7_conv2d_2_out_channels=256,
                                   module0_7_conv2d_2_stride=(1, 1),
                                   module0_7_conv2d_4_in_channels=256,
                                   module0_7_conv2d_4_out_channels=1024,
                                   module0_7_conv2d_6_in_channels=1024,
                                   module0_7_conv2d_6_out_channels=64,
                                   module0_7_conv2d_8_in_channels=64,
                                   module0_7_conv2d_8_out_channels=1024)
        self.module60_2 = Module60(module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_2_in_channels=256,
                                   module0_0_conv2d_2_out_channels=256,
                                   module0_0_conv2d_2_stride=(1, 1),
                                   module0_0_conv2d_4_in_channels=256,
                                   module0_0_conv2d_4_out_channels=1024,
                                   module0_0_conv2d_6_in_channels=1024,
                                   module0_0_conv2d_6_out_channels=64,
                                   module0_0_conv2d_8_in_channels=64,
                                   module0_0_conv2d_8_out_channels=1024,
                                   module0_1_conv2d_0_in_channels=1024,
                                   module0_1_conv2d_0_out_channels=256,
                                   module0_1_conv2d_2_in_channels=256,
                                   module0_1_conv2d_2_out_channels=256,
                                   module0_1_conv2d_2_stride=(1, 1),
                                   module0_1_conv2d_4_in_channels=256,
                                   module0_1_conv2d_4_out_channels=1024,
                                   module0_1_conv2d_6_in_channels=1024,
                                   module0_1_conv2d_6_out_channels=64,
                                   module0_1_conv2d_8_in_channels=64,
                                   module0_1_conv2d_8_out_channels=1024,
                                   module0_2_conv2d_0_in_channels=1024,
                                   module0_2_conv2d_0_out_channels=256,
                                   module0_2_conv2d_2_in_channels=256,
                                   module0_2_conv2d_2_out_channels=256,
                                   module0_2_conv2d_2_stride=(1, 1),
                                   module0_2_conv2d_4_in_channels=256,
                                   module0_2_conv2d_4_out_channels=1024,
                                   module0_2_conv2d_6_in_channels=1024,
                                   module0_2_conv2d_6_out_channels=64,
                                   module0_2_conv2d_8_in_channels=64,
                                   module0_2_conv2d_8_out_channels=1024,
                                   module0_3_conv2d_0_in_channels=1024,
                                   module0_3_conv2d_0_out_channels=256,
                                   module0_3_conv2d_2_in_channels=256,
                                   module0_3_conv2d_2_out_channels=256,
                                   module0_3_conv2d_2_stride=(1, 1),
                                   module0_3_conv2d_4_in_channels=256,
                                   module0_3_conv2d_4_out_channels=1024,
                                   module0_3_conv2d_6_in_channels=1024,
                                   module0_3_conv2d_6_out_channels=64,
                                   module0_3_conv2d_8_in_channels=64,
                                   module0_3_conv2d_8_out_channels=1024,
                                   module0_4_conv2d_0_in_channels=1024,
                                   module0_4_conv2d_0_out_channels=256,
                                   module0_4_conv2d_2_in_channels=256,
                                   module0_4_conv2d_2_out_channels=256,
                                   module0_4_conv2d_2_stride=(1, 1),
                                   module0_4_conv2d_4_in_channels=256,
                                   module0_4_conv2d_4_out_channels=1024,
                                   module0_4_conv2d_6_in_channels=1024,
                                   module0_4_conv2d_6_out_channels=64,
                                   module0_4_conv2d_8_in_channels=64,
                                   module0_4_conv2d_8_out_channels=1024,
                                   module0_5_conv2d_0_in_channels=1024,
                                   module0_5_conv2d_0_out_channels=256,
                                   module0_5_conv2d_2_in_channels=256,
                                   module0_5_conv2d_2_out_channels=256,
                                   module0_5_conv2d_2_stride=(1, 1),
                                   module0_5_conv2d_4_in_channels=256,
                                   module0_5_conv2d_4_out_channels=1024,
                                   module0_5_conv2d_6_in_channels=1024,
                                   module0_5_conv2d_6_out_channels=64,
                                   module0_5_conv2d_8_in_channels=64,
                                   module0_5_conv2d_8_out_channels=1024,
                                   module0_6_conv2d_0_in_channels=1024,
                                   module0_6_conv2d_0_out_channels=256,
                                   module0_6_conv2d_2_in_channels=256,
                                   module0_6_conv2d_2_out_channels=256,
                                   module0_6_conv2d_2_stride=(1, 1),
                                   module0_6_conv2d_4_in_channels=256,
                                   module0_6_conv2d_4_out_channels=1024,
                                   module0_6_conv2d_6_in_channels=1024,
                                   module0_6_conv2d_6_out_channels=64,
                                   module0_6_conv2d_8_in_channels=64,
                                   module0_6_conv2d_8_out_channels=1024,
                                   module0_7_conv2d_0_in_channels=1024,
                                   module0_7_conv2d_0_out_channels=256,
                                   module0_7_conv2d_2_in_channels=256,
                                   module0_7_conv2d_2_out_channels=256,
                                   module0_7_conv2d_2_stride=(1, 1),
                                   module0_7_conv2d_4_in_channels=256,
                                   module0_7_conv2d_4_out_channels=1024,
                                   module0_7_conv2d_6_in_channels=1024,
                                   module0_7_conv2d_6_out_channels=64,
                                   module0_7_conv2d_8_in_channels=64,
                                   module0_7_conv2d_8_out_channels=1024)
        self.module5_0 = Module5()
        self.module16_1 = Module16(module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=256,
                                   module0_0_conv2d_2_in_channels=256,
                                   module0_0_conv2d_2_out_channels=256,
                                   module0_0_conv2d_2_stride=(1, 1),
                                   module0_0_conv2d_4_in_channels=256,
                                   module0_0_conv2d_4_out_channels=1024,
                                   module0_0_conv2d_6_in_channels=1024,
                                   module0_0_conv2d_6_out_channels=64,
                                   module0_0_conv2d_8_in_channels=64,
                                   module0_0_conv2d_8_out_channels=1024,
                                   module0_1_conv2d_0_in_channels=1024,
                                   module0_1_conv2d_0_out_channels=256,
                                   module0_1_conv2d_2_in_channels=256,
                                   module0_1_conv2d_2_out_channels=256,
                                   module0_1_conv2d_2_stride=(1, 1),
                                   module0_1_conv2d_4_in_channels=256,
                                   module0_1_conv2d_4_out_channels=1024,
                                   module0_1_conv2d_6_in_channels=1024,
                                   module0_1_conv2d_6_out_channels=64,
                                   module0_1_conv2d_8_in_channels=64,
                                   module0_1_conv2d_8_out_channels=1024,
                                   module0_2_conv2d_0_in_channels=1024,
                                   module0_2_conv2d_0_out_channels=256,
                                   module0_2_conv2d_2_in_channels=256,
                                   module0_2_conv2d_2_out_channels=256,
                                   module0_2_conv2d_2_stride=(1, 1),
                                   module0_2_conv2d_4_in_channels=256,
                                   module0_2_conv2d_4_out_channels=1024,
                                   module0_2_conv2d_6_in_channels=1024,
                                   module0_2_conv2d_6_out_channels=64,
                                   module0_2_conv2d_8_in_channels=64,
                                   module0_2_conv2d_8_out_channels=1024,
                                   module0_3_conv2d_0_in_channels=1024,
                                   module0_3_conv2d_0_out_channels=256,
                                   module0_3_conv2d_2_in_channels=256,
                                   module0_3_conv2d_2_out_channels=256,
                                   module0_3_conv2d_2_stride=(1, 1),
                                   module0_3_conv2d_4_in_channels=256,
                                   module0_3_conv2d_4_out_channels=1024,
                                   module0_3_conv2d_6_in_channels=1024,
                                   module0_3_conv2d_6_out_channels=64,
                                   module0_3_conv2d_8_in_channels=64,
                                   module0_3_conv2d_8_out_channels=1024,
                                   module0_4_conv2d_0_in_channels=1024,
                                   module0_4_conv2d_0_out_channels=256,
                                   module0_4_conv2d_2_in_channels=256,
                                   module0_4_conv2d_2_out_channels=256,
                                   module0_4_conv2d_2_stride=(1, 1),
                                   module0_4_conv2d_4_in_channels=256,
                                   module0_4_conv2d_4_out_channels=1024,
                                   module0_4_conv2d_6_in_channels=1024,
                                   module0_4_conv2d_6_out_channels=64,
                                   module0_4_conv2d_8_in_channels=64,
                                   module0_4_conv2d_8_out_channels=1024,
                                   module0_5_conv2d_0_in_channels=1024,
                                   module0_5_conv2d_0_out_channels=256,
                                   module0_5_conv2d_2_in_channels=256,
                                   module0_5_conv2d_2_out_channels=256,
                                   module0_5_conv2d_2_stride=(1, 1),
                                   module0_5_conv2d_4_in_channels=256,
                                   module0_5_conv2d_4_out_channels=1024,
                                   module0_5_conv2d_6_in_channels=1024,
                                   module0_5_conv2d_6_out_channels=64,
                                   module0_5_conv2d_8_in_channels=64,
                                   module0_5_conv2d_8_out_channels=1024,
                                   module0_6_conv2d_0_in_channels=1024,
                                   module0_6_conv2d_0_out_channels=256,
                                   module0_6_conv2d_2_in_channels=256,
                                   module0_6_conv2d_2_out_channels=256,
                                   module0_6_conv2d_2_stride=(1, 1),
                                   module0_6_conv2d_4_in_channels=256,
                                   module0_6_conv2d_4_out_channels=1024,
                                   module0_6_conv2d_6_in_channels=1024,
                                   module0_6_conv2d_6_out_channels=64,
                                   module0_6_conv2d_8_in_channels=64,
                                   module0_6_conv2d_8_out_channels=1024)
        self.module0_3 = Module0(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=512,
                                 conv2d_2_in_channels=512,
                                 conv2d_2_out_channels=512,
                                 conv2d_2_stride=(2, 2),
                                 conv2d_4_in_channels=512,
                                 conv2d_4_out_channels=2048,
                                 conv2d_6_in_channels=2048,
                                 conv2d_6_out_channels=128,
                                 conv2d_8_in_channels=128,
                                 conv2d_8_out_channels=2048)
        self.module3_2 = Module3(conv2d_1_in_channels=1024, conv2d_1_out_channels=2048)
        self.relu_637 = nn.ReLU()
        self.module4_1 = Module4(module0_0_conv2d_0_in_channels=2048,
                                 module0_0_conv2d_0_out_channels=512,
                                 module0_0_conv2d_2_in_channels=512,
                                 module0_0_conv2d_2_out_channels=512,
                                 module0_0_conv2d_2_stride=(1, 1),
                                 module0_0_conv2d_4_in_channels=512,
                                 module0_0_conv2d_4_out_channels=2048,
                                 module0_0_conv2d_6_in_channels=2048,
                                 module0_0_conv2d_6_out_channels=128,
                                 module0_0_conv2d_8_in_channels=128,
                                 module0_0_conv2d_8_out_channels=2048,
                                 module0_1_conv2d_0_in_channels=2048,
                                 module0_1_conv2d_0_out_channels=512,
                                 module0_1_conv2d_2_in_channels=512,
                                 module0_1_conv2d_2_out_channels=512,
                                 module0_1_conv2d_2_stride=(1, 1),
                                 module0_1_conv2d_4_in_channels=512,
                                 module0_1_conv2d_4_out_channels=2048,
                                 module0_1_conv2d_6_in_channels=2048,
                                 module0_1_conv2d_6_out_channels=128,
                                 module0_1_conv2d_8_in_channels=128,
                                 module0_1_conv2d_8_out_channels=2048)
        self.avgpool2d_664 = nn.AvgPool2d(kernel_size=(8, 8))
        self.flatten_665 = nn.Flatten()
        self.dense_666 = nn.Dense(in_channels=2048, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_relu_3)
        opt_relu_5 = self.relu_5(opt_conv2d_4)
        opt_maxpool2d_6 = self.pad_maxpool2d_6(opt_relu_5)
        opt_maxpool2d_6 = self.maxpool2d_6(opt_maxpool2d_6)
        module0_0_opt = self.module0_0(opt_maxpool2d_6)
        opt_conv2d_8 = self.conv2d_8(opt_maxpool2d_6)
        opt_add_19 = P.Add()(module0_0_opt, opt_conv2d_8)
        opt_relu_20 = self.relu_20(opt_add_19)
        module4_0_opt = self.module4_0(opt_relu_20)
        module0_1_opt = self.module0_1(module4_0_opt)
        module3_0_opt = self.module3_0(module4_0_opt)
        opt_add_60 = P.Add()(module0_1_opt, module3_0_opt)
        opt_relu_61 = self.relu_61(opt_add_60)
        module16_0_opt = self.module16_0(opt_relu_61)
        module0_2_opt = self.module0_2(module16_0_opt)
        module3_1_opt = self.module3_1(module16_0_opt)
        opt_add_166 = P.Add()(module0_2_opt, module3_1_opt)
        opt_relu_167 = self.relu_167(opt_add_166)
        module60_0_opt = self.module60_0(opt_relu_167)
        module60_1_opt = self.module60_1(module60_0_opt)
        module60_2_opt = self.module60_2(module60_1_opt)
        module5_0_opt = self.module5_0(module60_2_opt)
        module16_1_opt = self.module16_1(module5_0_opt)
        module0_3_opt = self.module0_3(module16_1_opt)
        module3_2_opt = self.module3_2(module16_1_opt)
        opt_add_636 = P.Add()(module0_3_opt, module3_2_opt)
        opt_relu_637 = self.relu_637(opt_add_636)
        module4_1_opt = self.module4_1(opt_relu_637)
        opt_avgpool2d_664 = self.avgpool2d_664(module4_1_opt)
        opt_flatten_665 = self.flatten_665(opt_avgpool2d_664)
        opt_dense_666 = self.dense_666(opt_flatten_665)
        return opt_dense_666
