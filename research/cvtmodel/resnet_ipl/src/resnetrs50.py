import mindspore.ops as P
from mindspore import nn


class Module7(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_stride):
        super(Module7, self).__init__()
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


class Module8(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_2_stride, module0_0_conv2d_4_in_channels,
                 module0_0_conv2d_4_out_channels, module0_0_conv2d_6_in_channels, module0_0_conv2d_6_out_channels,
                 module0_0_conv2d_8_in_channels, module0_0_conv2d_8_out_channels, module0_1_conv2d_0_in_channels,
                 module0_1_conv2d_0_out_channels, module0_1_conv2d_2_in_channels, module0_1_conv2d_2_out_channels,
                 module0_1_conv2d_2_stride, module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels,
                 module0_1_conv2d_6_in_channels, module0_1_conv2d_6_out_channels, module0_1_conv2d_8_in_channels,
                 module0_1_conv2d_8_out_channels):
        super(Module8, self).__init__()
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


class Module4(nn.Cell):
    def __init__(self, conv2d_1_in_channels, conv2d_1_out_channels):
        super(Module4, self).__init__()
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


class Module11(nn.Cell):
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
        super(Module11, self).__init__()
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
        self.module7_0 = Module7(conv2d_0_in_channels=32, conv2d_0_out_channels=32, conv2d_0_stride=(1, 1))
        self.module7_1 = Module7(conv2d_0_in_channels=32, conv2d_0_out_channels=64, conv2d_0_stride=(1, 1))
        self.module7_2 = Module7(conv2d_0_in_channels=64, conv2d_0_out_channels=64, conv2d_0_stride=(2, 2))
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
        self.module8_0 = Module8(module0_0_conv2d_0_in_channels=256,
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
        self.module4_0 = Module4(conv2d_1_in_channels=256, conv2d_1_out_channels=512)
        self.relu_62 = nn.ReLU()
        self.module11_0 = Module11(module0_0_conv2d_0_in_channels=512,
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
        self.module4_1 = Module4(conv2d_1_in_channels=512, conv2d_1_out_channels=1024)
        self.relu_116 = nn.ReLU()
        self.module8_1 = Module8(module0_0_conv2d_0_in_channels=1024,
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
                                 module0_1_conv2d_8_out_channels=1024)
        self.module11_1 = Module11(module0_0_conv2d_0_in_channels=1024,
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
        self.module4_2 = Module4(conv2d_1_in_channels=1024, conv2d_1_out_channels=2048)
        self.relu_196 = nn.ReLU()
        self.module8_2 = Module8(module0_0_conv2d_0_in_channels=2048,
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
        self.avgpool2d_223 = nn.AvgPool2d(kernel_size=(7, 7))
        self.flatten_224 = nn.Flatten()
        self.dense_225 = nn.Dense(in_channels=2048, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        module7_0_opt = self.module7_0(opt_relu_1)
        module7_1_opt = self.module7_1(module7_0_opt)
        module7_2_opt = self.module7_2(module7_1_opt)
        module0_0_opt = self.module0_0(module7_2_opt)
        opt_conv2d_9 = self.conv2d_9(module7_2_opt)
        opt_add_20 = P.Add()(module0_0_opt, opt_conv2d_9)
        opt_relu_21 = self.relu_21(opt_add_20)
        module8_0_opt = self.module8_0(opt_relu_21)
        module0_1_opt = self.module0_1(module8_0_opt)
        module4_0_opt = self.module4_0(module8_0_opt)
        opt_add_61 = P.Add()(module0_1_opt, module4_0_opt)
        opt_relu_62 = self.relu_62(opt_add_61)
        module11_0_opt = self.module11_0(opt_relu_62)
        module0_2_opt = self.module0_2(module11_0_opt)
        module4_1_opt = self.module4_1(module11_0_opt)
        opt_add_115 = P.Add()(module0_2_opt, module4_1_opt)
        opt_relu_116 = self.relu_116(opt_add_115)
        module8_1_opt = self.module8_1(opt_relu_116)
        module11_1_opt = self.module11_1(module8_1_opt)
        module0_3_opt = self.module0_3(module11_1_opt)
        module4_2_opt = self.module4_2(module11_1_opt)
        opt_add_195 = P.Add()(module0_3_opt, module4_2_opt)
        opt_relu_196 = self.relu_196(opt_add_195)
        module8_2_opt = self.module8_2(opt_relu_196)
        opt_avgpool2d_223 = self.avgpool2d_223(module8_2_opt)
        opt_flatten_224 = self.flatten_224(opt_avgpool2d_223)
        opt_dense_225 = self.dense_225(opt_flatten_224)
        return opt_dense_225
