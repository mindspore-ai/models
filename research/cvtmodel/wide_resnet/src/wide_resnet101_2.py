import mindspore.ops as P
from mindspore import nn


class Module3(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_stride,
                 conv2d_0_padding, conv2d_0_pad_mode):
        super(Module3, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=conv2d_0_kernel_size,
                                  stride=conv2d_0_stride,
                                  padding=conv2d_0_padding,
                                  pad_mode=conv2d_0_pad_mode,
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        return opt_relu_1


class Module8(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module3_0_conv2d_0_in_channels,
                 module3_0_conv2d_0_out_channels, module3_0_conv2d_0_kernel_size, module3_0_conv2d_0_stride,
                 module3_0_conv2d_0_padding, module3_0_conv2d_0_pad_mode, module3_1_conv2d_0_in_channels,
                 module3_1_conv2d_0_out_channels, module3_1_conv2d_0_kernel_size, module3_1_conv2d_0_stride,
                 module3_1_conv2d_0_padding, module3_1_conv2d_0_pad_mode):
        super(Module8, self).__init__()
        self.module3_0 = Module3(conv2d_0_in_channels=module3_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module3_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module3_0_conv2d_0_kernel_size,
                                 conv2d_0_stride=module3_0_conv2d_0_stride,
                                 conv2d_0_padding=module3_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module3_0_conv2d_0_pad_mode)
        self.module3_1 = Module3(conv2d_0_in_channels=module3_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module3_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module3_1_conv2d_0_kernel_size,
                                 conv2d_0_stride=module3_1_conv2d_0_stride,
                                 conv2d_0_padding=module3_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module3_1_conv2d_0_pad_mode)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        module3_0_opt = self.module3_0(x)
        module3_1_opt = self.module3_1(module3_0_opt)
        opt_conv2d_0 = self.conv2d_0(module3_1_opt)
        return opt_conv2d_0


class Module0(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_2_in_channels, conv2d_2_out_channels,
                 conv2d_4_in_channels, conv2d_4_out_channels):
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
        self.relu_6 = nn.ReLU()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_conv2d_2 = self.conv2d_2(opt_relu_1)
        opt_relu_3 = self.relu_3(opt_conv2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_relu_3)
        opt_add_5 = P.Add()(opt_conv2d_4, x)
        opt_relu_6 = self.relu_6(opt_add_5)
        return opt_relu_6


class Module9(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_4_in_channels, module0_0_conv2d_4_out_channels,
                 module0_1_conv2d_0_in_channels, module0_1_conv2d_0_out_channels, module0_1_conv2d_2_in_channels,
                 module0_1_conv2d_2_out_channels, module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels):
        super(Module9, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_0_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_0_conv2d_4_out_channels)
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_1_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_1_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_1_conv2d_4_out_channels)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        return module0_1_opt


class Module13(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_4_in_channels, module0_0_conv2d_4_out_channels,
                 module0_1_conv2d_0_in_channels, module0_1_conv2d_0_out_channels, module0_1_conv2d_2_in_channels,
                 module0_1_conv2d_2_out_channels, module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels,
                 module0_2_conv2d_0_in_channels, module0_2_conv2d_0_out_channels, module0_2_conv2d_2_in_channels,
                 module0_2_conv2d_2_out_channels, module0_2_conv2d_4_in_channels, module0_2_conv2d_4_out_channels,
                 module0_3_conv2d_0_in_channels, module0_3_conv2d_0_out_channels, module0_3_conv2d_2_in_channels,
                 module0_3_conv2d_2_out_channels, module0_3_conv2d_4_in_channels, module0_3_conv2d_4_out_channels,
                 module0_4_conv2d_0_in_channels, module0_4_conv2d_0_out_channels, module0_4_conv2d_2_in_channels,
                 module0_4_conv2d_2_out_channels, module0_4_conv2d_4_in_channels, module0_4_conv2d_4_out_channels,
                 module0_5_conv2d_0_in_channels, module0_5_conv2d_0_out_channels, module0_5_conv2d_2_in_channels,
                 module0_5_conv2d_2_out_channels, module0_5_conv2d_4_in_channels, module0_5_conv2d_4_out_channels,
                 module0_6_conv2d_0_in_channels, module0_6_conv2d_0_out_channels, module0_6_conv2d_2_in_channels,
                 module0_6_conv2d_2_out_channels, module0_6_conv2d_4_in_channels, module0_6_conv2d_4_out_channels,
                 module0_7_conv2d_0_in_channels, module0_7_conv2d_0_out_channels, module0_7_conv2d_2_in_channels,
                 module0_7_conv2d_2_out_channels, module0_7_conv2d_4_in_channels, module0_7_conv2d_4_out_channels):
        super(Module13, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_0_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_0_conv2d_4_out_channels)
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_1_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_1_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_1_conv2d_4_out_channels)
        self.module0_2 = Module0(conv2d_0_in_channels=module0_2_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_2_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_2_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_2_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_2_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_2_conv2d_4_out_channels)
        self.module0_3 = Module0(conv2d_0_in_channels=module0_3_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_3_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_3_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_3_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_3_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_3_conv2d_4_out_channels)
        self.module0_4 = Module0(conv2d_0_in_channels=module0_4_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_4_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_4_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_4_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_4_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_4_conv2d_4_out_channels)
        self.module0_5 = Module0(conv2d_0_in_channels=module0_5_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_5_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_5_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_5_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_5_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_5_conv2d_4_out_channels)
        self.module0_6 = Module0(conv2d_0_in_channels=module0_6_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_6_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_6_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_6_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_6_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_6_conv2d_4_out_channels)
        self.module0_7 = Module0(conv2d_0_in_channels=module0_7_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_7_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_7_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_7_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_7_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_7_conv2d_4_out_channels)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        module0_2_opt = self.module0_2(module0_1_opt)
        module0_3_opt = self.module0_3(module0_2_opt)
        module0_4_opt = self.module0_4(module0_3_opt)
        module0_5_opt = self.module0_5(module0_4_opt)
        module0_6_opt = self.module0_6(module0_5_opt)
        module0_7_opt = self.module0_7(module0_6_opt)
        return module0_7_opt


class Module12(nn.Cell):
    def __init__(self, module0_0_conv2d_0_in_channels, module0_0_conv2d_0_out_channels, module0_0_conv2d_2_in_channels,
                 module0_0_conv2d_2_out_channels, module0_0_conv2d_4_in_channels, module0_0_conv2d_4_out_channels,
                 module0_1_conv2d_0_in_channels, module0_1_conv2d_0_out_channels, module0_1_conv2d_2_in_channels,
                 module0_1_conv2d_2_out_channels, module0_1_conv2d_4_in_channels, module0_1_conv2d_4_out_channels,
                 module0_2_conv2d_0_in_channels, module0_2_conv2d_0_out_channels, module0_2_conv2d_2_in_channels,
                 module0_2_conv2d_2_out_channels, module0_2_conv2d_4_in_channels, module0_2_conv2d_4_out_channels,
                 module0_3_conv2d_0_in_channels, module0_3_conv2d_0_out_channels, module0_3_conv2d_2_in_channels,
                 module0_3_conv2d_2_out_channels, module0_3_conv2d_4_in_channels, module0_3_conv2d_4_out_channels):
        super(Module12, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_0_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_0_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_0_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_0_conv2d_4_out_channels)
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_1_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_1_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_1_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_1_conv2d_4_out_channels)
        self.module0_2 = Module0(conv2d_0_in_channels=module0_2_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_2_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_2_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_2_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_2_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_2_conv2d_4_out_channels)
        self.module0_3 = Module0(conv2d_0_in_channels=module0_3_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_3_conv2d_0_out_channels,
                                 conv2d_2_in_channels=module0_3_conv2d_2_in_channels,
                                 conv2d_2_out_channels=module0_3_conv2d_2_out_channels,
                                 conv2d_4_in_channels=module0_3_conv2d_4_in_channels,
                                 conv2d_4_out_channels=module0_3_conv2d_4_out_channels)

    def construct(self, x):
        module0_0_opt = self.module0_0(x)
        module0_1_opt = self.module0_1(module0_0_opt)
        module0_2_opt = self.module0_2(module0_1_opt)
        module0_3_opt = self.module0_3(module0_2_opt)
        return module0_3_opt


class MainModel(nn.Cell):
    def __init__(self):
        super(MainModel, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=3,
                                  out_channels=64,
                                  kernel_size=(7, 7),
                                  stride=(2, 2),
                                  padding=(3, 3, 3, 3),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_1 = nn.ReLU()
        self.pad_maxpool2d_2 = nn.Pad(paddings=((0, 0), (0, 0), (1, 0), (1, 0)))
        self.maxpool2d_2 = nn.MaxPool2d(kernel_size=(3, 3), stride=(2, 2))
        self.module8_0 = Module8(conv2d_0_in_channels=128,
                                 conv2d_0_out_channels=256,
                                 module3_0_conv2d_0_in_channels=64,
                                 module3_0_conv2d_0_out_channels=128,
                                 module3_0_conv2d_0_kernel_size=(1, 1),
                                 module3_0_conv2d_0_stride=(1, 1),
                                 module3_0_conv2d_0_padding=0,
                                 module3_0_conv2d_0_pad_mode="valid",
                                 module3_1_conv2d_0_in_channels=128,
                                 module3_1_conv2d_0_out_channels=128,
                                 module3_1_conv2d_0_kernel_size=(3, 3),
                                 module3_1_conv2d_0_stride=(1, 1),
                                 module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module3_1_conv2d_0_pad_mode="pad")
        self.conv2d_4 = nn.Conv2d(in_channels=64,
                                  out_channels=256,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.relu_10 = nn.ReLU()
        self.module9_0 = Module9(module0_0_conv2d_0_in_channels=256,
                                 module0_0_conv2d_0_out_channels=128,
                                 module0_0_conv2d_2_in_channels=128,
                                 module0_0_conv2d_2_out_channels=128,
                                 module0_0_conv2d_4_in_channels=128,
                                 module0_0_conv2d_4_out_channels=256,
                                 module0_1_conv2d_0_in_channels=256,
                                 module0_1_conv2d_0_out_channels=128,
                                 module0_1_conv2d_2_in_channels=128,
                                 module0_1_conv2d_2_out_channels=128,
                                 module0_1_conv2d_4_in_channels=128,
                                 module0_1_conv2d_4_out_channels=256)
        self.module8_1 = Module8(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=512,
                                 module3_0_conv2d_0_in_channels=256,
                                 module3_0_conv2d_0_out_channels=256,
                                 module3_0_conv2d_0_kernel_size=(1, 1),
                                 module3_0_conv2d_0_stride=(1, 1),
                                 module3_0_conv2d_0_padding=0,
                                 module3_0_conv2d_0_pad_mode="valid",
                                 module3_1_conv2d_0_in_channels=256,
                                 module3_1_conv2d_0_out_channels=256,
                                 module3_1_conv2d_0_kernel_size=(3, 3),
                                 module3_1_conv2d_0_stride=(2, 2),
                                 module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module3_1_conv2d_0_pad_mode="pad")
        self.conv2d_26 = nn.Conv2d(in_channels=256,
                                   out_channels=512,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_32 = nn.ReLU()
        self.module9_1 = Module9(module0_0_conv2d_0_in_channels=512,
                                 module0_0_conv2d_0_out_channels=256,
                                 module0_0_conv2d_2_in_channels=256,
                                 module0_0_conv2d_2_out_channels=256,
                                 module0_0_conv2d_4_in_channels=256,
                                 module0_0_conv2d_4_out_channels=512,
                                 module0_1_conv2d_0_in_channels=512,
                                 module0_1_conv2d_0_out_channels=256,
                                 module0_1_conv2d_2_in_channels=256,
                                 module0_1_conv2d_2_out_channels=256,
                                 module0_1_conv2d_4_in_channels=256,
                                 module0_1_conv2d_4_out_channels=512)
        self.module0_0 = Module0(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=256,
                                 conv2d_2_in_channels=256,
                                 conv2d_2_out_channels=256,
                                 conv2d_4_in_channels=256,
                                 conv2d_4_out_channels=512)
        self.module8_2 = Module8(conv2d_0_in_channels=512,
                                 conv2d_0_out_channels=1024,
                                 module3_0_conv2d_0_in_channels=512,
                                 module3_0_conv2d_0_out_channels=512,
                                 module3_0_conv2d_0_kernel_size=(1, 1),
                                 module3_0_conv2d_0_stride=(1, 1),
                                 module3_0_conv2d_0_padding=0,
                                 module3_0_conv2d_0_pad_mode="valid",
                                 module3_1_conv2d_0_in_channels=512,
                                 module3_1_conv2d_0_out_channels=512,
                                 module3_1_conv2d_0_kernel_size=(3, 3),
                                 module3_1_conv2d_0_stride=(2, 2),
                                 module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module3_1_conv2d_0_pad_mode="pad")
        self.conv2d_55 = nn.Conv2d(in_channels=512,
                                   out_channels=1024,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.relu_61 = nn.ReLU()
        self.module13_0 = Module13(module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=512,
                                   module0_0_conv2d_2_in_channels=512,
                                   module0_0_conv2d_2_out_channels=512,
                                   module0_0_conv2d_4_in_channels=512,
                                   module0_0_conv2d_4_out_channels=1024,
                                   module0_1_conv2d_0_in_channels=1024,
                                   module0_1_conv2d_0_out_channels=512,
                                   module0_1_conv2d_2_in_channels=512,
                                   module0_1_conv2d_2_out_channels=512,
                                   module0_1_conv2d_4_in_channels=512,
                                   module0_1_conv2d_4_out_channels=1024,
                                   module0_2_conv2d_0_in_channels=1024,
                                   module0_2_conv2d_0_out_channels=512,
                                   module0_2_conv2d_2_in_channels=512,
                                   module0_2_conv2d_2_out_channels=512,
                                   module0_2_conv2d_4_in_channels=512,
                                   module0_2_conv2d_4_out_channels=1024,
                                   module0_3_conv2d_0_in_channels=1024,
                                   module0_3_conv2d_0_out_channels=512,
                                   module0_3_conv2d_2_in_channels=512,
                                   module0_3_conv2d_2_out_channels=512,
                                   module0_3_conv2d_4_in_channels=512,
                                   module0_3_conv2d_4_out_channels=1024,
                                   module0_4_conv2d_0_in_channels=1024,
                                   module0_4_conv2d_0_out_channels=512,
                                   module0_4_conv2d_2_in_channels=512,
                                   module0_4_conv2d_2_out_channels=512,
                                   module0_4_conv2d_4_in_channels=512,
                                   module0_4_conv2d_4_out_channels=1024,
                                   module0_5_conv2d_0_in_channels=1024,
                                   module0_5_conv2d_0_out_channels=512,
                                   module0_5_conv2d_2_in_channels=512,
                                   module0_5_conv2d_2_out_channels=512,
                                   module0_5_conv2d_4_in_channels=512,
                                   module0_5_conv2d_4_out_channels=1024,
                                   module0_6_conv2d_0_in_channels=1024,
                                   module0_6_conv2d_0_out_channels=512,
                                   module0_6_conv2d_2_in_channels=512,
                                   module0_6_conv2d_2_out_channels=512,
                                   module0_6_conv2d_4_in_channels=512,
                                   module0_6_conv2d_4_out_channels=1024,
                                   module0_7_conv2d_0_in_channels=1024,
                                   module0_7_conv2d_0_out_channels=512,
                                   module0_7_conv2d_2_in_channels=512,
                                   module0_7_conv2d_2_out_channels=512,
                                   module0_7_conv2d_4_in_channels=512,
                                   module0_7_conv2d_4_out_channels=1024)
        self.module13_1 = Module13(module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=512,
                                   module0_0_conv2d_2_in_channels=512,
                                   module0_0_conv2d_2_out_channels=512,
                                   module0_0_conv2d_4_in_channels=512,
                                   module0_0_conv2d_4_out_channels=1024,
                                   module0_1_conv2d_0_in_channels=1024,
                                   module0_1_conv2d_0_out_channels=512,
                                   module0_1_conv2d_2_in_channels=512,
                                   module0_1_conv2d_2_out_channels=512,
                                   module0_1_conv2d_4_in_channels=512,
                                   module0_1_conv2d_4_out_channels=1024,
                                   module0_2_conv2d_0_in_channels=1024,
                                   module0_2_conv2d_0_out_channels=512,
                                   module0_2_conv2d_2_in_channels=512,
                                   module0_2_conv2d_2_out_channels=512,
                                   module0_2_conv2d_4_in_channels=512,
                                   module0_2_conv2d_4_out_channels=1024,
                                   module0_3_conv2d_0_in_channels=1024,
                                   module0_3_conv2d_0_out_channels=512,
                                   module0_3_conv2d_2_in_channels=512,
                                   module0_3_conv2d_2_out_channels=512,
                                   module0_3_conv2d_4_in_channels=512,
                                   module0_3_conv2d_4_out_channels=1024,
                                   module0_4_conv2d_0_in_channels=1024,
                                   module0_4_conv2d_0_out_channels=512,
                                   module0_4_conv2d_2_in_channels=512,
                                   module0_4_conv2d_2_out_channels=512,
                                   module0_4_conv2d_4_in_channels=512,
                                   module0_4_conv2d_4_out_channels=1024,
                                   module0_5_conv2d_0_in_channels=1024,
                                   module0_5_conv2d_0_out_channels=512,
                                   module0_5_conv2d_2_in_channels=512,
                                   module0_5_conv2d_2_out_channels=512,
                                   module0_5_conv2d_4_in_channels=512,
                                   module0_5_conv2d_4_out_channels=1024,
                                   module0_6_conv2d_0_in_channels=1024,
                                   module0_6_conv2d_0_out_channels=512,
                                   module0_6_conv2d_2_in_channels=512,
                                   module0_6_conv2d_2_out_channels=512,
                                   module0_6_conv2d_4_in_channels=512,
                                   module0_6_conv2d_4_out_channels=1024,
                                   module0_7_conv2d_0_in_channels=1024,
                                   module0_7_conv2d_0_out_channels=512,
                                   module0_7_conv2d_2_in_channels=512,
                                   module0_7_conv2d_2_out_channels=512,
                                   module0_7_conv2d_4_in_channels=512,
                                   module0_7_conv2d_4_out_channels=1024)
        self.module12_0 = Module12(module0_0_conv2d_0_in_channels=1024,
                                   module0_0_conv2d_0_out_channels=512,
                                   module0_0_conv2d_2_in_channels=512,
                                   module0_0_conv2d_2_out_channels=512,
                                   module0_0_conv2d_4_in_channels=512,
                                   module0_0_conv2d_4_out_channels=1024,
                                   module0_1_conv2d_0_in_channels=1024,
                                   module0_1_conv2d_0_out_channels=512,
                                   module0_1_conv2d_2_in_channels=512,
                                   module0_1_conv2d_2_out_channels=512,
                                   module0_1_conv2d_4_in_channels=512,
                                   module0_1_conv2d_4_out_channels=1024,
                                   module0_2_conv2d_0_in_channels=1024,
                                   module0_2_conv2d_0_out_channels=512,
                                   module0_2_conv2d_2_in_channels=512,
                                   module0_2_conv2d_2_out_channels=512,
                                   module0_2_conv2d_4_in_channels=512,
                                   module0_2_conv2d_4_out_channels=1024,
                                   module0_3_conv2d_0_in_channels=1024,
                                   module0_3_conv2d_0_out_channels=512,
                                   module0_3_conv2d_2_in_channels=512,
                                   module0_3_conv2d_2_out_channels=512,
                                   module0_3_conv2d_4_in_channels=512,
                                   module0_3_conv2d_4_out_channels=1024)
        self.module9_2 = Module9(module0_0_conv2d_0_in_channels=1024,
                                 module0_0_conv2d_0_out_channels=512,
                                 module0_0_conv2d_2_in_channels=512,
                                 module0_0_conv2d_2_out_channels=512,
                                 module0_0_conv2d_4_in_channels=512,
                                 module0_0_conv2d_4_out_channels=1024,
                                 module0_1_conv2d_0_in_channels=1024,
                                 module0_1_conv2d_0_out_channels=512,
                                 module0_1_conv2d_2_in_channels=512,
                                 module0_1_conv2d_2_out_channels=512,
                                 module0_1_conv2d_4_in_channels=512,
                                 module0_1_conv2d_4_out_channels=1024)
        self.module8_3 = Module8(conv2d_0_in_channels=1024,
                                 conv2d_0_out_channels=2048,
                                 module3_0_conv2d_0_in_channels=1024,
                                 module3_0_conv2d_0_out_channels=1024,
                                 module3_0_conv2d_0_kernel_size=(1, 1),
                                 module3_0_conv2d_0_stride=(1, 1),
                                 module3_0_conv2d_0_padding=0,
                                 module3_0_conv2d_0_pad_mode="valid",
                                 module3_1_conv2d_0_in_channels=1024,
                                 module3_1_conv2d_0_out_channels=1024,
                                 module3_1_conv2d_0_kernel_size=(3, 3),
                                 module3_1_conv2d_0_stride=(2, 2),
                                 module3_1_conv2d_0_padding=(1, 1, 1, 1),
                                 module3_1_conv2d_0_pad_mode="pad")
        self.conv2d_217 = nn.Conv2d(in_channels=1024,
                                    out_channels=2048,
                                    kernel_size=(1, 1),
                                    stride=(2, 2),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.relu_223 = nn.ReLU()
        self.module9_3 = Module9(module0_0_conv2d_0_in_channels=2048,
                                 module0_0_conv2d_0_out_channels=1024,
                                 module0_0_conv2d_2_in_channels=1024,
                                 module0_0_conv2d_2_out_channels=1024,
                                 module0_0_conv2d_4_in_channels=1024,
                                 module0_0_conv2d_4_out_channels=2048,
                                 module0_1_conv2d_0_in_channels=2048,
                                 module0_1_conv2d_0_out_channels=1024,
                                 module0_1_conv2d_2_in_channels=1024,
                                 module0_1_conv2d_2_out_channels=1024,
                                 module0_1_conv2d_4_in_channels=1024,
                                 module0_1_conv2d_4_out_channels=2048)
        self.avgpool2d_238 = nn.AvgPool2d(kernel_size=(7, 7))
        self.flatten_239 = nn.Flatten()
        self.dense_240 = nn.Dense(in_channels=2048, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_relu_1 = self.relu_1(opt_conv2d_0)
        opt_maxpool2d_2 = self.pad_maxpool2d_2(opt_relu_1)
        opt_maxpool2d_2 = self.maxpool2d_2(opt_maxpool2d_2)
        module8_0_opt = self.module8_0(opt_maxpool2d_2)
        opt_conv2d_4 = self.conv2d_4(opt_maxpool2d_2)
        opt_add_9 = P.Add()(module8_0_opt, opt_conv2d_4)
        opt_relu_10 = self.relu_10(opt_add_9)
        module9_0_opt = self.module9_0(opt_relu_10)
        module8_1_opt = self.module8_1(module9_0_opt)
        opt_conv2d_26 = self.conv2d_26(module9_0_opt)
        opt_add_31 = P.Add()(module8_1_opt, opt_conv2d_26)
        opt_relu_32 = self.relu_32(opt_add_31)
        module9_1_opt = self.module9_1(opt_relu_32)
        module0_0_opt = self.module0_0(module9_1_opt)
        module8_2_opt = self.module8_2(module0_0_opt)
        opt_conv2d_55 = self.conv2d_55(module0_0_opt)
        opt_add_60 = P.Add()(module8_2_opt, opt_conv2d_55)
        opt_relu_61 = self.relu_61(opt_add_60)
        module13_0_opt = self.module13_0(opt_relu_61)
        module13_1_opt = self.module13_1(module13_0_opt)
        module12_0_opt = self.module12_0(module13_1_opt)
        module9_2_opt = self.module9_2(module12_0_opt)
        module8_3_opt = self.module8_3(module9_2_opt)
        opt_conv2d_217 = self.conv2d_217(module9_2_opt)
        opt_add_222 = P.Add()(module8_3_opt, opt_conv2d_217)
        opt_relu_223 = self.relu_223(opt_add_222)
        module9_3_opt = self.module9_3(opt_relu_223)
        opt_avgpool2d_238 = self.avgpool2d_238(module9_3_opt)
        opt_flatten_239 = self.flatten_239(opt_avgpool2d_238)
        opt_dense_240 = self.dense_240(opt_flatten_239)
        return opt_dense_240
