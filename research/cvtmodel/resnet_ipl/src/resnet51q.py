import mindspore.ops as P
from mindspore import nn


class Module0(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_0_kernel_size, conv2d_0_padding,
                 conv2d_0_pad_mode, conv2d_3_in_channels, conv2d_3_out_channels, conv2d_3_stride, conv2d_3_group):
        super(Module0, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=conv2d_0_kernel_size,
                                  stride=(1, 1),
                                  padding=conv2d_0_padding,
                                  pad_mode=conv2d_0_pad_mode,
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.sigmoid_1 = nn.Sigmoid()
        self.conv2d_3 = nn.Conv2d(in_channels=conv2d_3_in_channels,
                                  out_channels=conv2d_3_out_channels,
                                  kernel_size=(3, 3),
                                  stride=conv2d_3_stride,
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=conv2d_3_group,
                                  has_bias=True)
        self.sigmoid_4 = nn.Sigmoid()

    def construct(self, x):
        opt_conv2d_0 = self.conv2d_0(x)
        opt_sigmoid_1 = self.sigmoid_1(opt_conv2d_0)
        opt_mul_2 = P.Mul()(opt_conv2d_0, opt_sigmoid_1)
        opt_conv2d_3 = self.conv2d_3(opt_mul_2)
        opt_sigmoid_4 = self.sigmoid_4(opt_conv2d_3)
        opt_mul_5 = P.Mul()(opt_conv2d_3, opt_sigmoid_4)
        return opt_mul_5


class Module14(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, module0_0_conv2d_0_in_channels,
                 module0_0_conv2d_0_out_channels, module0_0_conv2d_0_kernel_size, module0_0_conv2d_0_padding,
                 module0_0_conv2d_0_pad_mode, module0_0_conv2d_3_in_channels, module0_0_conv2d_3_out_channels,
                 module0_0_conv2d_3_stride, module0_0_conv2d_3_group):
        super(Module14, self).__init__()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module0_0_conv2d_0_kernel_size,
                                 conv2d_0_padding=module0_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module0_0_conv2d_0_pad_mode,
                                 conv2d_3_in_channels=module0_0_conv2d_3_in_channels,
                                 conv2d_3_out_channels=module0_0_conv2d_3_out_channels,
                                 conv2d_3_stride=module0_0_conv2d_3_stride,
                                 conv2d_3_group=module0_0_conv2d_3_group)
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
        module0_0_opt = self.module0_0(x)
        opt_conv2d_0 = self.conv2d_0(module0_0_opt)
        return opt_conv2d_0


class Module4(nn.Cell):
    def __init__(self):
        super(Module4, self).__init__()
        self.sigmoid_0 = nn.Sigmoid()

    def construct(self, x):
        opt_sigmoid_0 = self.sigmoid_0(x)
        opt_mul_1 = P.Mul()(x, opt_sigmoid_0)
        return opt_mul_1


class Module12(nn.Cell):
    def __init__(self):
        super(Module12, self).__init__()
        self.module4_0 = Module4()
        self.module0_0 = Module0(conv2d_0_in_channels=256,
                                 conv2d_0_out_channels=64,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid",
                                 conv2d_3_in_channels=64,
                                 conv2d_3_out_channels=64,
                                 conv2d_3_stride=(1, 1),
                                 conv2d_3_group=2)
        self.conv2d_0 = nn.Conv2d(in_channels=64,
                                  out_channels=256,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.module4_1 = Module4()

    def construct(self, x):
        module4_0_opt = self.module4_0(x)
        module0_0_opt = self.module0_0(module4_0_opt)
        opt_conv2d_0 = self.conv2d_0(module0_0_opt)
        opt_add_1 = P.Add()(opt_conv2d_0, module4_0_opt)
        module4_1_opt = self.module4_1(opt_add_1)
        return module4_1_opt


class Module16(nn.Cell):
    def __init__(self, conv2d_0_in_channels, conv2d_0_out_channels, conv2d_2_in_channels, conv2d_2_out_channels,
                 conv2d_4_in_channels, conv2d_4_out_channels, module0_0_conv2d_0_in_channels,
                 module0_0_conv2d_0_out_channels, module0_0_conv2d_0_kernel_size, module0_0_conv2d_0_padding,
                 module0_0_conv2d_0_pad_mode, module0_0_conv2d_3_in_channels, module0_0_conv2d_3_out_channels,
                 module0_0_conv2d_3_stride, module0_0_conv2d_3_group, module0_1_conv2d_0_in_channels,
                 module0_1_conv2d_0_out_channels, module0_1_conv2d_0_kernel_size, module0_1_conv2d_0_padding,
                 module0_1_conv2d_0_pad_mode, module0_1_conv2d_3_in_channels, module0_1_conv2d_3_out_channels,
                 module0_1_conv2d_3_stride, module0_1_conv2d_3_group, module0_2_conv2d_0_in_channels,
                 module0_2_conv2d_0_out_channels, module0_2_conv2d_0_kernel_size, module0_2_conv2d_0_padding,
                 module0_2_conv2d_0_pad_mode, module0_2_conv2d_3_in_channels, module0_2_conv2d_3_out_channels,
                 module0_2_conv2d_3_stride, module0_2_conv2d_3_group):
        super(Module16, self).__init__()
        self.module4_0 = Module4()
        self.module0_0 = Module0(conv2d_0_in_channels=module0_0_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_0_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module0_0_conv2d_0_kernel_size,
                                 conv2d_0_padding=module0_0_conv2d_0_padding,
                                 conv2d_0_pad_mode=module0_0_conv2d_0_pad_mode,
                                 conv2d_3_in_channels=module0_0_conv2d_3_in_channels,
                                 conv2d_3_out_channels=module0_0_conv2d_3_out_channels,
                                 conv2d_3_stride=module0_0_conv2d_3_stride,
                                 conv2d_3_group=module0_0_conv2d_3_group)
        self.conv2d_0 = nn.Conv2d(in_channels=conv2d_0_in_channels,
                                  out_channels=conv2d_0_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.module4_1 = Module4()
        self.module0_1 = Module0(conv2d_0_in_channels=module0_1_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_1_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module0_1_conv2d_0_kernel_size,
                                 conv2d_0_padding=module0_1_conv2d_0_padding,
                                 conv2d_0_pad_mode=module0_1_conv2d_0_pad_mode,
                                 conv2d_3_in_channels=module0_1_conv2d_3_in_channels,
                                 conv2d_3_out_channels=module0_1_conv2d_3_out_channels,
                                 conv2d_3_stride=module0_1_conv2d_3_stride,
                                 conv2d_3_group=module0_1_conv2d_3_group)
        self.conv2d_2 = nn.Conv2d(in_channels=conv2d_2_in_channels,
                                  out_channels=conv2d_2_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.module4_2 = Module4()
        self.module0_2 = Module0(conv2d_0_in_channels=module0_2_conv2d_0_in_channels,
                                 conv2d_0_out_channels=module0_2_conv2d_0_out_channels,
                                 conv2d_0_kernel_size=module0_2_conv2d_0_kernel_size,
                                 conv2d_0_padding=module0_2_conv2d_0_padding,
                                 conv2d_0_pad_mode=module0_2_conv2d_0_pad_mode,
                                 conv2d_3_in_channels=module0_2_conv2d_3_in_channels,
                                 conv2d_3_out_channels=module0_2_conv2d_3_out_channels,
                                 conv2d_3_stride=module0_2_conv2d_3_stride,
                                 conv2d_3_group=module0_2_conv2d_3_group)
        self.conv2d_4 = nn.Conv2d(in_channels=conv2d_4_in_channels,
                                  out_channels=conv2d_4_out_channels,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.module4_3 = Module4()

    def construct(self, x):
        module4_0_opt = self.module4_0(x)
        module0_0_opt = self.module0_0(module4_0_opt)
        opt_conv2d_0 = self.conv2d_0(module0_0_opt)
        opt_add_1 = P.Add()(opt_conv2d_0, module4_0_opt)
        module4_1_opt = self.module4_1(opt_add_1)
        module0_1_opt = self.module0_1(module4_1_opt)
        opt_conv2d_2 = self.conv2d_2(module0_1_opt)
        opt_add_3 = P.Add()(opt_conv2d_2, module4_1_opt)
        module4_2_opt = self.module4_2(opt_add_3)
        module0_2_opt = self.module0_2(module4_2_opt)
        opt_conv2d_4 = self.conv2d_4(module0_2_opt)
        opt_add_5 = P.Add()(opt_conv2d_4, module4_2_opt)
        module4_3_opt = self.module4_3(opt_add_5)
        return module4_3_opt


class Module10(nn.Cell):
    def __init__(self):
        super(Module10, self).__init__()
        self.module4_0 = Module4()
        self.module0_0 = Module0(conv2d_0_in_channels=1536,
                                 conv2d_0_out_channels=384,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid",
                                 conv2d_3_in_channels=384,
                                 conv2d_3_out_channels=384,
                                 conv2d_3_stride=(1, 1),
                                 conv2d_3_group=12)
        self.conv2d_0 = nn.Conv2d(in_channels=384,
                                  out_channels=1536,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.module4_1 = Module4()
        self.module0_1 = Module0(conv2d_0_in_channels=1536,
                                 conv2d_0_out_channels=384,
                                 conv2d_0_kernel_size=(1, 1),
                                 conv2d_0_padding=0,
                                 conv2d_0_pad_mode="valid",
                                 conv2d_3_in_channels=384,
                                 conv2d_3_out_channels=384,
                                 conv2d_3_stride=(1, 1),
                                 conv2d_3_group=12)
        self.conv2d_2 = nn.Conv2d(in_channels=384,
                                  out_channels=1536,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)

    def construct(self, x):
        module4_0_opt = self.module4_0(x)
        module0_0_opt = self.module0_0(module4_0_opt)
        opt_conv2d_0 = self.conv2d_0(module0_0_opt)
        opt_add_1 = P.Add()(opt_conv2d_0, module4_0_opt)
        module4_1_opt = self.module4_1(opt_add_1)
        module0_1_opt = self.module0_1(module4_1_opt)
        opt_conv2d_2 = self.conv2d_2(module0_1_opt)
        opt_add_3 = P.Add()(opt_conv2d_2, module4_1_opt)
        return opt_add_3


class MindSporeModel(nn.Cell):
    def __init__(self):
        super(MindSporeModel, self).__init__()
        self.conv2d_0 = nn.Conv2d(in_channels=3,
                                  out_channels=16,
                                  kernel_size=(3, 3),
                                  stride=(2, 2),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=False)
        self.conv2d_1 = nn.Conv2d(in_channels=16,
                                  out_channels=32,
                                  kernel_size=(3, 3),
                                  stride=(1, 1),
                                  padding=(1, 1, 1, 1),
                                  pad_mode="pad",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=False)
        self.module0_0 = Module0(conv2d_0_in_channels=32,
                                 conv2d_0_out_channels=64,
                                 conv2d_0_kernel_size=(3, 3),
                                 conv2d_0_padding=(1, 1, 1, 1),
                                 conv2d_0_pad_mode="pad",
                                 conv2d_3_in_channels=64,
                                 conv2d_3_out_channels=128,
                                 conv2d_3_stride=(2, 2),
                                 conv2d_3_group=1)
        self.module14_0 = Module14(conv2d_0_in_channels=64,
                                   conv2d_0_out_channels=256,
                                   module0_0_conv2d_0_in_channels=128,
                                   module0_0_conv2d_0_out_channels=64,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid",
                                   module0_0_conv2d_3_in_channels=64,
                                   module0_0_conv2d_3_out_channels=64,
                                   module0_0_conv2d_3_stride=(1, 1),
                                   module0_0_conv2d_3_group=2)
        self.conv2d_9 = nn.Conv2d(in_channels=128,
                                  out_channels=256,
                                  kernel_size=(1, 1),
                                  stride=(1, 1),
                                  padding=0,
                                  pad_mode="valid",
                                  dilation=(1, 1),
                                  group=1,
                                  has_bias=True)
        self.module12_0 = Module12()
        self.module14_1 = Module14(conv2d_0_in_channels=128,
                                   conv2d_0_out_channels=512,
                                   module0_0_conv2d_0_in_channels=256,
                                   module0_0_conv2d_0_out_channels=128,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid",
                                   module0_0_conv2d_3_in_channels=128,
                                   module0_0_conv2d_3_out_channels=128,
                                   module0_0_conv2d_3_stride=(2, 2),
                                   module0_0_conv2d_3_group=4)
        self.conv2d_30 = nn.Conv2d(in_channels=256,
                                   out_channels=512,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module16_0 = Module16(conv2d_0_in_channels=128,
                                   conv2d_0_out_channels=512,
                                   conv2d_2_in_channels=128,
                                   conv2d_2_out_channels=512,
                                   conv2d_4_in_channels=128,
                                   conv2d_4_out_channels=512,
                                   module0_0_conv2d_0_in_channels=512,
                                   module0_0_conv2d_0_out_channels=128,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid",
                                   module0_0_conv2d_3_in_channels=128,
                                   module0_0_conv2d_3_out_channels=128,
                                   module0_0_conv2d_3_stride=(1, 1),
                                   module0_0_conv2d_3_group=4,
                                   module0_1_conv2d_0_in_channels=512,
                                   module0_1_conv2d_0_out_channels=128,
                                   module0_1_conv2d_0_kernel_size=(1, 1),
                                   module0_1_conv2d_0_padding=0,
                                   module0_1_conv2d_0_pad_mode="valid",
                                   module0_1_conv2d_3_in_channels=128,
                                   module0_1_conv2d_3_out_channels=128,
                                   module0_1_conv2d_3_stride=(1, 1),
                                   module0_1_conv2d_3_group=4,
                                   module0_2_conv2d_0_in_channels=512,
                                   module0_2_conv2d_0_out_channels=128,
                                   module0_2_conv2d_0_kernel_size=(1, 1),
                                   module0_2_conv2d_0_padding=0,
                                   module0_2_conv2d_0_pad_mode="valid",
                                   module0_2_conv2d_3_in_channels=128,
                                   module0_2_conv2d_3_out_channels=128,
                                   module0_2_conv2d_3_stride=(1, 1),
                                   module0_2_conv2d_3_group=4)
        self.module14_2 = Module14(conv2d_0_in_channels=384,
                                   conv2d_0_out_channels=1536,
                                   module0_0_conv2d_0_in_channels=512,
                                   module0_0_conv2d_0_out_channels=384,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid",
                                   module0_0_conv2d_3_in_channels=384,
                                   module0_0_conv2d_3_out_channels=384,
                                   module0_0_conv2d_3_stride=(2, 2),
                                   module0_0_conv2d_3_group=12)
        self.conv2d_71 = nn.Conv2d(in_channels=512,
                                   out_channels=1536,
                                   kernel_size=(1, 1),
                                   stride=(2, 2),
                                   padding=0,
                                   pad_mode="valid",
                                   dilation=(1, 1),
                                   group=1,
                                   has_bias=True)
        self.module10_0 = Module10()
        self.module16_1 = Module16(conv2d_0_in_channels=384,
                                   conv2d_0_out_channels=1536,
                                   conv2d_2_in_channels=384,
                                   conv2d_2_out_channels=1536,
                                   conv2d_4_in_channels=384,
                                   conv2d_4_out_channels=1536,
                                   module0_0_conv2d_0_in_channels=1536,
                                   module0_0_conv2d_0_out_channels=384,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid",
                                   module0_0_conv2d_3_in_channels=384,
                                   module0_0_conv2d_3_out_channels=384,
                                   module0_0_conv2d_3_stride=(1, 1),
                                   module0_0_conv2d_3_group=12,
                                   module0_1_conv2d_0_in_channels=1536,
                                   module0_1_conv2d_0_out_channels=384,
                                   module0_1_conv2d_0_kernel_size=(1, 1),
                                   module0_1_conv2d_0_padding=0,
                                   module0_1_conv2d_0_pad_mode="valid",
                                   module0_1_conv2d_3_in_channels=384,
                                   module0_1_conv2d_3_out_channels=384,
                                   module0_1_conv2d_3_stride=(1, 1),
                                   module0_1_conv2d_3_group=12,
                                   module0_2_conv2d_0_in_channels=1536,
                                   module0_2_conv2d_0_out_channels=384,
                                   module0_2_conv2d_0_kernel_size=(1, 1),
                                   module0_2_conv2d_0_padding=0,
                                   module0_2_conv2d_0_pad_mode="valid",
                                   module0_2_conv2d_3_in_channels=384,
                                   module0_2_conv2d_3_out_channels=384,
                                   module0_2_conv2d_3_stride=(1, 1),
                                   module0_2_conv2d_3_group=12)
        self.module14_3 = Module14(conv2d_0_in_channels=1536,
                                   conv2d_0_out_channels=1536,
                                   module0_0_conv2d_0_in_channels=1536,
                                   module0_0_conv2d_0_out_channels=1536,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid",
                                   module0_0_conv2d_3_in_channels=1536,
                                   module0_0_conv2d_3_out_channels=1536,
                                   module0_0_conv2d_3_stride=(2, 2),
                                   module0_0_conv2d_3_group=1536)
        self.conv2d_132 = nn.Conv2d(in_channels=1536,
                                    out_channels=1536,
                                    kernel_size=(1, 1),
                                    stride=(2, 2),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module16_2 = Module16(conv2d_0_in_channels=1536,
                                   conv2d_0_out_channels=1536,
                                   conv2d_2_in_channels=1536,
                                   conv2d_2_out_channels=1536,
                                   conv2d_4_in_channels=1536,
                                   conv2d_4_out_channels=1536,
                                   module0_0_conv2d_0_in_channels=1536,
                                   module0_0_conv2d_0_out_channels=1536,
                                   module0_0_conv2d_0_kernel_size=(1, 1),
                                   module0_0_conv2d_0_padding=0,
                                   module0_0_conv2d_0_pad_mode="valid",
                                   module0_0_conv2d_3_in_channels=1536,
                                   module0_0_conv2d_3_out_channels=1536,
                                   module0_0_conv2d_3_stride=(1, 1),
                                   module0_0_conv2d_3_group=1536,
                                   module0_1_conv2d_0_in_channels=1536,
                                   module0_1_conv2d_0_out_channels=1536,
                                   module0_1_conv2d_0_kernel_size=(1, 1),
                                   module0_1_conv2d_0_padding=0,
                                   module0_1_conv2d_0_pad_mode="valid",
                                   module0_1_conv2d_3_in_channels=1536,
                                   module0_1_conv2d_3_out_channels=1536,
                                   module0_1_conv2d_3_stride=(1, 1),
                                   module0_1_conv2d_3_group=1536,
                                   module0_2_conv2d_0_in_channels=1536,
                                   module0_2_conv2d_0_out_channels=1536,
                                   module0_2_conv2d_0_kernel_size=(1, 1),
                                   module0_2_conv2d_0_padding=0,
                                   module0_2_conv2d_0_pad_mode="valid",
                                   module0_2_conv2d_3_in_channels=1536,
                                   module0_2_conv2d_3_out_channels=1536,
                                   module0_2_conv2d_3_stride=(1, 1),
                                   module0_2_conv2d_3_group=1536)
        self.conv2d_172 = nn.Conv2d(in_channels=1536,
                                    out_channels=2048,
                                    kernel_size=(1, 1),
                                    stride=(1, 1),
                                    padding=0,
                                    pad_mode="valid",
                                    dilation=(1, 1),
                                    group=1,
                                    has_bias=True)
        self.module4_0 = Module4()
        self.avgpool2d_175 = nn.AvgPool2d(kernel_size=(8, 8))
        self.flatten_176 = nn.Flatten()
        self.dense_177 = nn.Dense(in_channels=2048, out_channels=1000, has_bias=True)

    def construct(self, input_1):
        opt_conv2d_0 = self.conv2d_0(input_1)
        opt_conv2d_1 = self.conv2d_1(opt_conv2d_0)
        module0_0_opt = self.module0_0(opt_conv2d_1)
        module14_0_opt = self.module14_0(module0_0_opt)
        opt_conv2d_9 = self.conv2d_9(module0_0_opt)
        opt_add_16 = P.Add()(module14_0_opt, opt_conv2d_9)
        module12_0_opt = self.module12_0(opt_add_16)
        module14_1_opt = self.module14_1(module12_0_opt)
        opt_conv2d_30 = self.conv2d_30(module12_0_opt)
        opt_add_37 = P.Add()(module14_1_opt, opt_conv2d_30)
        module16_0_opt = self.module16_0(opt_add_37)
        module14_2_opt = self.module14_2(module16_0_opt)
        opt_conv2d_71 = self.conv2d_71(module16_0_opt)
        opt_add_78 = P.Add()(module14_2_opt, opt_conv2d_71)
        module10_0_opt = self.module10_0(opt_add_78)
        module16_1_opt = self.module16_1(module10_0_opt)
        module14_3_opt = self.module14_3(module16_1_opt)
        opt_conv2d_132 = self.conv2d_132(module16_1_opt)
        opt_add_139 = P.Add()(module14_3_opt, opt_conv2d_132)
        module16_2_opt = self.module16_2(opt_add_139)
        opt_conv2d_172 = self.conv2d_172(module16_2_opt)
        module4_0_opt = self.module4_0(opt_conv2d_172)
        opt_avgpool2d_175 = self.avgpool2d_175(module4_0_opt)
        opt_flatten_176 = self.flatten_176(opt_avgpool2d_175)
        opt_dense_177 = self.dense_177(opt_flatten_176)
        return opt_dense_177
