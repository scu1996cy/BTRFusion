import torch
import torchvision
import torch.nn as nn
from torch.nn import init
import functools
from torch.optim import lr_scheduler
import numpy as np
import torch.nn.functional as nnf
from torch.distributions.normal import Normal

def get_norm_layer(norm_type='instance'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
    elif norm_type == 'switchable':
        norm_layer = SwitchNorm2d
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer
def get_scheduler(optimizer, opt):
    if opt.lr_policy == 'lambda':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    elif opt.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif opt.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=opt.niter, eta_min=0)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler
def update_learning_rate(scheduler, optimizer):
    scheduler.step()
    lr = optimizer.param_groups[0]['lr']
    print('learning rate = %.7f' % lr)
def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)
def init_net(net, init_type='normal', init_gain=0.02):
    init_weights(net, init_type, gain=init_gain)
    return net

def define_G_pix2pix(input_nc, output_nc, ngf, norm='batch', use_dropout=False, init_type='normal', init_gain=0.02):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    # pix2pix <-> unet_256, no_dropout: False, norm: batch
    net = UnetGenerator(input_nc, output_nc, 8, ngf, norm_layer=norm_layer, use_dropout=use_dropout)
   
    return init_net(net, init_type, init_gain)
def define_G_cyclegan(input_nc, output_nc, ngf, norm='instance', use_dropout=False, init_type='normal', init_gain=0.02):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)
    # cyclegan <-> resnet_9blocks, no_dropout: False, norm: instance
    net = ResnetGenerator(input_nc, output_nc, ngf, norm_layer=norm_layer, use_dropout=use_dropout, n_blocks=9)

    return init_net(net, init_type, init_gain)
def define_RegMorph(input_nc, init_type='normal', init_gain=0.02):
    net = None
    net = RegMorph(input_nc)

    return init_net(net, init_type, init_gain)

class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.InstanceNorm2d, use_dropout=False, n_blocks=9, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out

class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_nc, output_nc, num_downs, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet generator
        Parameters:
            input_nc (int)  -- the number of channels in input images
            output_nc (int) -- the number of channels in output images
            num_downs (int) -- the number of downsamplings in UNet. For example, # if |num_downs| == 7,
                                image of size 128x128 will become of size 1x1 # at the bottleneck
            ngf (int)       -- the number of filters in the last conv layer
            norm_layer      -- normalization layer

        We construct the U-Net from the innermost layer to the outermost layer.
        It is a recursive process.
        """
        super(UnetGenerator, self).__init__()
        # construct unet structure
        unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=None, norm_layer=norm_layer, innermost=True)  # add the innermost layer
        for i in range(num_downs - 5):          # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(ngf * 8, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer, use_dropout=use_dropout)
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(ngf * 4, ngf * 8, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf * 2, ngf * 4, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(ngf, ngf * 2, input_nc=None, submodule=unet_block, norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_nc, ngf, input_nc=input_nc, submodule=unet_block, outermost=True, norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        """Standard forward"""
        return self.model(input)
class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_nc, inner_nc, input_nc=None,
                 submodule=None, outermost=False, innermost=False, norm_layer=nn.BatchNorm2d, use_dropout=False):
        """Construct a Unet submodule with skip connections.

        Parameters:
            outer_nc (int) -- the number of filters in the outer conv layer
            inner_nc (int) -- the number of filters in the inner conv layer
            input_nc (int) -- the number of channels in input images/features
            submodule (UnetSkipConnectionBlock) -- previously defined submodules
            outermost (bool)    -- if this module is the outermost module
            innermost (bool)    -- if this module is the innermost module
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
        """
        super(UnetSkipConnectionBlock, self).__init__()
        self.outermost = outermost
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        if input_nc is None:
            input_nc = outer_nc
        downconv = nn.Conv2d(input_nc, inner_nc, kernel_size=4,
                             stride=2, padding=1, bias=use_bias)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_nc)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_nc)

        if outermost:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1)
            down = [downconv]
            up = [uprelu, upconv, nn.Tanh()]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.ConvTranspose2d(inner_nc, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.ConvTranspose2d(inner_nc * 2, outer_nc,
                                        kernel_size=4, stride=2,
                                        padding=1, bias=use_bias)
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            if use_dropout:
                model = down + [submodule] + up + [nn.Dropout(0.5)]
            else:
                model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:   # add skip connections
            return torch.cat([x, self.model(x)], 1)

class AffineHead_block(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.affine_head_1 = nn.Linear(in_channels, in_channels // 2, bias=False)
        self.affine_head_2 = nn.Linear(in_channels // 2, 6, bias=False)  # Changed output size to 6 for 2D
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()
    def forward(self, x_in):
        x = torch.mean(x_in, dim=(2, 3))
        x = self.affine_head_1(x)
        x = self.ReLU(x)
        x = self.affine_head_2(x)
        affine_para = self.Tanh(x)
        self.id = torch.zeros((1, 2, 3)).cuda()  # Changed dimensions to 2D
        self.id[0, 0, 0] = 1
        self.id[0, 1, 1] = 1
        affine_matrix = affine_para.reshape(x_in.size(0), 2, 3) + self.id
        affine_grid = nnf.affine_grid(affine_matrix, x_in.shape, align_corners=True)
        shape = x_in.shape[2:]
        vectors = [torch.arange(0, s) for s in shape]
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = grid.type(torch.FloatTensor)
        grid = grid.cuda()
        affine_grid = affine_grid[..., [1, 0]]  # Updated to 2D grid
        affine_grid = affine_grid.permute(0, 3, 1, 2)
        for i in range(len(shape)):
            affine_grid[:, i, ...] = (affine_grid[:, i, ...] / 2.0 + 0.5) * (shape[i] - 1)
        flow = affine_grid - grid
        return flow
class DeformHead_block(nn.Module):
    def __init__(self, in_channels: int):
        super().__init__()
        self.reg_head = nn.Conv2d(in_channels, 2, kernel_size=3, stride=1, padding=1)
        self.reg_head.weight = nn.Parameter(Normal(0, 1e-5).sample(self.reg_head.weight.shape))
        self.reg_head.bias = nn.Parameter(torch.zeros(self.reg_head.bias.shape))
    def forward(self, x_in):
        x_out = self.reg_head(x_in)
        return x_out
class SpatialTransformer_block(nn.Module):
    def __init__(self, mode='bilinear'):
        super().__init__()
        self.mode = mode
    def forward(self, src, flow):
        shape = flow.shape[2:]
        vectors = [torch.arange(0, s) for s in shape]
        # grids = torch.meshgrid(vectors, indexing='ij')
        grids = torch.meshgrid(vectors)
        grid = torch.stack(grids)
        grid = torch.unsqueeze(grid, 0)
        grid = grid.type(torch.FloatTensor)
        grid = grid.cuda()
        new_locs = grid + flow
        for i in range(len(shape)):
            new_locs[:, i, ...] = 2 * (new_locs[:, i, ...] / (shape[i] - 1) - 0.5)
        new_locs = new_locs.permute(0, 2, 3, 1)
        new_locs = new_locs[..., [1,0]]
        return nnf.grid_sample(src, new_locs, align_corners=True, mode=self.mode)
class ResizeTransformer_block(nn.Module):
    def __init__(self, resize_factor=2, ndims=2):
        super().__init__()
        self.factor = resize_factor
        self.mode = 'linear'
        if ndims == 2:
            self.mode = 'bi' + self.mode
        elif ndims == 3:
            self.mode = 'tri' + self.mode
    def forward(self, x):
        if self.factor < 1:
            # resize first to save memory
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
            x = self.factor * x
        elif self.factor > 1:
            # multiply first to save memory
            x = self.factor * x
            x = nnf.interpolate(x, align_corners=True, scale_factor=self.factor, mode=self.mode)
        return x
class VecInt(nn.Module):
    def __init__(self, nsteps=7):
        super().__init__()
        assert nsteps >= 0, 'nsteps should be >= 0, found: %d' % nsteps
        self.nsteps = nsteps
        self.scale = 1.0 / (2 ** self.nsteps)
        self.transformer = SpatialTransformer_block()
    def forward(self, vec):
        vec = vec * self.scale
        for _ in range(self.nsteps):
            vec = vec + self.transformer(vec, vec)
        return vec

class ConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        x_out = self.act(x)
        return x_out
class DualConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        self.norm1 = nn.InstanceNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)
        self.norm2 = nn.InstanceNorm2d(out_channels)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x_out = self.act(x)
        return x_out
class ConvEncoder(nn.Module):
    def __init__(self, in_channels=3, channel_num=16):
        super().__init__()
        self.conv_1 = ConvNormAct(in_channels, channel_num)
        self.conv_2 = ConvNormAct(channel_num, channel_num * 2)
        self.conv_3 = ConvNormAct(channel_num * 2, channel_num * 4)
        self.conv_4 = ConvNormAct(channel_num * 4, channel_num * 8)
        self.conv_5 = ConvNormAct(channel_num * 8, channel_num * 16)
        self.downsample = nn.AvgPool2d(2, stride=2)
    def forward(self, x_in):
        x_1 = self.conv_1(x_in)
        x = self.downsample(x_1)
        x_2 = self.conv_2(x)
        x = self.downsample(x_2)
        x_3 = self.conv_3(x)
        x = self.downsample(x_3)
        x_4 = self.conv_4(x)
        x = self.downsample(x_4)
        x_5 = self.conv_5(x)
        return [x_1, x_2, x_3, x_4, x_5]
class TransConvNormAct(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=2, stride=2):
        super().__init__()
        self.deconv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride)
        self.norm = nn.InstanceNorm2d(out_channels)
        self.act = nn.GELU()
    def forward(self, x):
        x = self.deconv(x)
        x = self.norm(x)
        x_out = self.act(x)
        return x_out
class RegMorph(nn.Module):
    def __init__(self, input_nc=3, channel_num=16):
        super(RegMorph, self).__init__()

        self.encoder = ConvEncoder(in_channels=input_nc, channel_num=channel_num)

        self.conv_1 = ConvNormAct(channel_num * 1 * 3, channel_num * 1)
        self.conv_2 = ConvNormAct(channel_num * 2 * 3, channel_num * 2)
        self.conv_3 = ConvNormAct(channel_num * 4 * 3, channel_num * 4)
        self.conv_4 = ConvNormAct(channel_num * 8 * 3, channel_num * 8)
        self.conv_5 = ConvNormAct(channel_num * 16 * 2, channel_num * 16)

        self.upsample_1 = TransConvNormAct(channel_num * 2, channel_num * 1)
        self.upsample_2 = TransConvNormAct(channel_num * 4, channel_num * 2)
        self.upsample_3 = TransConvNormAct(channel_num * 8, channel_num * 4)
        self.upsample_4 = TransConvNormAct(channel_num * 16, channel_num * 8)

        self.reghead_1 = DeformHead_block(channel_num*1)
        self.reghead_2 = DeformHead_block(channel_num*2)
        self.reghead_3 = DeformHead_block(channel_num*4)
        self.reghead_4 = DeformHead_block(channel_num*8)
        self.reghead_5 = AffineHead_block(channel_num*16)

        self.resize_transformer = nn.ModuleList()
        self.spatial_transformer = nn.ModuleList()
        self.integrate = nn.ModuleList()
        for i in range(4):
            self.resize_transformer.append(ResizeTransformer_block())
            self.spatial_transformer.append(SpatialTransformer_block())
            self.integrate.append(VecInt())

    def forward(self, moving, fixed):

        mov_1, mov_2, mov_3, mov_4, mov_5 = self.encoder(moving)
        fix_1, fix_2, fix_3, fix_4, fix_5 = self.encoder(fixed)

        # Step 1
        cat = torch.cat([fix_5, mov_5], dim=1)
        dconv_5 = self.conv_5(cat)
        flow_5 = self.reghead_5(dconv_5)

        # Step 2
        flow_5_up = self.resize_transformer[3](flow_5)
        mov_4 = self.spatial_transformer[3](mov_4, flow_5_up)

        dconv_5_up = self.upsample_4(dconv_5)
        cat = torch.cat([fix_4, dconv_5_up, mov_4], dim=1)
        dconv_4 = self.conv_4(cat)
        flow_4 = self.reghead_4(dconv_4)
        flow_4 = self.spatial_transformer[3](flow_5_up, flow_4)+flow_4

        # Step 3
        flow_4_up = self.resize_transformer[2](flow_4)
        mov_3 = self.spatial_transformer[2](mov_3, flow_4_up)

        dconv_4_up = self.upsample_3(dconv_4)
        cat = torch.cat([fix_3, dconv_4_up, mov_3], dim=1)
        dconv_3 = self.conv_3(cat)
        flow_3 = self.reghead_3(dconv_3)
        flow_3 = self.spatial_transformer[2](flow_4_up, flow_3)+flow_3

        # Step 4
        flow_3_up = self.resize_transformer[1](flow_3)
        mov_2 = self.spatial_transformer[1](mov_2, flow_3_up)

        dconv_3_up = self.upsample_2(dconv_3)
        cat = torch.cat([fix_2, dconv_3_up, mov_2], dim=1)
        dconv_2 = self.conv_2(cat)
        flow_2 = self.reghead_2(dconv_2)
        flow_2 = self.spatial_transformer[1](flow_3_up, flow_2)+flow_2

        # Step 5
        flow_2_up = self.resize_transformer[0](flow_2)
        mov_1 = self.spatial_transformer[0](mov_1, flow_2_up)

        dconv_1_up = self.upsample_1(dconv_2)
        cat = torch.cat([fix_1, dconv_1_up, mov_1], dim=1)
        dconv_1 = self.conv_1(cat)
        flow_1 = self.reghead_1(dconv_1)
        flow_1 = self.spatial_transformer[0](flow_2_up, flow_1)+flow_1

        moved = self.spatial_transformer[0](moving, flow_1)

        return moved, flow_1

def define_D(input_nc, ndf, netD, n_layers_D=3, norm='batch', use_sigmoid=False, init_type='normal', init_gain=0.02):
    net = None
    norm_layer = get_norm_layer(norm_type=norm)

    if netD == 'basic':
        net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'n_layers':
        net = NLayerDiscriminator(input_nc, ndf, n_layers_D, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    elif netD == 'pixel':
        net = PixelDiscriminator(input_nc, ndf, norm_layer=norm_layer, use_sigmoid=use_sigmoid)
    else:
        raise NotImplementedError('Discriminator model name [%s] is not recognized' % net)

    return init_net(net, init_type, init_gain)

# Defines the PatchGAN discriminator with the specified arguments.
class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        kw = 4
        padw = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                          kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult,
                      kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)
class PixelDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, norm_layer=nn.BatchNorm2d, use_sigmoid=False):
        super(PixelDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        self.net = [
            nn.Conv2d(input_nc, ndf, kernel_size=1, stride=1, padding=0),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf, ndf * 2, kernel_size=1, stride=1, padding=0, bias=use_bias),
            norm_layer(ndf * 2),
            nn.LeakyReLU(0.2, True),
            nn.Conv2d(ndf * 2, 1, kernel_size=1, stride=1, padding=0, bias=use_bias)]

        if use_sigmoid:
            self.net.append(nn.Sigmoid())

        self.net = nn.Sequential(*self.net)

    def forward(self, input):
        return self.net(input)

class GANLoss(nn.Module):
    def __init__(self, use_lsgan=True, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        if use_lsgan:
            self.loss = nn.MSELoss()
        else:
            self.loss = nn.BCELoss()

    def get_target_tensor(self, input, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(input)

    def __call__(self, input, target_is_real):
        target_tensor = self.get_target_tensor(input, target_is_real)
        return self.loss(input, target_tensor)
class GradientLoss(nn.Module):
    def __init__(self):
        super(GradientLoss, self).__init__()

    def forward(self, s, penalty='l2'):
        dy = torch.abs(s[:, :, 1:, :] - s[:, :, :-1, :])
        dx = torch.abs(s[:, :, :, 1:] - s[:, :, :, :-1])
        if (penalty == 'l2'):
            dy = dy * dy
            dx = dx * dx
        d = torch.mean(dx) + torch.mean(dy)
        return d / 2.0
