import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
import torch.nn.functional as F
from basicsr.models.archs.architecture2 import SPADEResnetBlock as SPADEResnetBlock
import math

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

class PositionalEncoding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, noise_level):
        count = self.dim // 2
        step = torch.arange(count, dtype=noise_level.dtype,
                            device=noise_level.device) / count
        encoding = noise_level.unsqueeze(
            1) * torch.exp(-math.log(1e4) * step.unsqueeze(0))
        encoding = torch.cat(
            [torch.sin(encoding), torch.cos(encoding)], dim=-1)
        return encoding
    
## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img
class Up_ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, activation=nn.LeakyReLU(0.2,False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        '''self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size),
            activation)'''
        # norm_layer = 
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size)),
            activation,
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(pw),
            spectral_norm(nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size)),
            activation
        )
        

    def forward(self, x):
        # conv1 = self.conv1(x)
        y = self.conv_block(x)
        return y
    

class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


class HINet(nn.Module):

    def __init__(self, in_chn=3, wf=64, depth=5, relu_slope=0.2, hin_position_left=0, hin_position_right=4, with_noise_level_emb=True):
        super(HINet, self).__init__()


        noise_level_channel = 32
        self.noise_level_mlp = nn.Sequential(
            PositionalEncoding(32),
            nn.Linear(32, 32 * 4),
            Swish(),
            nn.Linear(32 * 4, 32)
        )
            
        self.depth = depth
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.ad1_list = nn.ModuleList()
        # self.ad2_list = nn.ModuleList()

        prev_channels = self.get_input_chn(wf)
        # print("HINet generator normalization", opt.norm_G)
        norm_G = "spectralspadesyncbatch3x3"
        for i in range(depth): #0,1,2,3,4
            use_HIN = True if hin_position_left <= i and i <= hin_position_right else False
            downsample = True if (i+1) < depth else False
            self.down_path_1.append(UNetConvBlock_wT(prev_channels, (2**i) * wf, downsample, relu_slope, use_HIN=use_HIN, noise_level_emb_dim=noise_level_channel))
            # self.down_path_2.append(UNetConvBlock_wT(prev_channels, (2**i) * wf, downsample, relu_slope, use_csff=downsample, use_HIN=use_HIN))
            self.ad1_list.append(SPADEResnetBlock((2**i) * wf, (2**i) * wf, norm_G, label_nc=(2**i) * wf))
            # self.ad2_list.append(SPADEResnetBlock((2**i) * wf, (2**i) * wf, norm_G, label_nc=(2**i) * wf))
            '''if i == 0:
                self.ad1_list.append(SPADEResnetBlock((2**i) * wf, (2**i) * wf, opt, label_nc=(2**i) * wf))
                # self.ad2_list.append(SPADEResnetBlock((2**i) * wf, (2**i) * wf, opt, label_nc=(2**i) * wf))
            else:
                self.ad1_list.append(SPADEResnetBlock((2**(i-1)) * wf, (2**(i-1)) * wf, opt, label_nc=(2**(i-1)) * wf))
                # self.ad2_list.append(SPADEResnetBlock((2**(i-1)) * wf, (2**(i-1)) * wf, opt, label_nc=(2**(i-1)) * wf))'''
            prev_channels = (2**i) * wf

        self.up_path_1 = nn.ModuleList()
        self.ad1_list = self.ad1_list[0:-1]
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope, noise_level_emb_dim=noise_level_channel))
            prev_channels = (2**i)*wf

        self.last = conv3x3(prev_channels, in_chn, bias=True)
        

    def forward(self, x, latent_list, time):
        t = self.noise_level_mlp(time)
        image = x
        #stage 1
        x1 = self.conv_01(image)
        encs = []
        decs = []
        # print("x1", x1.shape)
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:
                x1, x1_up = down(x1, t) # 64, 128, 128 -- 64, 256, 256
                encs.append(x1_up)
            else:
                x1 = down(x1, t) # 2048, 8, 8
    

        for i, up in enumerate(self.up_path_1):
            # temps = self.skip_conv_1[i](encs[-i-1])
            # (8,8) ---- (1024,16,16) --- (16,16)
            temps2 = self.ad1_list[-1-i](encs[-i-1], latent_list[-1-i])
            x1 = up(x1, temps2, t)
            decs.append(x1)
        out = self.last(x1)
        out = out + image
        return out

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


class UNetConvBlock_wT(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False, noise_level_emb_dim=None):
        super(UNetConvBlock_wT, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, 0)
        self.use_csff = use_csff

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)

        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 3, 1, 1)
            self.csff_dec = nn.Conv2d(out_size, out_size, 3, 1, 1)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size//2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)
            
        self.noise_func = FeatureWiseAffine(noise_level_emb_dim, out_size)

    def forward(self, x, time_emb, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.noise_func(out, time_emb)
        out = self.relu_2(self.conv_2(out))

        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope, noise_level_emb_dim=None):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock_wT(in_size, out_size, False, relu_slope, noise_level_emb_dim=noise_level_emb_dim)

    def forward(self, x, bridge, time):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out, time)
        return out

class Subspace(nn.Module):

    def __init__(self, in_size, out_size):
        super(Subspace, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(UNetConvBlock_wT(in_size, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x + sc


class skip_blocks(nn.Module):

    def __init__(self, in_size, out_size, repeat_num=1):
        super(skip_blocks, self).__init__()
        self.blocks = nn.ModuleList()
        self.re_num = repeat_num
        mid_c = 128
        self.blocks.append(UNetConvBlock_wT(in_size, mid_c, False, 0.2))
        for i in range(self.re_num - 2):
            self.blocks.append(UNetConvBlock_wT(mid_c, mid_c, False, 0.2))
        self.blocks.append(UNetConvBlock_wT(mid_c, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for m in self.blocks:
            x = m(x)
        return x + sc


class FeatureWiseAffine(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(FeatureWiseAffine, self).__init__()
        self.noise_func = nn.Sequential(
            nn.Linear(in_channels, out_channels)
        )

    def forward(self, x, noise_embed):
        batch = x.shape[0]
        x = x + self.noise_func(noise_embed).view(batch, -1, 1, 1)
        return x