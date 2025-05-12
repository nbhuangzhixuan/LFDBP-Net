# from mamba_ssm.modules.mamba_simple import Mamba
import torch
from matplotlib import pyplot as plt
from torch import nn
import torch.nn.functional as F
import functools
from models.NAF_block.models.archs.arch_util import LayerNorm2d
from einops import rearrange

# from mamba_ssm.modules.mamba_simple import Mamba
from models.wtconv.wtconv2d import Fusion
import numbers

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight


class WithBias_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(WithBias_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        mu = x.mean(-1, keepdim=True)
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return (x - mu) / torch.sqrt(sigma + 1e-5) * self.weight + self.bias


class LayerNorm_mamba(nn.Module):
    def __init__(self, dim, LayerNorm_type):
        super(LayerNorm_mamba, self).__init__()
        if LayerNorm_type =='BiasFree':
            self.body = BiasFree_LayerNorm(dim)
        else:
            self.body = WithBias_LayerNorm(dim)

    def forward(self, x):
        if len(x.shape)==4:
            h, w = x.shape[-2:]
            return to_4d(self.body(to_3d(x)), h, w)
        else:
            return self.body(x)
def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

class PatchUnEmbed(nn.Module):
    def __init__(self, basefilter) -> None:
        super().__init__()
        self.nc = basefilter

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """

    def __init__(self, patch_size=4, stride=4, in_chans=36, embed_dim=32 * 32 * 32, norm_layer=None, flatten=True):
        super().__init__()
        # patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.flatten = flatten

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = LayerNorm_mamba(embed_dim, 'BiasFree')

    def forward(self, x):
        # （b,c,h,w)->(b,c*s*p,h//s,w//s)
        # (b,h*w//s**2,c*s**2)
        B, C, H, W = x.shape
        # x = F.unfold(x, self.patch_size, stride=self.patch_size)
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        # x = self.norm(x)
        return x


def show(feature, feature_names=None):
    # 将每个通道的特征图转为 numpy 数组
    feature_map_data_list = [feature[0, i].detach().cpu().numpy() for i in range(feature.shape[1])]

    # 可视化每个特征图的热力图
    plt.figure(figsize=(30, 30))
    for i, feature_map_data in enumerate(feature_map_data_list):
        if i == 1:  # 只显示前 5 个特征图
            break

        # 计算每个特征图的最小值和最大值
        feature_min = feature_map_data.min()
        feature_max = feature_map_data.max()

        plt.subplot(1, 1, i + 1)
        # 使用 vmin 和 vmax 来显示特征图的最小最大范围
        plt.imshow(feature_map_data, cmap="jet", vmin=feature_min, vmax=feature_max)

        # 显示特征图的标题
        if feature_names is not None and i < len(feature_names):
            plt.title(f"{feature_names[i]}")
        else:
            plt.title(f"Feature Map {i + 1}")

        plt.axis('off')  # 关闭坐标轴显示

    plt.show()




class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    def __init__(self, c, DW_Expand=1, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1,
                               groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        # Simplified Channel Attention
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGate
        self.sg = SimpleGate()

        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1,
                               bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1,
                               groups=1, bias=True)

        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        x = self.norm1(x)

        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)
        x = x * self.sca(x)
        x = self.conv3(x)

        x = self.dropout1(x)

        y = inp + x * self.beta

        x = self.conv4(self.norm2(y))
        x = self.sg(x)
        x = self.conv5(x)

        x = self.dropout2(x)

        return y + x * self.gamma






class Encoder(nn.Module):
    def __init__(self, img_channel=3, width=24, enc_blk_nums=[1,1,1]):
        super(Encoder, self).__init__()

        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1,
                               groups=1,
                               bias=True)

        self.encoders = nn.ModuleList()
        self.downs = nn.ModuleList()

        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(width) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(width, width*2, 2, 2)
            )
            width = width*2

    def forward(self, inp):

        x = self.intro(inp)

        encs = []

        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        return x,encs

class Bottleneck_i(nn.Module):
    def __init__(self, width=96, middle_blk_num=4):
        super(Bottleneck_i, self).__init__()
        self.middle_blks_1 = nn.ModuleList()
        self.down = nn.Conv2d(width*2, width, kernel_size=1)
        self.middle_blks_2 = nn.ModuleList()
        for _ in range(middle_blk_num):
            self.middle_blks_1.append(NAFBlock(width*2))
            self.middle_blks_2.append(NAFBlock(width))

    def forward(self, x,y):
        fusion = torch.cat([x,y], dim=1)
        for blk in self.middle_blks_1:
            fusion = blk(fusion)
        fusion = self.down(fusion)
        for blk in self.middle_blks_2:
            fusion = blk(fusion)

        return x

class Bottleneck(nn.Module):
    def __init__(self, width=96, middle_blk_num=12):
        super(Bottleneck, self).__init__()
        self.middle_blks = nn.ModuleList()
        for _ in range(middle_blk_num):
            self.middle_blks.append(NAFBlock(width))

    def forward(self, x):
        for blk in self.middle_blks:
            x = blk(x)

        return x

class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class Decoder_nor_mff(nn.Module):
    # def __init__(self, img_channel=3, width=16, middle_blk_num=6, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
    def __init__(self, img_channel=3, chan=96, width=12, dec_blk_nums=[1,1,1]):
        super(Decoder_nor_mff, self).__init__()

        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)
        self.ending_second = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)


        self.decoders = nn.ModuleList()

        self.ups = nn.ModuleList()
        self.decoders_second = nn.ModuleList()

        self.ups_second = nn.ModuleList()




        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            self.ups_second.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )

            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.decoders_second.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )


        self.mff_forward = MFF_forward(width=width)
        self.mff_backward = MFF_backward(width=width)



    def forward(self, x,enc_skips):
        masks = []
        first_decoder = []
        #first
        mff_fa1,mff_fa2,mff_fa3,down1s,down2s,down3s = self.mff_forward(enc_skips)
        enc_skips_first = [mff_fa1,mff_fa2,mff_fa3]



        x_first = x
        x_second = x
        for decoder, up, enc_skip_first in zip(self.decoders, self.ups, enc_skips_first[::-1]):
            x_first = up(x_first)
            x_first = x_first + enc_skip_first
            x_first = decoder(x_first)
            mask = enc_skip_first - x_first
            # show(mask)
            first_decoder.append(x_first)
            masks.append(mask.detach())

        out_first = self.ending(x_first)




        #second
        mff_fa1_second,mff_fa2_second,mff_fa3_second = self.mff_backward(first_decoder[::-1],masks[::-1],down1s,down2s,down3s)
        enc_skips_second = [mff_fa1_second+mff_fa1.detach(),mff_fa2_second+mff_fa2.detach(),mff_fa3_second+mff_fa3.detach()]


        # for a,b in zip(enc_skips_second, enc_skips_first):
        #     a_heat = torch.mean(a,1,keepdim=True)-torch.mean(b,1,keepdim=True)
        #     show(a_heat)

        for decoder_second, up_second, enc_skip_second in zip(self.decoders_second, self.ups_second, enc_skips_second[::-1]):
            x_second = up_second(x_second)
            x_second = x_second + enc_skip_second
            x_second = decoder_second(x_second)
        out_second = self.ending_second(x_second)



        return out_first,out_second





class Decoder_nor_sigmoid(nn.Module):
    # def __init__(self, img_channel=3, width=16, middle_blk_num=6, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
    def __init__(self, img_channel=3, chan=96, width=24, dec_blk_nums=[1,1,1]):
        super(Decoder_nor_sigmoid, self).__init__()

        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x,enc_skips):


        for decoder, up, enc_skip in zip(self.decoders, self.ups, enc_skips[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        out = self.sigmoid(self.ending(x))

        return out

class Decoder_nor(nn.Module):
    # def __init__(self, img_channel=3, width=16, middle_blk_num=6, enc_blk_nums=[2, 2, 4, 8], dec_blk_nums=[2, 2, 2, 2]):
    def __init__(self, img_channel=3, chan=96, width=24, dec_blk_nums=[1,1,1]):
        super(Decoder_nor, self).__init__()

        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1,
                                groups=1,
                                bias=True)
        self.decoders = nn.ModuleList()
        self.ups = nn.ModuleList()

        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

    def forward(self, x,enc_skips):


        for decoder, up, enc_skip in zip(self.decoders, self.ups, enc_skips[::-1]):
            x = up(x)
            x = x + enc_skip
            x = decoder(x)
        out = self.ending(x)

        return out




class MFF_block_add(nn.Module):
    def __init__(self, width=12,main=1):
        super(MFF_block_add, self).__init__()
        self.main = main
        if self.main==1:
            self.up1 = nn.Sequential(
                nn.ConvTranspose2d(width*2, width, 2,2,bias=False),
            )
            self.up2 = nn.Sequential(
                nn.ConvTranspose2d(width*4, width*2, 2,2,bias=False),
                nn.ConvTranspose2d(width * 2, width * 1, 2, 2, bias=False),
            )
            self.iden = nn.Conv2d(width, width, 1, 1, bias=False)

        if self.main==2:
            self.up1 = nn.Conv2d(width, 2 * width, 2, 2)
            self.up2 = nn.Sequential(
                nn.ConvTranspose2d(width*4, width*2, 2,2,bias=False),
            )
            self.iden = nn.Conv2d(width * 2, width * 2, 1, 1, bias=False)
        if self.main==3:
            self.up1 = nn.Sequential(nn.Conv2d(width, 2 * width, 2, 2)
                                     ,nn.Conv2d(2 *width,4 * width, 2, 2))
            self.up2 = nn.Conv2d(2 *width, 4 * width, 2, 2)
            self.iden = nn.Conv2d(width * 4, width * 4, 1, 1, bias=False)




    def forward(self, x,y,z):
        if self.main==1:
            x = self.iden(x)
            y = self.up1(y)
            z = self.up2(z)
            out = x+y+z
        elif self.main==2:
            x = self.up1(x)
            y= self.iden(y)
            z = self.up2(z)
            out = x+y+z
        elif self.main==3:
            x = self.up1(x)
            y = self.up2(y)
            z= self.iden(z)
            out = x+y+z
        return out,x,y,z


class MFF_block_query(nn.Module):
    def __init__(self, width=24,main=1):
        super(MFF_block_query, self).__init__()
        self.mamba = Fusion(width)

    def forward(self,up,mask,x,y,z):

        out = self.mamba(up,mask,x, y, z)
        return out

class MFF_backward(nn.Module):
    def __init__(self, width=24):
        super(MFF_backward, self).__init__()
        self.mff1 = MFF_block_query(width=width)
        self.mff2 = MFF_block_query(width=width*2)
        self.mff3 = MFF_block_query(width=width*4)


    def forward(self, up,masks,down1s,down2s,down3s):
        up1,up2,up3 = up[0], up[1], up[2]
        mask1, mask2, mask3 = masks[0], masks[1], masks[2]
        donw1_down1, donw1_down2, donw1_down3 = down1s[0], down1s[1], down1s[2]
        donw2_down1, donw2_down2, donw2_down3 = down2s[0], down2s[1], down2s[2]
        donw3_down1, donw3_down2, donw3_down3 = down3s[0], down3s[1], down3s[2]

        mff_fa1 = self.mff1(up1,mask1,donw1_down1,donw1_down2,donw1_down3)
        # show(mff_fa1-down1)
        mff_fa2 = self.mff2(up2,mask2,donw2_down1,donw2_down2,donw2_down3)
        # show(mff_fa2-down2)
        mff_fa3 = self.mff3(up3,mask3,donw3_down1,donw3_down2,donw3_down3)
        # show(mff_fa3-down3)


        return mff_fa1,mff_fa2,mff_fa3

class MFF_forward(nn.Module):
    def __init__(self, width=24):
        super(MFF_forward, self).__init__()
        self.mff1 = MFF_block_add(width=width,main=1)
        self.mff2 = MFF_block_add(width=width,main=2)
        self.mff3 = MFF_block_add(width=width,main=3)


    def forward(self, encs):
        down1,down2,down3 = encs[0], encs[1],encs[2]
        mff_fa1, donw1_down1,donw1_down2,donw1_down3 = self.mff1(down1,down2,down3)
        # show(mff_fa1-down1)
        mff_fa2, donw2_down1,donw2_down2,donw2_down3 = self.mff2(down1,down2,down3)
        # show(mff_fa2-down2)
        mff_fa3,donw3_down1,donw3_down2,donw3_down3 = self.mff3(down1,down2,down3)
        # show(mff_fa3-down3)
        down1s = [ donw1_down1,donw1_down2,donw1_down3 ]
        down2s = [ donw2_down1,donw2_down2,donw2_down3 ]
        down3s = [ donw3_down1,donw3_down2,donw3_down3 ]

        return mff_fa1,mff_fa2,mff_fa3,down1s,down2s,down3s



class DecloudingNetwork(nn.Module):
    def __init__(self,g_j_before,width=12):
        super().__init__()
        self.i_en_g = Encoder(width=width)  # 共享的i_en_g
        self.c_t_before = Bottleneck(width=width*8)
        self.c_t_after = Bottleneck(width=width*8)
        self.c_a_after = Bottleneck(width=width * 8)
        self.g_j_before = g_j_before
        self.g_j_after = Bottleneck(width=width*8)
        self.i_de_j = Decoder_nor_mff(chan=width*8, width=width)
        self.i_de_t = Decoder_nor_sigmoid(chan=width*8, width=width)
        self.i_de_a = Decoder_nor_sigmoid(chan=width * 8, width=width)
        self.upsample = F.upsample_nearest
        self.relu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, cloudy):
        i_fea, i_res_g = self.i_en_g(cloudy)
        i_c = self.c_t_before(i_fea)
        fea_t = self.c_t_after(i_c)
        fea_a = self.c_a_after(i_c)
        i_g = self.g_j_before(i_fea)
        fea_j = self.g_j_after(i_g)

        clear_x,clear_x_after = self.i_de_j(fea_j, i_res_g)
        tran = self.i_de_t(fea_t, i_res_g)
        atp = self.i_de_a(fea_a, i_res_g)


        shape_out1 = atp.data.size()
        shape_out = shape_out1[2:4]
        if shape_out1[2] >= shape_out1[3]:
            atp = F.avg_pool2d(atp, shape_out1[3])
        else:
            atp = F.avg_pool2d(atp, shape_out1[2])
        atp = self.upsample(atp, size=shape_out)



        fake_x_after = (clear_x_after * tran) + atp * (1 - tran)


        return clear_x,tran,i_g, i_c,clear_x_after,fake_x_after,atp

class CloudGeneratingNetwork(nn.Module):
    def __init__(self, g_j_before, width=12):
        super().__init__()
        self.j_en_g = Encoder(width=width)  # 共享的i_en_g
        self.un_j_en_g = Encoder(width=width)  # 共享的i_en_g
        self.g_j_before = g_j_before
        self.g_c_m_j_down = Bottleneck(width=width*8)
        self.i_de_j = Decoder_nor(chan=width*8, width=width)
        self.j_de_i = Decoder_nor(chan=width*8, width=width)
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels=width * 8, out_channels=width * 8, kernel_size=1),
            LayerNorm2d(width * 8),
            nn.GELU(),
            nn.Conv2d(width * 8, width * 8, 1)
        )

    def freeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = False

    def unfreeze_module(self, module):
        for param in module.parameters():
            param.requires_grad = True

    def forward(self, clear_x, un_clean, i_g, i_c):
        j_g, j_res = self.j_en_g(clear_x)
        self.freeze_module(self.g_j_before)
        j_g = self.g_j_before(j_g)
        self.unfreeze_module(self.g_j_before)
        shared_j_g = self.shared_conv(j_g)
        shared_i_g = self.shared_conv(i_g)
        shared_i_c = self.shared_conv(i_c)
        fea_j2_middle = self.g_c_m_j_down(j_g+i_c)
        g_cloudy_x = self.j_de_i(fea_j2_middle, j_res)
        un_j_g, un_j_res = self.un_j_en_g(un_clean)
        un_j_g = self.g_j_before(un_j_g)
        fea_j2_middle_un = self.g_c_m_j_down(un_j_g+i_c)
        g_cloudy_x_un = self.j_de_i(fea_j2_middle_un, un_j_res)
        clean_un = self.i_de_j(un_j_g, un_j_res)


        return g_cloudy_x, g_cloudy_x_un, shared_i_g, shared_i_c, shared_j_g, un_j_g, clean_un

class Test_net(torch.nn.Module):
    def __init__(self,width=12):
        super().__init__()
        self.i_en_g = Encoder(width=width)  # 共享的i_en_g
        self.g_j_before = Bottleneck(width=width * 8)
        self.g_j_after = Bottleneck(width=width * 8)
        self.i_de_j = Decoder_nor_mff(chan=width*8, width=width)

    def forward(self, cloudy):
        i_fea, i_res_g = self.i_en_g(cloudy)
        i_g = self.g_j_before(i_fea)
        fea_j = self.g_j_after(i_g)
        clear_x,_ = self.i_de_j(fea_j, i_res_g)


        return clear_x

if __name__ == '__main__':
    from thop import profile, clever_format

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Test_net().to(device)
    # 获取 netG_A（生成器 A）
    # 创建一个输入张量，假设输入图像大小为 3x512x512
    from ptflops import get_model_complexity_info

    macs, params = get_model_complexity_info(model, (3, 512, 512), as_strings=True,
                                             print_per_layer_stat=True, verbose=True)
    print('{:<30} {:<8}'.format('Flops:', macs))
    print('params:' + str(params))



class Discriminator(nn.Module):
    """
    Discriminator class
    """

    def __init__(self, inp=3, out=1):
        """
        Initializes the PatchGAN model with 3 layers as discriminator

        Args:
        inp: number of input image channels
        out: number of output image channels
        """

        super(Discriminator, self).__init__()

        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)

        model = [
            nn.Conv2d(inp, 64, kernel_size=4, stride=2, padding=1),  # input 3 channels
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
            norm_layer(128),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
            norm_layer(256),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=True),
            norm_layer(512),
            nn.LeakyReLU(0.2, True),

            nn.Conv2d(512, out, kernel_size=4, stride=1, padding=1)  # output only 1 channel (prediction map)
        ]
        self.model = nn.Sequential(*model)

    def forward(self, input):
        """
            Feed forward the image produced by generator through discriminator

            Args:
            input: input image

            Returns:
            outputs prediction map with 1 channel
        """
        result = self.model(input)

        return result




# class Feature_discriminator(nn.Module):
#     def __init__(self, ngf=96):
#         super(Feature_discriminator, self).__init__()
#         norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
#
#         model = [
#             nn.Conv2d(ngf, ngf*2, kernel_size=4, stride=2, padding=1),  # input 3 channels
#             nn.LeakyReLU(0.2, True),
#
#             nn.Conv2d(ngf*2, ngf*4, kernel_size=4, stride=2, padding=1, bias=True),
#             norm_layer(ngf*4),
#             nn.LeakyReLU(0.2, True),
#
#             nn.Conv2d(ngf*4, ngf*8, kernel_size=4, stride=2, padding=1, bias=True),
#             norm_layer(ngf*8),
#             nn.LeakyReLU(0.2, True),
#
#
#             nn.Conv2d(ngf*8, 1, kernel_size=4, stride=1, padding=1)  # output only 1 channel (prediction map)
#         ]
#         self.model = nn.Sequential(*model)
#
#     def forward(self, input):
#         """
#             Feed forward the image produced by generator through discriminator
#
#             Args:
#             input: input image
#
#             Returns:
#             outputs prediction map with 1 channel
#         """
#         result = self.model(input)
#
#         return result

class Feature_discriminator(nn.Module):
    def __init__(self, in_ch=3, image_size=128, d=64):
        super(Feature_discriminator, self).__init__()
        self.feature_map_size = image_size // 32
        self.d = d

        self.features = nn.Sequential(
            nn.Conv2d(in_ch, d, kernel_size=3, stride=1, padding=1),  # input is 3 x 128 x 128
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d, d, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 64 x 64 x 64
            LayerNorm2d(d),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d, d * 2, kernel_size=3, stride=1, padding=1, bias=False),
            LayerNorm2d(d * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 2, d * 2, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 128 x 32 x 32
            LayerNorm2d(d * 2),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 2, d * 4, kernel_size=3, stride=1, padding=1, bias=False),
            LayerNorm2d(d * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 4, d * 4, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 256 x 16 x 16
            LayerNorm2d(d * 4),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 4, d * 8, kernel_size=3, stride=1, padding=1, bias=False),
            LayerNorm2d(d * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 8, d * 8, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 512 x 8 x 8
            LayerNorm2d(d * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 8, d * 8, kernel_size=3, stride=1, padding=1, bias=False),
            LayerNorm2d(d * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.Conv2d(d * 8, d * 8, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 512 x 4 x 4
            LayerNorm2d(d * 8),
            nn.LeakyReLU(negative_slope=0.2, inplace=True)
        )

        self.classifier = nn.Sequential(
            nn.Linear((self.d * 8) * self.feature_map_size * self.feature_map_size, 100),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Linear(100, 1)
        )

    def forward(self, x):
        out = self.features(x)
        out = torch.flatten(out, 1)
        out = self.classifier(out)

        return out

#
# class Feature_discriminator(nn.Module):
#
#     def __init__(self, in_channels=512, out=1):
#         """
#         Initializes the PatchGAN model with 3 layers as discriminator
#
#         Args:
#         inp: number of input image channels
#         out: number of output image channels
#         """
#
#         super(Feature_discriminator, self).__init__()
#
#         norm_layer = functools.partial(nn.InstanceNorm2d, affine=False, track_running_stats=False)
#
#         model = [
#             nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),  # input 3 channels
#             nn.LeakyReLU(0.2, True),
#
#             nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1, bias=True),
#             norm_layer(128),
#             nn.LeakyReLU(0.2, True),
#
#             nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1, bias=True),
#             norm_layer(256),
#             nn.LeakyReLU(0.2, True),
#
#             nn.Conv2d(256, 512, kernel_size=4, stride=1, padding=1, bias=True),
#             norm_layer(512),
#             nn.LeakyReLU(0.2, True),
#
#             nn.Conv2d(512, out, kernel_size=4, stride=1, padding=1)  # output only 1 channel (prediction map)
#         ]
#         self.model = nn.Sequential(*model)
#
#     def forward(self, input):
#         """
#             Feed forward the image produced by generator through discriminator
#
#             Args:
#             input: input image
#
#             Returns:
#             outputs prediction map with 1 channel
#         """
#         result = self.model(input)
#
#         return result
#
# class Feature_discriminator(nn.Module):
#     def __init__(self, d=128):
#         super(Feature_discriminator, self).__init__()
#
#
#         self.features = nn.Sequential(
#             nn.Conv2d(d, d, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 64 x 64 x 64
#             nn.BatchNorm2d(d),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#
#             nn.Conv2d(d, d * 2, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(d * 2),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#
#             nn.Conv2d(d * 2, d * 2, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 128 x 32 x 32
#             nn.BatchNorm2d(d * 2),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#
#             nn.Conv2d(d * 2, d * 4, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(d * 4),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#
#             nn.Conv2d(d * 4, d * 4, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 256 x 16 x 16
#             nn.BatchNorm2d(d * 4),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#
#             nn.Conv2d(d * 4, d * 8, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(d * 8),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#
#             nn.Conv2d(d * 8, d * 8, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 512 x 8 x 8
#             nn.BatchNorm2d(d * 8),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#
#             nn.Conv2d(d * 8, d * 8, kernel_size=3, stride=1, padding=1, bias=False),
#             nn.BatchNorm2d(d * 8),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#
#             nn.Conv2d(d * 8, d * 8, kernel_size=3, stride=2, padding=1, bias=False),  # state size. 512 x 4 x 4
#             nn.BatchNorm2d(d * 8),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True)
#         )
#
#         self.classifier = nn.Sequential(
#             nn.Linear((self.d * 8) * self.feature_map_size * self.feature_map_size, 100),
#             nn.LeakyReLU(negative_slope=0.2, inplace=True),
#             nn.Linear(100, 1)
#         )
#
#     def forward(self, x):
#         out = self.features(x)
#         out = torch.flatten(out, 1)
#         out = self.classifier(out)
#
#         return out
#
# class Feature_discriminator(nn.Module):
#     def __init__(self, in_channels=512,ngf=96):
#         super(Feature_discriminator, self).__init__()
#         self.net = nn.Sequential(
#             nn.Conv2d(in_channels, ngf, kernel_size=3, padding=1),
#             nn.LeakyReLU(0.2, True),
#             # nn.ReflectionPad2d(1),
#             nn.Conv2d(ngf, ngf, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(ngf),
#             nn.LeakyReLU(0.2, True),
#             # nn.ReflectionPad2d(1),
#             nn.Conv2d(ngf, ngf * 2, kernel_size=3, padding=0),
#             nn.InstanceNorm2d(ngf * 2),
#             nn.LeakyReLU(0.2, True),
#             # nn.ReflectionPad2d(1),
#             nn.Conv2d(ngf * 2, ngf * 2, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(ngf * 2),
#             nn.LeakyReLU(0.2, True),
#
#             # nn.ReflectionPad2d(1),
#             nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, padding=1),
#             nn.InstanceNorm2d(ngf * 4),
#             nn.LeakyReLU(0.2, True),
#
#             # nn.ReflectionPad2d(1),
#             nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, stride=1, padding=1),
#             nn.InstanceNorm2d(ngf * 4),
#             nn.LeakyReLU(0.2, True),
#
#             nn.AdaptiveAvgPool2d(1),
#             nn.Conv2d(ngf * 4, ngf * 8, kernel_size=1),
#             nn.LeakyReLU(0.2, True),
#             nn.Conv2d(ngf * 8, 1, kernel_size=1)
#         )
#
#     def forward(self, x):
#         res = self.net(x)
#         return res

