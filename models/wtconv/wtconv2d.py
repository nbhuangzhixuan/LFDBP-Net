import numbers
import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import torch.nn.functional as F
from matplotlib import pyplot as plt
from models.NAF_block.models.archs.arch_util import LayerNorm2d
from utils.Selective_scan_interface import selective_scan_fn

try:
    from causal_conv1d import causal_conv1d_fn, causal_conv1d_update
except ImportError:
    causal_conv1d_fn, causal_conv1d_update = None, None
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
    def  __init__(self, basefilter) -> None:
        super().__init__()
        self.nc = basefilter

    def forward(self, x, x_size):
        B, HW, C = x.shape
        x = x.transpose(1, 2).view(B, self.nc, x_size[0], x_size[1])  # B Ph*Pw C
        return x
class PatchEmbed_noconv(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        b,c,h,w = x.shape
        x = x.permute(0, 2, 3, 1).reshape(b, -1,c)
        if self.norm is not None:
            x = self.norm(x)
        return x
class PatchEmbed(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=1, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        b,c,h,w = x.shape
        x = self.proj(x).permute(0, 2, 3, 1).reshape(b, -1,c)
        if self.norm is not None:
            x = self.norm(x)
        return x

class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-5, device='cuda'):
        '''
        The improvement of layer normalization
        '''
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device))

    def forward(self, x):
        output = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps) * self.weight
        return output


class _ScaleModule(nn.Module):
    def __init__(self, dims, init_scale=1.0, init_bias=0):
        super(_ScaleModule, self).__init__()
        self.dims = dims
        self.weight = nn.Parameter(torch.ones(*dims) * init_scale)
        self.bias = None

    def forward(self, x):
        return torch.mul(self.weight, x)


class depthwise_separable_conv(nn.Module):
    def __init__(self, nin, nout, kernel_size = 3, padding = 1, bias=False):
        super(depthwise_separable_conv, self).__init__()
        self.depthwise = nn.Conv2d(nin, nin, kernel_size=kernel_size, padding=padding, groups=nin, bias=bias)
        self.pointwise = nn.Conv2d(nin, nout, kernel_size=1, bias=bias)
        self.norm=LayerNorm2d(nout)
        self.act =nn.GELU()


    def forward(self, x):
        out = self.depthwise(x)
        out = self.pointwise(out)
        out = self.act(self.norm(out))
        return out
def show1(feature):
    feature_map_data_list = [feature[0, i].detach().cpu().numpy() for i in range(feature.shape[1])]
    # 可视化每个特征图的热力图
    plt.figure(figsize=(6, 6))
    for i, feature_map_data in enumerate(feature_map_data_list):
        if i == 4:
            break
        plt.subplot(2, 2, i + 1)
        plt.imshow(feature_map_data, cmap="jet")
        plt.title(f"Feature Map {i + 1}")
        plt.axis('off')
    plt.show()
class Fusion(nn.Module):
    def __init__(self, channel):
        super(Fusion, self).__init__()
        self.fusion_x = Fusion_mamba(channel,16)
        self.fusion_y = Fusion_mamba(channel,16)
        self.fusion_z = Fusion_mamba(channel,16)
        # self.info_fusion = nn.Conv2d(channel*2, channel, kernel_size=1, bias=True)
    def forward(self, up,mask,x,y,z):

        # show1(x)
        align_x = self.fusion_x(x,mask,up)
        # show1(align_x)
        # show1(y)
        align_y = self.fusion_y(y,mask,up)
        # show1(align_y)
        # show1(z)
        align_z = self.fusion_z(z,mask,up)
        # show1(align_z-z)
        fusion = align_x+align_y+align_z


        return fusion







class Fusion_mamba(nn.Module):
    def __init__(self, d_model, state_size, expand=2, d_conv=3, bias=False, ):
        super(Fusion_mamba, self).__init__()
        self.d_model = d_model
        self.d_state = state_size
        self.expand = expand
        self.d_conv = d_conv
        self.d_inner = int(self.expand * self.d_model)

        self.patch_size = 1




        self.norm_info = RMSNorm(d_model)
        self.in_proj_info = nn.Linear(d_model, self.d_inner, bias=bias)
        self.fea_embed_info = PatchEmbed(in_chans=d_model,
                                      embed_dim=d_model* self.patch_size * self.patch_size,
                                      patch_size=self.patch_size)


        #define x module

        self.fea_embed_x = PatchEmbed(in_chans=d_model, embed_dim=d_model * self.patch_size * self.patch_size,
                                      patch_size=self.patch_size)
        self.norm_x = RMSNorm(d_model)
        self.in_proj_x = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.final_unembed_x = PatchUnEmbed(self.d_model)

        self.fusion_ssm = SSM_Manipulation_fusion(d_model=self.d_model, state_size=self.d_state, d_inner=self.d_inner,
                                       d_conv=self.d_conv,
                                       conv_bias=True, adjust=False)
        self.out_proj_x = nn.Linear(self.d_inner, self.d_model, bias=bias)

        self.norm_mask = RMSNorm(d_model)
        self.in_proj_mask = nn.Linear(d_model, self.d_inner, bias=bias)
        self.fea_embed_mask = PatchEmbed(in_chans=d_model,
                                         embed_dim=d_model * self.patch_size * self.patch_size,
                                         patch_size=self.patch_size)






    def forward(self,x,mask,info):


        b, c, h, w = x.shape
        pre_x = x
        x = self.fea_embed_x(x)
        x = self.norm_x(x)  # RMSNorm, an improvement for layer normazatliion
        xz = rearrange(
            self.in_proj_x.weight @ rearrange(x, "b l d -> d (b l)"),
            "d (b l) -> b d l",
            l=x.shape[1],
        )

        x, x_act = xz.chunk(2, dim=1)


        info = self.fea_embed_info(info)
        info = self.norm_info(info)  # RMSNorm, an improvement for layer normazatliion
        info = self.in_proj_info(info)
        info = rearrange(info, "b l d -> b d l")

        mask = self.fea_embed_mask(mask)
        mask = self.norm_mask(mask)  # RMSNorm, an improvement for layer normazatliion
        mask = self.in_proj_mask(mask)
        mask = rearrange(mask, "b l d -> b d l")

        out = self.fusion_ssm(x,mask,info)





        x_residual = F.silu(
            x_act)  # The linear representation and followed a Silu activation to obtain the gated key
        x_combined = out * x_residual  # Key and   value are multiplied
        x_combined = rearrange(x_combined, "b d l -> b l d")
        x_combined = self.out_proj_x(x_combined)  # Adjust the channel dimension to the initial ones
        x_out = self.final_unembed_x(x_combined, (h, w))
        x_out = x_out + pre_x
        return x_out


class SSM_Manipulation(nn.Module):
    '''
    The ssm manipulation to capture the long-range dependencies
    '''

    def __init__(self, d_model, state_size, d_inner, d_conv, conv_bias=True, use_casual1D=True, adjust=True):
        super(SSM_Manipulation, self).__init__()
        self.conv1d = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1
        )  # 2N->2N, depth-wise convolution, L+d_conv-1
        self.ssm = S6(d_model=d_model, d_state=state_size, d_inner=d_inner)
        self.activation = "silu"  # y=x*sigmoid(x)
        self.act = nn.SiLU()
        self.use_casual1D = use_casual1D
        if adjust:
            self.adjust = nn.Conv1d(d_inner, d_inner, kernel_size=1)
        else:
            self.adjust = nn.Identity()

    def forward(self, x):
        assert self.activation in ["silu", "swish"]
        if self.use_casual1D:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d.weight, "d 1 w -> d w"),
                bias=self.conv1d.bias,
                activation=self.activation
            )
        else:
            x = self.act(self.conv1d(x)[..., :x.shape[-1]])
        x_ssm = self.ssm(x)  # The SSM to capture the long-range dependencies
        x_ssm = self.adjust(x_ssm)
        return x_ssm


class S6(nn.Module):
    def __init__(self, d_model, d_state=16, d_inner=128, dt_rank="auto", dt_min=0.001,
                 dt_max=0.1,
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4, use_scan_cuda=True):
        super(S6, self).__init__()

        self.d_model = d_model  # N,  feature dimension
        self.d_state = d_state  # D,  hidden state size
        self.d_inner = d_inner  # 2N, feature dimension after expansion
        self.use_scan_cuda = use_scan_cuda
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  # N/16, inner rank size

        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        self.x_proj = nn.Linear(
            self.d_inner, self.dt_rank + self.d_state * 2,
            bias=False)  # Projection to generate Delta, B and C, 2N->N/16+D+D

        dt_init_std = self.dt_rank ** -0.5 * dt_scale  # 1/sqrt(rank)
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)  # Constant initialization
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)  # Uniform distribution initialization
        else:
            raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            # dt_min is 1e-3 and dt_max is 0.1,
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)  ### Limite the minimal value as 1e-4
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))  ### Calculate the inverse
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)  ## Keep the gradients fixed
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()  # Transition matrix A using HiPPO
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

    def forward(self, u):
        batch, dim, seqlen = u.shape

        A = -torch.exp(self.A_log.float())

        # assert self.activation in ["silu", "swish"]
        x_dbl = self.x_proj(rearrange(u, "b d l -> (b l) d"))  # (bl d)
        delta, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        delta = self.dt_proj.weight @ delta.t()
        delta = rearrange(delta, "d (b l) -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        delta_bias = self.dt_proj.bias.float()
        delta_softplus = True
        dtype_in = u.dtype
        if not self.use_scan_cuda:
            u = u.float()
            delta = delta.float()
            if delta_bias is not None:
                delta = delta + delta_bias[..., None].float()
            if delta_softplus:
                delta = F.softplus(delta)  # delta = log(1+exp(delta))
            batch, dim, dstate = u.shape[0], A.shape[0], A.shape[1]
            is_variable_B = B.dim() >= 3
            is_variable_C = C.dim() >= 3
            if A.is_complex():
                if is_variable_B:
                    B = torch.view_as_complex(rearrange(B.float(), "... (L two) -> ... L two", two=2))
                if is_variable_C:
                    C = torch.view_as_complex(rearrange(C.float(), "... (L two) -> ... L two", two=2))
            else:
                B = B.float()
                C = C.float()
            x = A.new_zeros((batch, dim, dstate))
            ys = []
            deltaA = torch.exp(torch.einsum('bdl,dn->bdln', delta, A))
            if not is_variable_B:
                deltaB_u = torch.einsum('bdl,dn,bdl->bdln', delta, B, u)
            else:
                if B.dim() == 3:
                    deltaB_u = torch.einsum('bdl,bnl,bdl->bdln', delta, B, u)
                else:
                    B = repeat(B, "B G N L -> B (G H) N L", H=dim // B.shape[1])
                    deltaB_u = torch.einsum('bdl,bdnl,bdl->bdln', delta, B, u)
            if is_variable_C and C.dim() == 4:
                C = repeat(C, "B G N L -> B (G H) N L", H=dim // C.shape[1])
            last_state = None
            for i in range(u.shape[2]):
                x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
                if not is_variable_C:
                    y = torch.einsum('bdn,dn->bd', x, C)
                else:
                    if C.dim() == 3:
                        y = torch.einsum('bdn,bn->bd', x, C[:, :, i])
                    else:
                        y = torch.einsum('bdn,bdn->bd', x, C[:, :, :, i])
                if i == u.shape[2] - 1:
                    last_state = x
                if y.is_complex():
                    y = y.real * 2
                ys.append(y)
            y = torch.stack(ys, dim=2)  # (batch dim L)
            out = y
            out = out.to(dtype=dtype_in)
            return out

        else:
            y = selective_scan_fn(
                u,
                delta,
                A,
                B,
                C,
                None,
                z=None,
                delta_bias=self.dt_proj.bias.float(),
                delta_softplus=True,
                return_last_state=False,
            )
            out = y
            out = out.to(dtype=dtype_in)
            return out

class SSM_Manipulation_fusion(nn.Module):
    '''
    The ssm manipulation to capture the long-range dependencies
    '''

    def  __init__(self, d_model, state_size, d_inner, d_conv, conv_bias=True, use_casual1D=True, adjust=True):
        super(SSM_Manipulation_fusion, self).__init__()
        self.conv1d_x = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1
        )  # 2N->2N, depth-wise convolution, L+d_conv-1
        self.conv1d_info = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1
        )  # 2N->2N, depth-wise convolution, L+d_conv-1
        self.conv1d_mask = nn.Conv1d(
            in_channels=d_inner,
            out_channels=d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            groups=d_inner,
            padding=d_conv - 1
        )  # 2N->2N, depth-wise convolution, L+d_conv-1
        self.ssm = S6_fusion(d_model=d_model, d_state=state_size, d_inner=d_inner)
        self.activation = "silu"  # y=x*sigmoid(x)
        self.act = nn.SiLU()
        self.use_casual1D = use_casual1D
        if adjust:
            self.adjust = nn.Conv1d(d_inner, d_inner, kernel_size=1)
        else:
            self.adjust = nn.Identity()

    def forward(self, x,mask,info):
        assert self.activation in ["silu", "swish"]
        if self.use_casual1D:
            x = causal_conv1d_fn(
                x=x,
                weight=rearrange(self.conv1d_x.weight, "d 1 w -> d w"),
                bias=self.conv1d_x.bias,
                activation=self.activation
            )
            mask = causal_conv1d_fn(
                x=mask,
                weight=rearrange(self.conv1d_mask.weight, "d 1 w -> d w"),
                bias=self.conv1d_mask.bias,
                activation=self.activation
            )
            info = causal_conv1d_fn(
                x=info,
                weight=rearrange(self.conv1d_info.weight, "d 1 w -> d w"),
                bias=self.conv1d_info.bias,
                activation=self.activation
            )
        else:
            x = self.act(self.conv1d(x)[..., :x.shape[-1]])
        x_ssm = self.ssm(x,mask,info)# The SSM to capture the long-range dependencies
        x_ssm = self.adjust(x_ssm)
        return x_ssm


class S6_fusion(nn.Module):
    def __init__(self, d_model, d_state=16, d_inner=128, dt_rank="auto", dt_min=0.001,
                 dt_max=0.1,
                 dt_init="random",
                 dt_scale=1.0,
                 dt_init_floor=1e-4, use_scan_cuda=True):
        super(S6_fusion, self).__init__()

        self.d_model = d_model  # N,  feature dimension
        self.d_state = d_state  # D,  hidden state size
        self.d_inner = d_inner  # 2N, feature dimension after expansion
        self.use_scan_cuda = use_scan_cuda
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank  # N/16, inner rank size

        self.C_proj = nn.Linear(
            self.d_inner, self.d_state,
            bias=False)  # Projection to generate Delta, B and C, 2N->N/16+D+D
        self.B_proj = nn.Linear(self.d_inner, d_state, bias=False)
        self.delta_proj = nn.Linear(self.d_inner, self.dt_rank, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        dt_init_std = self.dt_rank ** -0.5 * dt_scale  # 1/sqrt(rank)
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)  # Constant initialization
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

            # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
            # dt_min is 1e-3 and dt_max is 0.1,
        dt = torch.exp(
            torch.rand(self.d_inner) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)  ### Limite the minimal value as 1e-4
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = + torch.log(-torch.expm1(-dt))  ### Calculate the inverse
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)  ## Keep the gradients fixed
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        self.dt_proj.bias._no_reinit = True

        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32),
            "n -> d n",
            d=self.d_inner,
        ).contiguous()  # Transition matrix A using HiPPO
        A_log = torch.log(A)  # Keep A_log in fp32
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True

        # self.D = nn.Parameter(torch.ones(self.d_inner))  # Keep in fp32
        # self.D._no_weight_decay = True

    def forward(self, u,mask,info):
        batch, dim, seqlen = u.shape
        A = -torch.exp(self.A_log.float())

        # assert self.activation in ["silu", "swish"]
        delta = self.delta_proj(rearrange(mask,"b d l -> (b l) d"))
        C = self.C_proj(rearrange(u, "b d l -> (b l) d"))  # (bl d)
        delta = self.dt_proj.weight @ delta.t()
        delta = rearrange(delta, "d (b l) -> b d l", l=seqlen)
        B = self.B_proj(rearrange(info, "b d l -> (b l) d"))
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()

        dtype_in = u.dtype

        out = selective_scan_fn(
            u,
            delta,
            A,
            B,
            C,
            None,
            z=None,
            delta_bias=self.dt_proj.bias.float(),
            delta_softplus=True,
            return_last_state=False,
        )

        out = out.to(dtype=dtype_in)
        return out



class Mix(nn.Module):
    def __init__(self, m=-0.80):
        super(Mix, self).__init__()
        w = torch.nn.Parameter(torch.FloatTensor([m]), requires_grad=True)
        w = torch.nn.Parameter(w, requires_grad=True)
        self.w = w
        self.mix_block = nn.Sigmoid()

    def forward(self, fea1, fea2):
        mix_factor = self.mix_block(self.w)
        out = fea1 * mix_factor.expand_as(fea1) + fea2 * (1 - mix_factor.expand_as(fea2))
        return out

if __name__ == '__main__':
    # pass
    import time

    #
    start = time.perf_counter()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # ssm = S6(d_model=128, d_state=128, d_conv=3, expand=2, device=device, dtype=torch.float32)
    ssm = S6(d_model=16, d_state=16, d_inner=32).cuda()
    x = torch.randn(80, 32, 10000).to(device)
    output = ssm(x)
    end = time.perf_counter()
    print('excuting time is %s' % (end - start))