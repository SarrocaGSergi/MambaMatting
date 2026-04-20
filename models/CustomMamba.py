import torch
import math
import torch.nn as nn
from mamba_ssm import Mamba
import torch.utils.checkpoint as cp
from timm.layers import trunc_normal_, to_2tuple
from monai.networks.blocks.dynunet_block import UnetOutBlock
from monai.networks.blocks.unetr_block import UnetrBasicBlock, UnetrUpBlock


class PatchEmbed3D(nn.Module):
    """Image / Video to 3D Patch Embedding (as in VideoMamba)

    This is the same PatchEmbed class used in videomamba_video.py:
    it uses a Conv3d with kernel_size=(tubelet_size, patch_h, patch_w)
    and stride=(tubelet_size, patch_h, patch_w).

    Input expected: x shape [B, C, T, H, W]
    Output: conv output shape [B, embed_dim, T_out, H_out, W_out]
    """
    def __init__(self, img_size=224, patch_size=16, kernel_size=1, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches
        self.tubelet_size = kernel_size

        # 3D conv: (time, height, width)
        self.proj = nn.Conv3d(
            in_chans, embed_dim,
            kernel_size=(kernel_size, patch_size[0], patch_size[1]),
            stride=(kernel_size, patch_size[0], patch_size[1])
        )

    def forward(self, x):
        # x: [B, C, T, H, W]
        x = self.proj(x)
        return x


class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """
    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)
        if self.norm is not None:
            x = self.norm(x)
        return x


class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        if x.ndim == 5:
            B, F, H, W, C = x.shape
            x = x.view(B*F, H, W, C)

        F, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(F, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t, use_fp16=False):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        if use_fp16:
            t_freq = t_freq.to(dtype=torch.float16)
        t_emb = self.mlp(t_freq)
        return t_emb


#################################################################################
#               Attention Layers from TIMM                                      #
#################################################################################

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0., use_lora=False,
                 attention_mode='math'):
        super().__init__()
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5
        self.attention_mode = attention_mode
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4).contiguous()
        q, k, v = qkv.unbind(0)  # make torchscript happy (cannot use tensor as tuple)

        '''if self.attention_mode == 'xformers':  # cause loss nan while using with amp
            # https://github.com/facebookresearch/xformers/blob/e8bd8f932c2f48e3a3171d06749eecbbf1de420c/xformers/ops/fmha/__init__.py#L135
            q_xf = q.transpose(1, 2).contiguous()
            k_xf = k.transpose(1, 2).contiguous()
            v_xf = v.transpose(1, 2).contiguous()
            x = xformers.ops.memory_efficient_attention(q_xf, k_xf, v_xf).reshape(B, N, C)
        '''
        if self.attention_mode == 'flash':
            # cause loss nan while using with amp
            # Optionally use the context manager to ensure one of the fused kerenels is run
            with torch.backends.cuda.sdp_kernel(enable_math=False):
                x = torch.nn.functional.scaled_dot_product_attention(q, k, v).reshape(B, N, C)  # require pytorch 2.0

        elif self.attention_mode == 'math':
            attn = (q @ k.transpose(-2, -1)) * self.scale
            attn = attn.softmax(dim=-1)
            attn = self.attn_drop(attn)
            x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        else:
            raise NotImplemented

        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Head(nn.Module):
    def __init__(self, in_chans=3, embed_dim=96, patch_size=2, norm_layer=nn.LayerNorm, use_fp16=False,
                 patch_norm=True):
        super().__init__()

        self.timestep_embed = TimestepEmbedder(hidden_size=embed_dim)

        self.norm_layer = norm_layer
        self.use_fp16 = use_fp16
        self.embed_dim = embed_dim  # after merging
        self.patch_size = patch_size

        self.patch_embed = PatchEmbed2D(patch_size=patch_size, in_chans=in_chans, embed_dim=self.embed_dim,
                                        norm_layer=norm_layer if patch_norm else None)

        self.pos_embed = None

    def forward(self, x):
        # Make Sure x dimensions are 5
        org_dim = x.ndim # Save them if image mating instead of video matting
        if org_dim == 4:
            x = x.unsqueeze(0)

        B, F, C, H, W = x.shape
        device = x.device

        # Flatten batch and frame dimensions for parallel processing
        x = x.view(B * F, C, H, W)  # (B*F, C, H, W)

        # Patch embedding
        x = self.patch_embed(x)  # (B*F, H', W', d1)
        nh, nw = x.shape[1], x.shape[2]
        assert nh==H//2 and nw==W//2, "Error on H2 or W2 values"

        # Add timestep embeddings
        t = torch.arange(F, device=device).repeat_interleave(B)
        t_emb = self.timestep_embed(t, use_fp16=self.use_fp16)
        x = x + t_emb[:, None, None, :]  # (B*F, H'', W'', d)

        # Tokenization from (B, F, H, W, C) to (B, nf * nw * nw, d)
        x = x.view(B, F, nh * nw, self.embed_dim)
        z_hat = x.reshape(B, F * nh * nw, self.embed_dim)  # (B, total_tokens, d)

        # Step 4: Add learned spatio-temporal positional embedding
        total_tokens = F * nh * nw
        if self.pos_embed is None or self.pos_embed.shape[1] != total_tokens:
            self.pos_embed = nn.Parameter(torch.randn(1, total_tokens, self.embed_dim, device=device))

        z = z_hat + self.pos_embed  # (B, total_tokens, d)
        z_proj = z.view(B, F, nh, nw, self.embed_dim)
        return z, z_proj

class MattenBlock(nn.Module):
    def __init__(self, dim, num_heads=8, attn_drop=0., proj_drop=0., d_state=16, attention_mode='math'):
        super().__init__()
        self.spatial_layer = Mamba(
        # This module uses roughly 3 * expand * d_model^2 parameters
        d_model=dim,        # Model dimension d_model
        d_state=d_state,    # SSM state expansion factor
        d_conv=4,           # Local convolution width
        expand=2,           # Block expansion factor
        use_fast_path=False,
        )

        self.spatial_attn = Attention(dim, num_heads=num_heads,
                                      attn_drop=attn_drop, proj_drop=proj_drop,
                                      attention_mode=attention_mode)

        self.temp_attn = Attention(dim, num_heads=num_heads,
                                      attn_drop=attn_drop, proj_drop=proj_drop,
                                      attention_mode=attention_mode)

    def forward(self, z, nf, nh, nw):
            """
            Args:
                z: (B, F*H*W, C) token sequence
                nf: number of frames
                nh: height (in patches)
                nw: width (in patches)
            Returns:
                output: (B, nf*nh*nw, C)
            """

            B, N, C = z.shape
            assert N == nf * int(nh) * int(nw), f"Token mismatch: got {N}, expected {nf*nh*nw}"

            # Spatial-First Mamba scan
            z = self.spatial_layer(z)
            # Spatial Attention
            z_spatial = z.view(B, nf, nh * nw, C)  # (B, F, HW, C)
            z_spatial = z_spatial.reshape(B * nf, nh * nw, C)
            z_spatial = self.spatial_attn(z_spatial)
            z_spatial = z_spatial.view(B, nf, nh * nw, C)

            # Temporal Attention
            z_temporal = z_spatial.permute(0, 2, 1, 3)  # (B, HW, F, C)
            z_temporal = z_temporal.reshape(B * nh * nw, nf, C)
            z_temporal = self.temp_attn(z_temporal)
            z_temporal = z_temporal.view(B, nh * nw, nf, C).permute(0, 2, 1, 3)  # (B, F, HW, C)

            # Reshape to 1D token sequence
            z_out = z_temporal.reshape(B, nf * nh * nw, C)
            z_proj = z_out.view(B, nf, nh, nw, C)
            return z_out, z_proj

class MattenEncoder(nn.Module):
    def __init__(self, patch_size=2, in_chans=3, depths=[2, 2, 9, 2], dims=[96, 192, 384, 768],
                 d_state=16, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.2, norm_layer=nn.LayerNorm,
                 use_checkpoint: bool=False, **kwargs):
        super().__init__()
        self.num_layers = len(depths)
        self.use_checkpoint = use_checkpoint
        if isinstance(dims, int):
            dims = [int(dims * 2 ** i_layer) for i_layer in range(self.num_layers)]

        self.embed_dim = dims[0]
        self.dims = dims


        self.head = Head(patch_size=patch_size, in_chans=in_chans, embed_dim=dims[0])

        self.pos_embed = None

        self.layers = nn.ModuleList()
        self.downsamples = nn.ModuleList()

        for i_layer in range(self.num_layers):
            layer = MattenBlock(dim=dims[i_layer],
                                num_heads=8,
                                attn_drop=attn_drop_rate,
                                proj_drop=0.,
                                attention_mode='math',
                                d_state=d_state
            )

            self.layers.append(layer)
            if i_layer < self.num_layers - 1:
                self.downsamples.append(PatchMerging2D(dim=dims[i_layer], norm_layer=norm_layer))

        self.apply(self._init_weights)

    def _init_weights(self, m: nn.Module):
        """
        out_proj.weight which is previously initilized in VSSBlock, would be cleared in nn.Linear
        no fc.weight found in the any of the model parameters
        no nn.Embedding found in the any of the model parameters
        so the thing is, VSSBlock initialization is useless

        Conv2D is not intialized !!!
        """
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        B, F, C, H, W = x.shape
        x_ret = []
        x_ret.append(x.view(B*F, C, H, W))

        z_token, z_proj = self.head(x)
        B, nf, nh, nw, d = z_proj.shape
        for s, layer in enumerate(self.layers):
            z_token, z_proj = layer(z_token, nf, nh, nw)
            z_proj_2 = z_proj.permute(0, 1, 4, 3, 2)
            x_ret.append(z_proj_2.reshape(B*nf, d, nh, nw))
            if s < len(self.downsamples):
                z_proj = self.downsamples[s](z_proj)
                F, nh, nw, d = z_proj.shape
                z_token = z_proj.view(B, nf*nh*nw, d)

        return x_ret


class MatMat(nn.Module):
    def __init__(self, patch_size, in_channels=3, out_chans=1, feat_size=[48, 96, 192, 384, 768], drop_path_rate=0,
                 layer_scale_init_value=1e-6, hidden_size: int = 768, norm_name="instance", res_block: bool = True,
                 spatial_dims=2, deep_supervision: bool = False, use_checkpoint: bool = True) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.in_channels = in_channels
        self.use_checkpoint = use_checkpoint
        self.out_chans = out_chans
        self.spatial_dims = spatial_dims
        self.drop_path_rate = drop_path_rate
        self.feat_size = feat_size
        self.layer_scale_init_value = layer_scale_init_value

        self.matten_encoder = MattenEncoder(patch_size=patch_size, in_chans=feat_size[0], dims=feat_size[1:],
                                            use_checkpoint=self.use_checkpoint)


        self.stem = nn.Sequential(
            nn.Conv2d(self.in_channels, feat_size[0], kernel_size=7, stride=2, padding=3),
            nn.InstanceNorm2d(feat_size[0], eps=1e-5, affine=True),
        )



        self.encoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.in_channels,
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder2 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[1],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder3 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[2],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.encoder4 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[3],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.encoder5 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[4],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder6 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[4],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )

        self.decoder5 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.hidden_size,
            out_channels=self.feat_size[3],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder4 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[3],
            out_channels=self.feat_size[2],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder3 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[2],
            out_channels=self.feat_size[1],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder2 = UnetrUpBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[1],
            out_channels=self.feat_size[0],
            kernel_size=3,
            upsample_kernel_size=2,
            norm_name=norm_name,
            res_block=res_block,
        )
        self.decoder1 = UnetrBasicBlock(
            spatial_dims=spatial_dims,
            in_channels=self.feat_size[0],
            out_channels=self.feat_size[0],
            kernel_size=3,
            stride=1,
            norm_name=norm_name,
            res_block=res_block,
        )

        # deep supervision support
        self.deep_supervision = deep_supervision
        self.matting_out_layers = nn.ModuleList()
        for i in range(4):
            self.matting_out_layers.append(UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[i],
                out_channels=4
            ))
        self.seg_out_layer = UnetOutBlock(
                spatial_dims=spatial_dims,
                in_channels=self.feat_size[0],
                out_channels=1
            )

    def _cp1(self, fn, *tensors):
        return cp.checkpoint(lambda *x: fn(*x), *tensors,
                             use_reentrant=False, preserve_rng_state=False) if self.use_checkpoint else fn(*tensors)



    def forward(self, x, seg_pass = False):
        if x.ndim == 5:
            B, F, C, H, W = x.shape
            x_spat = x.view(B * F, C, H, W)
        else:
            x = x.unsqueeze(0)
            B, F, C, H, W = x.shape
            x_spat = x.view(B*F, C, H, W)
        x1 = self.stem(x_spat)
        nf, d, nh, nw = x1.shape
        vss_outs = self.matten_encoder(x1.view(B, F, d, nh, nw))
        enc1 = self.encoder1(x_spat)
        enc2 = self.encoder2(vss_outs[0])
        enc3 = self.encoder3(vss_outs[1])
        enc4 = self.encoder4(vss_outs[2])
        enc5 = self.encoder5(vss_outs[3])
        enc_hidden = vss_outs[4]

        dec4 = self.decoder6(enc_hidden, enc5)
        dec3 = self.decoder5(dec4, enc4)
        dec2 = self.decoder4(dec3, enc3)
        dec1 = self.decoder3(dec2, enc2)
        dec0 = self.decoder2(dec1, enc1)
        dec_out = self.decoder1(dec0)

        if self.deep_supervision and not seg_pass:
            feat_out = [dec_out, dec1, dec2, dec3]
            out = []
            for i in range(4):
                pred = self.matting_out_layers[i](feat_out[i])  # [B*F, C, H_i, W_i]
                pred = torch.clamp(pred, 0., 1.)
                pred = pred.view(B, F, pred.shape[1], pred.shape[2], pred.shape[3])  # → [B, F, C, H_i, W_i]
                out.append(pred)
                return out

        elif not self.deep_supervision and not seg_pass:
            out = []
            pred = self.matting_out_layers[0](dec_out)
            pred_fgr, pred_pha = pred.split([3, 1], dim=-3)
            pred_fgr = pred_fgr.view(B, F, 3, pred.shape[2], pred.shape[3])
            pred_fgr = pred_fgr + x
            pred_fgr = torch.clamp(pred_fgr, 0., 1.)
            # pred_fgr = pred_fgr.view(B, F, 3, pred.shape[2], pred.shape[3])
            out.append(pred_fgr)
            pred_pha = torch.clamp(pred_pha, 0., 1.)
            pred_pha = pred_pha.view(B, F, 1, pred.shape[2], pred.shape[3])
            out.append(pred_pha)
            return out

        elif not self.deep_supervision and seg_pass:
            out = self.seg_out_layer(dec_out)
            out = out.view(B, F, out.shape[1], out.shape[2], out.shape[3])
            return out

if __name__ == '__main__':
    x = torch.randn(1, 3, 15, 224, 512).to("cuda")
    model = PatchEmbed3D().to("cuda")
    pred = model(x)
    print(pred)