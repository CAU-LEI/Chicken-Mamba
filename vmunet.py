# vmunet.py
from .vmamba import VSSM
import torch
import torch.nn.functional as F
from torch import nn


# ----------------------------
# Utilities: Haar DWT (fixed-weight) 2D
# ----------------------------
class HaarDWT2D(nn.Module):
    """
    Simple 2D Haar wavelet decomposition (LL, LH, HL, HH), implemented with fixed convolution kernels + stride=2.
    Input:  NCHW
    Output: (LL, LH, HL, HH), all with shape [N, C, H/2, W/2]
    """
    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        # Haar filters (2x2), constructed according to common definitions
        ll = torch.tensor([[0.5, 0.5],
                           [0.5, 0.5]], dtype=torch.float32)
        lh = torch.tensor([[-0.5, -0.5],
                           [ 0.5,  0.5]], dtype=torch.float32)
        hl = torch.tensor([[-0.5,  0.5],
                           [-0.5,  0.5]], dtype=torch.float32)
        hh = torch.tensor([[ 0.5, -0.5],
                           [-0.5,  0.5]], dtype=torch.float32)

        # 4 kernels of 2x2, for group convolution
        weight = torch.stack([ll, lh, hl, hh], dim=0)  # [4, 2, 2]
        weight = weight.unsqueeze(1)                   # [4, 1, 2, 2]
        weight = weight.repeat(channels, 1, 1, 1)      # [4*C, 1, 2, 2]

        self.register_buffer('weight', weight)  # Fixed parameters
        self.groups = channels

    def forward(self, x):
        # x: [N,C,H,W]
        N, C, H, W = x.shape
        # Convolution kernels expanded by group=channels: 4 filters per group
        # Trick: repeat input channels 4 times, then use group convolution to get 4*C channels at once, then split into 4 parts
        x_rep = x.repeat(1, 4, 1, 1)  # [N, 4C, H, W]
        y = F.conv2d(x_rep, self.weight, stride=2, padding=0, groups=self.groups*4)  # [N,4C,H/2,W/2]
        # Split into 4 parts [N,C,H/2,W/2]
        y = y.view(N, 4, C, H//2, W//2).contiguous()
        LL, LH, HL, HH = y[:, 0], y[:, 1], y[:, 2], y[:, 3]
        return LL, LH, HL, HH


# ----------------------------
# MWT: Multiscale Wavelet Transform Module
# ----------------------------
class MWT(nn.Module):
    """
    Multiscale wavelet module (Figure b): Three spatial convolution branches + two wavelet frequency domain branches, 
    finally concatenated at 1/8 scale to form ~1280 channels, compressed to 512 with 1x1, 
    then adaptive pooling to 1/32 (aligned with VSSM bottleneck).
    """
    def __init__(self, in_ch=3, out_ch=512):
        super().__init__()
        self.out_ch = out_ch

        # --- Spatial branches: Output unified 256 channels ---
        def conv_bn_act(ci, co, k=3, s=1, p=1):
            return nn.Sequential(
                nn.Conv2d(ci, co, k, s, p, bias=False),
                nn.BatchNorm2d(co),
                nn.ReLU(inplace=True),
            )

        # Branch S1: 1x1 -> 3x3 -> 3x3
        self.s1 = nn.Sequential(
            conv_bn_act(in_ch,   128, k=1, s=1, p=0),
            conv_bn_act(128,     192, k=3, s=1, p=1),
            conv_bn_act(192,     256, k=3, s=1, p=1),
        )
        # Branch S2: 3x3 -> 1x1
        self.s2 = nn.Sequential(
            conv_bn_act(in_ch,   192, k=3, s=1, p=1),
            conv_bn_act(192,     256, k=1, s=1, p=0),
        )
        # Branch S3: 3x3 -> 3x3
        self.s3 = nn.Sequential(
            conv_bn_act(in_ch,   192, k=3, s=1, p=1),
            conv_bn_act(192,     256, k=3, s=1, p=1),
        )
        # Pooling branch
        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)  # Light downsampling, will be aligned later

        # --- Frequency branches via DWT ---
        self.dwt = HaarDWT2D(in_ch)
        self.conv_ll = conv_bn_act(in_ch, 256, k=3, s=1, p=1)          # LL → 3x3
        self.conv_h  = conv_bn_act(in_ch*3, 256, k=3, s=1, p=1)        # concat(LH,HL,HH) → 3x3

        # Fusion to 1280 (= 256*5) -> 512
        self.fuse = nn.Sequential(
            nn.Conv2d(256*5, out_ch, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        # x: [N,3,H,W]
        N, C, H, W = x.shape
        target_1_8 = (max(H // 8, 1), max(W // 8, 1))     # Target 1/8 spatial scale
        target_1_32 = (max(H // 32, 1), max(W // 32, 1))  # Target 1/32 spatial scale

        # Spatial branches
        s1 = self.s1(x)
        s2 = self.s2(x)
        s3 = self.s3(x)
        sp = self.pool(x)  # Light downsampling

        # Frequency domain branches (DWT)
        LL, LH, HL, HH = self.dwt(x)      # [N,C,H/2,W/2]
        fr_ll = self.conv_ll(LL)          # [N,256,H/2,W/2]
        fr_h  = self.conv_h(torch.cat([LH, HL, HH], dim=1))  # [N,256,H/2,W/2]

        # Align to 1/8 resolution
        s1 = F.interpolate(s1, size=target_1_8, mode='bilinear', align_corners=False)
        s2 = F.interpolate(s2, size=target_1_8, mode='bilinear', align_corners=False)
        s3 = F.interpolate(s3, size=target_1_8, mode='bilinear', align_corners=False)
        sp = F.interpolate(sp, size=target_1_8, mode='bilinear', align_corners=False)
        fr_ll = F.interpolate(fr_ll, size=target_1_8, mode='bilinear', align_corners=False)
        fr_h  = F.interpolate(fr_h,  size=target_1_8, mode='bilinear', align_corners=False)

        # Concatenate -> 1x1 compression to 512
        feat_1_8 = torch.cat([s1, s2, s3, fr_ll, fr_h], dim=1)  # [N,1280,H/8,W/8]
        feat_512 = self.fuse(feat_1_8)                          # [N,512,H/8,W/8]

        # Align to 1/32 (use adaptive pooling to safely adapt to various input sizes)
        feat_1_32 = F.adaptive_avg_pool2d(feat_512, output_size=target_1_32)  # [N,512,H/32,W/32]
        return feat_1_32


# ----------------------------
# DCF: Multi-scale Dilated Convolution Fusion
# ----------------------------
class DCF(nn.Module):
    """
    Figure (c) DCF: Three 3x3 dilated convolution branches (r=1/3/5) → Sum → 3x3 compression → Softmax attention →
    Weighted aggregation → Residual + 1x1 output. Keep channels unchanged.
    Input/Output: NCHW
    """
    def __init__(self, channels):
        super().__init__()
        self.branch1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, dilation=1, bias=False)
        self.branch3 = nn.Conv2d(channels, channels, kernel_size=3, padding=3, dilation=3, bias=False)
        self.branch5 = nn.Conv2d(channels, channels, kernel_size=3, padding=5, dilation=5, bias=False)

        # Generate attention map (3 scale weights), pixel-wise softmax
        self.attn_prep = nn.Conv2d(channels, 3, kernel_size=3, padding=1, bias=True)

        # Output projection
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1, bias=False)

        # Normalization/Activation
        self.bn = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        F_a = self.branch1(x)
        F_b = self.branch3(x)
        F_c = self.branch5(x)

        # Element sum → 3x3 → softmax to get attention for three scales
        S = F_a + F_b + F_c
        attn_logits = self.attn_prep(S)             # [N,3,H,W]
        attn = F.softmax(attn_logits, dim=1)        # A',B',C'

        # Pixel-wise weighted sum
        F_attn = (F_a * attn[:, 0:1]) + (F_b * attn[:, 1:2]) + (F_c * attn[:, 2:3])

        # Residual + 1x1
        out = self.out_proj(self.relu(self.bn(F_attn + x)))
        return out


# ----------------------------
# New VMUNet: Integrated with MWT + DCF
# ----------------------------
class VMUNet_MWT_DCF(nn.Module):
    """
    Enhanced VMUNet based on VSSM:
    - Input first passes through MWT to get 1/32 scale 512-channel frequency/spatial domain features;
    - Encoder output bottleneck (NHWC) converted to NCHW and added to MWT features after alignment;
    - Pass through DCF fusion, then sent to decoder;
    - Others remain consistent with original version.
    """
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None):
        super().__init__()

        self.num_classes = num_classes
        self.load_ckpt_path = load_ckpt_path

        # 1) VSSM backbone (don't modify vmamba.py)
        self.vssm = VSSM(in_chans=input_channels,
                         num_classes=num_classes,
                         depths=depths,
                         depths_decoder=depths_decoder,
                         drop_rate=0.0,
                         attn_drop_rate=0.0,
                         drop_path_rate=drop_path_rate)

        # Bottleneck channels (usually dims[-1], default 768)
        self.bottleneck_ch = self.vssm.dims[-1]

        # 2) MWT: Extract 1/32 scale, 512-channel fusion features from original input
        self.mwt = MWT(in_ch=max(3, input_channels), out_ch=512)

        # 3) Feature Connection: Project MWT's 512 → to bottleneck channels
        self.mwt_proj = nn.Conv2d(512, self.bottleneck_ch, kernel_size=1, bias=False)

        # 4) DCF: Bottleneck multi-scale dilated convolution fusion
        self.dcf = DCF(self.bottleneck_ch)

    def forward(self, x):
        # Copy single channel input to 3 channels (MWT uses 3 channels)
        if x.size(1) == 1:
            x_rgb = x.repeat(1, 3, 1, 1)
        else:
            x_rgb = x

        # --- MWT path (NCHW) ---
        mwt_1_32 = self.mwt(x_rgb)                # [N,512,H/32,W/32]
        mwt_aligned = self.mwt_proj(mwt_1_32)     # [N,768,H/32,W/32] aligned with bottleneck

        # --- VSSM Encoder ---
        # Use VSSM's feature interface to get bottleneck & skip (NHWC)
        x_bottleneck, skip_list = self.vssm.forward_features(x)  # x: NCHW → patch_embed internally converts to NHWC
        # NHWC → NCHW
        x_bn = x_bottleneck.permute(0, 3, 1, 2).contiguous()     # [N,768,H/32,W/32]

        # --- Feature Connection + DCF ---
        x_bn = x_bn + mwt_aligned                                 # Element-wise addition
        x_bn = self.dcf(x_bn)                                     # DCF fusion

        # Back to NHWC to enter decoder
        x_bn = x_bn.permute(0, 2, 3, 1).contiguous()             # [N,H/32,W/32,768]

        # --- VSSM Decoder + Final Projection ---
        x_up = self.vssm.forward_features_up(x_bn, skip_list)    # NHWC
        logits = self.vssm.forward_final(x_up)                   # NCHW, num_classes

        if self.num_classes == 1:
            return torch.sigmoid(logits)
        else:
            return logits

    # Compatible with your original loading method (optional)
    def load_from(self):
        if self.load_ckpt_path is None:
            return
        model_dict = self.vssm.state_dict()
        modelCheckpoint = torch.load(self.load_ckpt_path)
        pretrained = modelCheckpoint['model']
        # Only load weights aligned with VSSM
        new_dict = {k: v for k, v in pretrained.items() if k in model_dict}
        model_dict.update(new_dict)
        self.vssm.load_state_dict(model_dict)
        print(f'Backbone loaded: update {len(new_dict)}/{len(model_dict)} params')


# ----------------------------
# Keep original VMUNet (without MWT/DCF) for compatibility with old code
# ----------------------------
class VMUNet(nn.Module):
    def __init__(self,
                 input_channels=3,
                 num_classes=1,
                 depths=[2, 2, 9, 2],
                 depths_decoder=[2, 9, 2, 2],
                 drop_path_rate=0.2,
                 load_ckpt_path=None):
        super().__init__()
        self.num_classes = num_classes
        self.load_ckpt_path = load_ckpt_path
        self.vmunet = VSSM(in_chans=input_channels,
                           num_classes=num_classes,
                           depths=depths,
                           depths_decoder=depths_decoder,
                           drop_path_rate=drop_path_rate)

    def forward(self, x):
        if x.size(1) == 1:
            x = x.repeat(1, 3, 1, 1)
        logits = self.vmunet(x)
        return torch.sigmoid(logits) if self.num_classes == 1 else logits
