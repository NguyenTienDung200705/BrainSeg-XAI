"""
Model factory — hỗ trợ cả U-Net thuần (fallback) và
U-Net++ / DeepLabV3+ với EfficientNet/ResNet encoder (qua segmentation-models-pytorch).
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# ────────────────────────────────────────────────────────────
# 1.  U-Net thuần (giữ lại để backward-compat với weight cũ)
# ────────────────────────────────────────────────────────────
class DoubleConv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch), nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.conv(x)


class Down(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.pool_conv = nn.Sequential(nn.MaxPool2d(2), DoubleConv(ic, oc))
    def forward(self, x): return self.pool_conv(x)


class Up(nn.Module):
    def __init__(self, ic, oc):
        super().__init__()
        self.up   = nn.ConvTranspose2d(ic, ic // 2, 2, stride=2)
        self.conv = DoubleConv(ic, oc)
    def forward(self, x1, x2):
        x1 = self.up(x1)
        dy, dx = x2.size(2)-x1.size(2), x2.size(3)-x1.size(3)
        x1 = F.pad(x1, [dx//2, dx-dx//2, dy//2, dy-dy//2])
        return self.conv(torch.cat([x2, x1], 1))


class UNet(nn.Module):
    def __init__(self, n_channels=3, n_classes=1, features=[64,128,256,512]):
        super().__init__()
        f = features
        self.inc   = DoubleConv(n_channels, f[0])
        self.down1 = Down(f[0], f[1])
        self.down2 = Down(f[1], f[2])
        self.down3 = Down(f[2], f[3])
        self.down4 = Down(f[3], f[3]*2)
        self.up1   = Up(f[3]*2, f[3])
        self.up2   = Up(f[3],   f[2])
        self.up3   = Up(f[2],   f[1])
        self.up4   = Up(f[1],   f[0])
        self.outc  = nn.Conv2d(f[0], n_classes, 1)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        x  = self.up1(x5, x4)
        x  = self.up2(x,  x3)
        x  = self.up3(x,  x2)
        x  = self.up4(x,  x1)
        return self.outc(x)


# ────────────────────────────────────────────────────────────
# 2.  SMP wrapper — U-Net++ / DeepLabV3+ với pretrained encoder
# ────────────────────────────────────────────────────────────
def build_smp_model(
    arch: str = "unetplusplus",
    encoder: str = "efficientnet-b4",
    encoder_weights: str = "imagenet",
    n_classes: int = 1,
):
    """
    arch options    : 'unetplusplus' | 'unet' | 'deeplabv3plus' | 'fpn' | 'manet'
    encoder options : 'efficientnet-b4' | 'resnet50' | 'resnet34' | 'se_resnext50_32x4d'
    """
    try:
        import segmentation_models_pytorch as smp
    except ImportError:
        raise ImportError(
            "segmentation-models-pytorch chưa được cài.\n"
            "Chạy: pip install segmentation-models-pytorch timm"
        )

    model = smp.create_model(
        arch,
        encoder_name=encoder,
        encoder_weights=encoder_weights,
        in_channels=3,
        classes=n_classes,
        activation=None,          # raw logits — sigmoid trong loss/post-proc
    )
    return model


# ────────────────────────────────────────────────────────────
# 3.  Auto-loader: thử SMP trước, fallback về UNet thuần
# ────────────────────────────────────────────────────────────
def get_model(
    arch: str = "unetplusplus",
    encoder: str = "efficientnet-b4",
    n_classes: int = 1,
    pretrained: bool = True,
):
    try:
        model = build_smp_model(
            arch=arch,
            encoder=encoder,
            encoder_weights="imagenet" if pretrained else None,
            n_classes=n_classes,
        )
        print(f"[Model] Dùng SMP — arch={arch}, encoder={encoder}")
        return model, "smp"
    except Exception as e:
        print(f"[Model] SMP không khả dụng ({e}), fallback UNet thuần.")
        return UNet(n_channels=3, n_classes=n_classes), "vanilla"
