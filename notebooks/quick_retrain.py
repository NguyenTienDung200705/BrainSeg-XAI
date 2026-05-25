"""
quick_retrain.py — Script retrain nhanh với config đã fix.
Chạy: python quick_retrain.py [--gpu 0] [--epochs 80] [--size 384]

Áp dụng tất cả fix từ phân tích Dice=0.71:
  1. Bỏ no-tumor images
  2. pos_weight=15 cho BCELoss
  3. Dropout 0.3 trong decoder
  4. Image size 384
  5. Dual scheduler (cosine + plateau)
  6. Early stopping
"""
import os, sys, random, glob, argparse
import numpy as np
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import segmentation_models_pytorch as smp
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--data',    default='../dataset/BrainMRI')
parser.add_argument('--save',    default='../backend/weights/unet_best.pth')
parser.add_argument('--arch',    default='unetplusplus')
parser.add_argument('--encoder', default='efficientnet-b2')
parser.add_argument('--size',    type=int, default=384)
parser.add_argument('--epochs',  type=int, default=80)
parser.add_argument('--batch',   type=int, default=6)
parser.add_argument('--lr',      type=float, default=2e-4)
parser.add_argument('--pos-weight', type=float, default=15.0, dest='pos_weight')
parser.add_argument('--dropout', type=float, default=0.3)
args = parser.parse_args()

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n{'='*55}")
print(f"  Brain Tumor Segmentation — Retrain v2")
print(f"{'='*55}")
print(f"  Device   : {DEVICE}")
print(f"  Model    : {args.arch} + {args.encoder}")
print(f"  Img size : {args.size}×{args.size}")
print(f"  Batch    : {args.batch} | Epochs: {args.epochs}")
print(f"  pos_weight: {args.pos_weight} | dropout: {args.dropout}")
print(f"{'='*55}\n")

# ── Transforms ────────────────────────────────────────────────────────────
def get_tfm(mode):
    sz = args.size
    if mode == 'train':
        return A.Compose([
            A.Resize(sz, sz),
            A.HorizontalFlip(p=0.5), A.VerticalFlip(p=0.3),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2,
                               rotate_limit=30, border_mode=cv2.BORDER_REFLECT, p=0.6),
            A.ElasticTransform(alpha=120, sigma=6, alpha_affine=3.6, p=0.3),
            A.OneOf([
                A.RandomBrightnessContrast(0.3, 0.3, p=1),
                A.RandomGamma((70,130), p=1),
                A.CLAHE(4.0, p=1),
            ], p=0.6),
            A.GaussianBlur(blur_limit=(3,7), p=0.2),
            A.GaussNoise(var_limit=(10,50), p=0.2),
            A.CoarseDropout(max_holes=4, max_height=32, max_width=32, fill_value=0, p=0.2),
            A.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(sz, sz),
        A.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
        ToTensorV2(),
    ])

class DS(Dataset):
    def __init__(self, imgs, masks, tfm):
        self.imgs, self.masks, self.tfm = imgs, masks, tfm
    def __len__(self): return len(self.imgs)
    def __getitem__(self, i):
        img  = cv2.cvtColor(cv2.imread(self.imgs[i]), cv2.COLOR_BGR2RGB)
        mask = (cv2.imread(self.masks[i], cv2.IMREAD_GRAYSCALE) > 127).astype(np.uint8)
        aug  = self.tfm(image=img, mask=mask)
        return aug['image'], aug['mask'].float().unsqueeze(0)

# ── Load & filter data ────────────────────────────────────────────────────
all_imgs  = sorted(glob.glob(os.path.join(args.data,'**','*[!_mask].tif'), recursive=True))
all_masks = sorted(glob.glob(os.path.join(args.data,'**','*_mask.tif'),    recursive=True))
print(f"Scanning {len(all_imgs)} images...")

pairs = list(zip(all_imgs, all_masks))
random.seed(42); random.shuffle(pairs)

has_tumor = [(i,m) for i,m in pairs
             if cv2.imread(m, cv2.IMREAD_GRAYSCALE) is not None
             and cv2.imread(m, cv2.IMREAD_GRAYSCALE).max() > 0]

print(f"Has tumor: {len(has_tumor)}/{len(pairs)} (loại bỏ {len(pairs)-len(has_tumor)} no-tumor)")

n_val = int(len(has_tumor) * 0.15)
val_p, train_p = has_tumor[:n_val], has_tumor[n_val:]
ti, tm = zip(*train_p); vi, vm = zip(*val_p)

train_dl = DataLoader(DS(list(ti), list(tm), get_tfm('train')),
                      batch_size=args.batch, shuffle=True, num_workers=2, pin_memory=True, drop_last=True)
val_dl   = DataLoader(DS(list(vi), list(vm), get_tfm('val')),
                      batch_size=args.batch, shuffle=False, num_workers=2, pin_memory=True)
print(f"Train: {len(train_dl)} batches | Val: {len(val_dl)} batches\n")

# ── Model ─────────────────────────────────────────────────────────────────
model = smp.create_model(args.arch, encoder_name=args.encoder,
                         encoder_weights='imagenet', in_channels=3, classes=1, activation=None)

if args.dropout > 0:
    drop2d = nn.Dropout2d(p=args.dropout)
    def drop_hook(m, inp, out):
        if m.training: return drop2d(out)
    for blk in list(model.decoder.blocks)[-2:]:
        blk.conv2.register_forward_hook(drop_hook)
    print(f"Dropout {args.dropout} injected into last 2 decoder blocks")

model = model.to(DEVICE)
print(f"Params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M\n")

# ── Loss ──────────────────────────────────────────────────────────────────
PW = torch.tensor([args.pos_weight], device=DEVICE)

def loss_fn(pred, tgt):
    p = torch.sigmoid(pred).view(-1); t = tgt.view(-1)
    inter = (p*t).sum()
    dice = 1 - (2*inter+1)/(p.sum()+t.sum()+1)
    TP = inter; FP=((1-t)*p).sum(); FN=(t*(1-p)).sum()
    tv = 1 - ((TP+1)/(TP+0.7*FN+0.3*FP+1))**0.75
    bce = F.binary_cross_entropy_with_logits(pred, tgt, pos_weight=PW)
    return 0.5*dice + 0.3*tv + 0.2*bce

def d_score(pred, tgt, t=0.5):
    p=(torch.sigmoid(pred)>t).float().view(-1); tgt=tgt.view(-1)
    return ((2*(p*tgt).sum()+1)/(p.sum()+tgt.sum()+1)).item()

# ── Optimizer + Schedulers ────────────────────────────────────────────────
enc_p = list(model.encoder.parameters())
dec_p = [p for n,p in model.named_parameters() if not n.startswith('encoder')]
opt = torch.optim.AdamW([{'params':enc_p,'lr':args.lr/10},
                          {'params':dec_p,'lr':args.lr}], weight_decay=1e-4)

def warmup_cosine(e, w=5, T=args.epochs):
    if e < w: return (e+1)/w
    return max(0.05, 0.5*(1+np.cos(np.pi*(e-w)/(T-w))))

sched_cos = torch.optim.lr_scheduler.LambdaLR(opt, warmup_cosine)
sched_plt = torch.optim.lr_scheduler.ReduceLROnPlateau(
    opt, mode='max', factor=0.5, patience=8, min_lr=1e-7, verbose=True)
scaler = GradScaler(enabled=torch.cuda.is_available())

# ── Train ─────────────────────────────────────────────────────────────────
best, no_imp = 0.0, 0
os.makedirs(os.path.dirname(args.save), exist_ok=True)

for ep in range(1, args.epochs+1):
    model.train()
    tl = 0.0
    for imgs, masks in tqdm(train_dl, desc=f'Ep {ep:3d}/{args.epochs}', ncols=90):
        imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
        with autocast(enabled=torch.cuda.is_available()):
            loss = loss_fn(model(imgs), masks)
        opt.zero_grad(); scaler.scale(loss).backward()
        scaler.unscale_(opt); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        tl += loss.item()
    tl /= len(train_dl)

    model.eval(); vl, vd = 0.0, 0.0
    with torch.no_grad():
        for imgs, masks in val_dl:
            imgs, masks = imgs.to(DEVICE), masks.to(DEVICE)
            p = model(imgs)
            vl += loss_fn(p, masks).item(); vd += d_score(p, masks)
    vl /= len(val_dl); vd /= len(val_dl)
    sched_cos.step(); sched_plt.step(vd)

    lr = opt.param_groups[1]['lr']
    print(f"Ep {ep:3d} | Loss:{tl:.4f} | VLoss:{vl:.4f} | Gap:{vl-tl:+.4f} | Dice:{vd:.4f} | LR:{lr:.2e}")

    if vd > best:
        best = vd; no_imp = 0
        torch.save({'epoch':ep,'arch':args.arch,'encoder':args.encoder,
                    'model_state_dict':model.state_dict(),'best_dice':best},
                   args.save)
        print(f"  ✓ Saved → {args.save}  Dice={best:.4f}")
    else:
        no_imp += 1
        if no_imp >= 15:
            print(f"\n⏹ Early stop tại epoch {ep}"); break

print(f"\n✅ Done! Best Dice = {best:.4f}  Saved: {args.save}")
