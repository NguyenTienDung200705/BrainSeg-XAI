"""
Grad-CAM compatible với cả UNet thuần và SMP models.
"""
import numpy as np
import torch
import torch.nn.functional as F


class GradCAM:
    def __init__(self, model: torch.nn.Module, target_layer: torch.nn.Module):
        self.model = model
        self.target_layer = target_layer
        self._features = None
        self._grads = None
        self._hooks = []
        self._register()

    def _register(self):
        self._hooks.append(
            self.target_layer.register_forward_hook(self._save_features))
        self._hooks.append(
            self.target_layer.register_full_backward_hook(self._save_grads))

    def _save_features(self, m, inp, out): self._features = out.detach()
    def _save_grads(self, m, gi, go):     self._grads    = go[0].detach()

    def remove_hooks(self):
        for h in self._hooks: h.remove()

    def generate(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model.eval()
        input_tensor = input_tensor.detach().requires_grad_(True)

        output = self.model(input_tensor)
        score  = torch.sigmoid(output).sum()
        self.model.zero_grad()
        score.backward()

        if self._grads is None or self._features is None:
            return np.zeros(input_tensor.shape[-2:], dtype=np.float32)

        weights = self._grads.mean(dim=(2, 3), keepdim=True)
        cam = F.relu((weights * self._features).sum(dim=1, keepdim=True))
        cam = F.interpolate(cam, size=input_tensor.shape[-2:],
                            mode="bilinear", align_corners=False)
        cam = cam.squeeze().cpu().numpy()

        lo, hi = cam.min(), cam.max()
        if hi - lo > 1e-8:
            cam = (cam - lo) / (hi - lo)
        else:
            cam = np.zeros_like(cam)
        return cam.astype(np.float32)


def get_target_layer(model, model_type: str = "vanilla"):
    """
    Trả về layer phù hợp để hook Grad-CAM tùy model type.
    """
    if model_type == "smp":
        # SMP UNet++ / UNet / FPN: hook vào segmentation_head conv đầu tiên
        try:
            return model.segmentation_head[0]
        except Exception:
            pass
        # Fallback: decoder blocks cuối
        try:
            blocks = list(model.decoder.blocks)
            return blocks[-1].conv2[0]
        except Exception:
            pass
        # Last resort: bất kỳ Conv2d cuối
        convs = [m for m in model.modules() if isinstance(m, torch.nn.Conv2d)]
        return convs[-2] if len(convs) >= 2 else convs[-1]
    else:
        # Vanilla UNet
        return model.up4.conv.conv[-3]
