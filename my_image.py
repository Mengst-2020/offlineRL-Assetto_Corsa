import os
import cv2
import deepdish as dd
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from tqdm import tqdm

import matplotlib.pyplot as plt


IMAGE_H = 240
IMAGE_W = 320
HISTORY_LEN = 16
ACTION_DIM = 3

BATCH_SIZE = 8          # 根据显存调整
LR = 1e-3
EPOCHS = 100
NUM_WORKERS = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DATA_PATHS = [
    "mydata/monza_image/sac-v1/data/",
    "mydata/barcelona_image/sac-v1/data/",
]

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# def preprocess_image_bgr(image_bgr):
#     """
#     uint8 BGR (240,320,3) -> float32 RGB normalized (3,240,320)
#     """
#     image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
#     image_rgb = image_rgb.astype(np.float32) / 255.0
#     image_rgb = (image_rgb - IMAGENET_MEAN) / IMAGENET_STD
#     image_rgb = np.transpose(image_rgb, (2, 0, 1))
#     return image_rgb

def debug_show_crop(image_bgr, crop_top=0.45, crop_bottom=0.34):
    h, w = image_bgr.shape[:2]
    y0 = int(h * crop_top)
    y1 = int(h * (1.0 - crop_bottom))
    roi = image_bgr[y0:y1, :, :]
    roi_rs = cv2.resize(roi, (IMAGE_W, IMAGE_H), interpolation=cv2.INTER_AREA)

    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1); plt.title("Original"); plt.imshow(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)); plt.axis("off")
    plt.subplot(1,2,2); plt.title("Cropped+Resized"); plt.imshow(cv2.cvtColor(roi_rs, cv2.COLOR_BGR2RGB)); plt.axis("off")
    plt.tight_layout(); plt.show()


def preprocess_image_bgr(
    image_bgr,
    out_h=IMAGE_H,
    out_w=IMAGE_W,
    crop_top=0.45,      # 裁掉上方比例
    crop_bottom=0.34,   # 裁掉下方比例
):
    """
    输入:
        image_bgr: uint8, (H,W,3) BGR
    输出:
        float32, (3,out_h,out_w) RGB normalized for ImageNet
    """
    assert image_bgr is not None, "image_bgr is None"
    h, w = image_bgr.shape[:2]

    # 1) 裁剪 ROI
    y0 = int(h * crop_top)
    y1 = int(h * (1.0 - crop_bottom))
    y0 = max(0, min(y0, h - 1))
    y1 = max(y0 + 1, min(y1, h))
    roi = image_bgr[y0:y1, :, :]  # (h', w, 3)

    # 2) resize 回网络期望输入尺寸
    roi = cv2.resize(roi, (out_w, out_h), interpolation=cv2.INTER_AREA)

    # 3) BGR -> RGB, [0,1]
    image_rgb = cv2.cvtColor(roi, cv2.COLOR_BGR2RGB)
    image_rgb = image_rgb.astype(np.float32) / 255.0

    # 4) ImageNet normalize
    image_rgb = (image_rgb - IMAGENET_MEAN) / IMAGENET_STD

    # 5) HWC -> CHW
    image_chw = np.transpose(image_rgb, (2, 0, 1)).astype(np.float32)
    return image_chw



# =========================================================
# Dataset (lazy & episode-safe)
# =========================================================
class VisionILDataset(Dataset):
    """
    Returns:
        images: [H,3,240,320]
        action: [4]
    """
    def __init__(self, paths, history_len=16):
        self.history_len = history_len
        self.episodes = []
        self.indices = []

        for base_path in paths:
            dataset_all = dd.io.load(os.path.join(base_path, "main_data.hdf5"))

            for epi_key in dataset_all:
                ep = dataset_all[epi_key]
                imgs = ep["observations"]["image"]
                acts = ep["actions"]
                terms = ep["terminations"]

                T = min(
                    imgs.shape[0],
                    acts.shape[0],
                    terms.shape[0]
                )

                self.episodes.append((imgs, acts, terms))

                epi_id = len(self.episodes) - 1
                for t in range(self.history_len - 1, T):
                    if terms[t]:
                        continue
                    self.indices.append((epi_id, t))

        print(f"[Dataset] total samples: {len(self.indices)}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        epi_id, t = self.indices[idx]
        imgs, acts, _ = self.episodes[epi_id]

        img_seq = []
        for k in range(t - self.history_len + 1, t + 1):
            img_seq.append(preprocess_image_bgr(imgs[k]))

        images = np.stack(img_seq, axis=0)      # [H,3,240,320]
        action = acts[t].astype(np.float32)     # [4]

        return (
            torch.from_numpy(images),
            torch.from_numpy(action),
        )


class ResNetEmbed128(nn.Module):
    def __init__(self, freeze_backbone=True):
        super().__init__()
        resnet = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2
        )
        self.backbone = nn.Sequential(*list(resnet.children())[:-1])
        self.proj = nn.Linear(2048, 128)

        if freeze_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, x):
        feat = self.backbone(x).flatten(1)
        return self.proj(feat)


class CausalConv1d(nn.Module):
    def __init__(self, c_in, c_out, k=3, dilation=1):
        super().__init__()
        self.pad = (k - 1) * dilation
        self.conv = nn.Conv1d(c_in, c_out, k, dilation=dilation)

    def forward(self, x):
        x = F.pad(x, (self.pad, 0))
        return self.conv(x)


class TCN(nn.Module):
    def __init__(self, c=128, dilations=(1, 2, 4, 8), k=3):
        super().__init__()
        layers = []
        for d in dilations:
            layers.append(CausalConv1d(c, c, k=k, dilation=d))
            layers.append(nn.ReLU())
        self.net = nn.Sequential(*layers)

        for i, d in enumerate(dilations):
            layers.append(CausalConv1d(c, c, k=k, dilation=d))
            if i != len(dilations) - 1:
                layers.append(nn.ReLU())

    def forward(self, x):
        return self.net(x)

# class TCN(nn.Module):
#     def __init__(self, c=128, dilations=(1, 2, 4, 8), k=3):
#         super().__init__()
#         layers = []
#         for i, d in enumerate(dilations):
#             layers.append(CausalConv1d(c, c, k=k, dilation=d))
#             # 最后一层不加 ReLU
#             if i != len(dilations) - 1:
#                 layers.append(nn.ReLU())
#         self.net = nn.Sequential(*layers)

#     def forward(self, x):
#         return self.net(x)



class VisionStudentPolicy(nn.Module):
    def __init__(self, history_len=16):
        super().__init__()
        self.encoder = ResNetEmbed128(freeze_backbone=True)
        self.tcn = TCN(128)
        self.mlp = nn.Sequential(
            nn.Linear(128, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, ACTION_DIM),
            # nn.Tanh(),
        )

    def forward(self, imgs_seq):
        B, H, C, Hh, Ww = imgs_seq.shape
        imgs = imgs_seq.view(B * H, C, Hh, Ww)
        emb = self.encoder(imgs).view(B, H, 128)
        x = emb.transpose(1, 2)
        y = self.tcn(x)
        last = y[:, :, -1]
        return self.mlp(last)
    
from collections import deque

class VisionPolicyFastRunner:
    def __init__(self, model, history_len, device):
        self.model = model
        self.H = history_len
        self.device = device

        self.encoder = model.encoder
        self.tcn = model.tcn
        self.mlp = model.mlp

        self.embed_buffer = deque(maxlen=self.H)
        self.model.eval()

    @torch.no_grad()
    def reset(self, first_image_bgr):
        self.embed_buffer.clear()

        img = preprocess_image_bgr(first_image_bgr)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)

        emb = self.encoder(img)  # [1,128]

        for _ in range(self.H):
            self.embed_buffer.append(emb)

    @torch.no_grad()
    def act(self, new_image_bgr):
        img = preprocess_image_bgr(new_image_bgr)
        img = torch.from_numpy(img).unsqueeze(0).to(self.device)

        emb = self.encoder(img)      # 只算 1 帧
        self.embed_buffer.append(emb)

        x = torch.stack(list(self.embed_buffer), dim=2)  # [1,128,H]
        y = self.tcn(x)
        z = y[:, :, -1]

        action = self.mlp(z)[0]
        action = action.cpu().numpy()
        action = np.clip(action, -1.0, 1.0)

        return action



def train():
    dataset = VisionILDataset(DATA_PATHS, HISTORY_LEN)
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,   # ← 关键
        pin_memory=False,
        drop_last=True,
    )

    model = VisionStudentPolicy(HISTORY_LEN).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/{EPOCHS}")
        for imgs, acts in pbar:
            imgs = imgs.to(DEVICE)
            acts = acts.to(DEVICE)

            pred = model(imgs)
            loss = F.mse_loss(pred, acts)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")

            avg_loss = total_loss / len(loader)
        print(f"[Epoch {epoch:03d}] loss = {avg_loss:.6f}")

    torch.save(model.state_dict(), "vision_student_policy.pth")
    print("Model saved: vision_student_policy.pth")


def unnormalize_chw_to_rgb01(img_chw):
    """
    img_chw: torch.Tensor or np.ndarray, shape (3,H,W), normalized
    return: np.ndarray shape (H,W,3) in [0,1] RGB
    """
    if torch.is_tensor(img_chw):
        img = img_chw.detach().cpu().numpy()
    else:
        img = img_chw
    img = np.transpose(img, (1, 2, 0))  # HWC
    img = img * IMAGENET_STD + IMAGENET_MEAN
    img = np.clip(img, 0.0, 1.0)
    return img

def overlay_cam_on_rgb(rgb01, cam01, alpha=0.45):
    """
    rgb01: (H,W,3) in [0,1]
    cam01: (H,W) in [0,1]
    """
    heat = (cam01 * 255).astype(np.uint8)
    heat = cv2.applyColorMap(heat, cv2.COLORMAP_JET)  # BGR uint8
    heat = cv2.cvtColor(heat, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    out = (1 - alpha) * rgb01 + alpha * heat
    out = np.clip(out, 0.0, 1.0)
    return out

def find_last_conv2d(module: torch.nn.Module):
    last = None
    for m in module.modules():
        if isinstance(m, torch.nn.Conv2d):
            last = m
    return last


class GradCAMHook:
    """
    Reliable Grad-CAM hook even when backbone params are frozen.
    """
    def __init__(self, target_module: torch.nn.Module):
        self.target_module = target_module
        self.activations = None
        self.gradients = None

        def fwd_hook(module, inp, out):
            # out: [N,C,H,W]
            # IMPORTANT: even if params are frozen, we can force out to require grad
            out = out.requires_grad_(True)

            self.activations = out

            def _tensor_grad_hook(grad):
                self.gradients = grad

            out.register_hook(_tensor_grad_hook)

        self.handle = target_module.register_forward_hook(fwd_hook)

    def close(self):
        self.handle.remove()



def gradcam_for_sequence_action(
    model,
    imgs_seq,                  # torch.Tensor [1,H,3,240,320], normalized
    target_action_dim=0,       # 0/1/2
    target_frame_idx=None,     # 0..H-1; 默认最后一帧
    device="cpu",
    target_layer_name="layer4" # 选 ResNet 的最后卷积块
):
    """
    Returns:
      rgb01: (240,320,3) 原图(反归一化) for target frame
      cam01: (240,320) Grad-CAM in [0,1] for target frame
    """
    model.eval()
    imgs_seq = imgs_seq.to(device)

    H = imgs_seq.shape[1]
    if target_frame_idx is None:
        target_frame_idx = H - 1

    # 1) 找到要 hook 的卷积层：ResNet backbone 内的 layer4（推荐）
    # 你的 encoder.backbone 是 resnet.children()[:-1] 的 Sequential
    # 其中包含 layer4 这块，我们用名字定位更稳。
    target_module = find_last_conv2d(model.encoder.backbone)
    if target_module is None:
        raise RuntimeError("Could not find a Conv2d layer in encoder.backbone for Grad-CAM.")

    hook = GradCAMHook(target_module)

    # 2) 前向 + 对某个动作维度做反传
    model.zero_grad(set_to_none=True)

    # 允许梯度（Grad-CAM 必须）
    with torch.enable_grad():
        out = model(imgs_seq)                 # [1,3]
        score = out[0, target_action_dim]     # scalar
        score.backward(retain_graph=False)

    # 3) 从 hook 取 activations & gradients
    acts = hook.activations   # [B*H, C, h, w]（因为 forward 里 B*H 展平了）
    grads = hook.gradients    # 同形状

    hook.close()

    if acts is None or grads is None:
        raise RuntimeError("Grad-CAM hook did not capture activations/gradients. Check target layer.")

    # 4) 取目标帧对应的 index：B=1，所以 frame_idx 就是 target_frame_idx
    # 注意：forward 内部是 imgs = imgs_seq.view(B*H, ...)
    # B=1 → 展平顺序是 frame0..frameH-1
    idx = target_frame_idx

    a = acts[idx]   # [C,h,w]
    g = grads[idx]  # [C,h,w]

    # 5) Grad-CAM 权重：对空间求均值
    w = g.mean(dim=(1, 2), keepdim=True)   # [C,1,1]
    cam = (w * a).sum(dim=0)               # [h,w]
    cam = F.relu(cam)

    # 6) 归一化到 [0,1] 并上采样到 240x320
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)
    cam = cam.unsqueeze(0).unsqueeze(0)  # [1,1,h,w]
    cam = F.interpolate(cam, size=(IMAGE_H, IMAGE_W), mode="bilinear", align_corners=False)
    cam01 = cam[0, 0].detach().cpu().numpy()

    # 7) 取目标帧的“可视化原图”（反归一化）
    rgb01 = unnormalize_chw_to_rgb01(imgs_seq[0, target_frame_idx])

    return rgb01, cam01

def show_activation_from_dataset(
    model_path="model_image/vision_student_policy.pth",
    data_path="mydata/monza_image/sac-v1/data/",
    episode_key="episode_0",
    t=200,                 # 选时间步
    history_len=16,
    action_dim=0,          # 0/1/2
    frame_idx=None,        # None 表示最后一帧
    device=DEVICE
):
    # 1) 读取一段序列图像
    dataset_all = dd.io.load(os.path.join(data_path, "main_data.hdf5"))
    ep = dataset_all[episode_key]
    imgs = ep["observations"]["image"]

    assert t >= history_len - 1, "t must be >= history_len-1"

    seq = []
    for k in range(t - history_len + 1, t + 1):
        seq.append(preprocess_image_bgr(imgs[k]))  # (3,240,320)
    seq = np.stack(seq, axis=0)                    # (H,3,240,320)

    imgs_seq = torch.from_numpy(seq).unsqueeze(0).float()  # [1,H,3,240,320]

    # 2) 加载模型
    model = VisionStudentPolicy(history_len=history_len).to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # 3) Grad-CAM
    rgb01, cam01 = gradcam_for_sequence_action(
        model=model,
        imgs_seq=imgs_seq,
        target_action_dim=action_dim,
        target_frame_idx=frame_idx,
        device=device,
        target_layer_name="layer4"
    )

    overlay = overlay_cam_on_rgb(rgb01, cam01, alpha=0.45)

    # 4) 展示
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1); plt.title("Frame (RGB)"); plt.imshow(rgb01); plt.axis("off")
    plt.subplot(1, 3, 2); plt.title("Grad-CAM"); plt.imshow(cam01, cmap="jet"); plt.axis("off")
    plt.subplot(1, 3, 3); plt.title("Overlay"); plt.imshow(overlay); plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    show_activation_from_dataset(
    model_path="model_image/vision_student_policy.pth",
    data_path="mydata/monza_image/sac-v1/data/",
    episode_key="episode_0",
    t=250,
    history_len=16,
    action_dim=0,   # 0/1/2 试一下分别对应 steer/throttle/brake（按你数据定义）
    frame_idx=15,   # 看最后一帧（也可 None）
    device=DEVICE)

    # dataset_all = dd.io.load(os.path.join(DATA_PATHS[0], "main_data.hdf5"))
    # img0 = dataset_all["episode_0"]["observations"]["image"][200]
    # debug_show_crop(img0, crop_top=0.45, crop_bottom=0.34)

    # train()
