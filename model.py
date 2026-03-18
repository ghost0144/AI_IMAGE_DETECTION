import torch
import torch.nn as nn
import timm
import math


# =========================
# LoRA Linear
# =========================
class LoRALinear(nn.Module):

    def __init__(self, linear, r=8, alpha=16):
        super().__init__()

        self.linear = linear

        # 关键：补齐属性（DINOv3会用）
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        in_dim = linear.in_features
        out_dim = linear.out_features

        self.r = r
        self.alpha = alpha
        self.scale = alpha / r

        self.lora_A = nn.Parameter(torch.zeros(r, in_dim))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, r))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # 冻结原始权重
        for p in self.linear.parameters():
            p.requires_grad = False

    def forward(self, x):

        original = self.linear(x)

        lora = (x @ self.lora_A.T) @ self.lora_B.T

        return original + self.scale * lora


# =========================
# DINOv3 Backbone
# =========================
class DinoV3Backbone(nn.Module):

    def __init__(self, ckpt_path, lora_r=8):
        super().__init__()

        # 🔥 改这里：用官方模型
        repo_path = "/data/hdd3/pw/dinov3-main"  # 你自己改路径

        self.backbone = torch.hub.load(
            repo_path,
            "dinov3_vitb16",
            source="local",
            weights=ckpt_path
        )

        print("Loaded official DINOv3")

        # =========================
        # 注入 LoRA 到 qkv
        # =========================
        for block in self.backbone.blocks:

            block.attn.qkv = LoRALinear(
                block.attn.qkv,
                r=lora_r,
                alpha=2 * lora_r
            )

        print("LoRA injected into attention qkv")

        # =========================
        # 冻结 backbone 原始参数
        # =========================
        for name, p in self.backbone.named_parameters():

            if "lora_" not in name:
                p.requires_grad = False

    def forward(self, x):

        feat = self.backbone.forward_features(x)

        cls_token = feat["x_norm_clstoken"] 

        return cls_token


# =========================
# Detector
# =========================
class DinoV3Detector(nn.Module):

    def __init__(self, ckpt_path, num_classes=2):
        super().__init__()

        self.backbone = DinoV3Backbone(ckpt_path)

        self.classifier = nn.Sequential(
            nn.Linear(768,512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512,2)
        )

    def forward(self, x):

        feat = self.backbone(x)

        out = self.classifier(feat)

        return out