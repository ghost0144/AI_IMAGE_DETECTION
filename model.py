import torch
import torch.nn as nn
import timm


class DinoV3Backbone(nn.Module):

    def __init__(self, ckpt_path):
        super().__init__()

        # 创建ViT模型
        self.backbone = timm.create_model(
            "vit_base_patch16_224",
            pretrained=False
        )

        # 读取权重
        ckpt = torch.load(
            ckpt_path,
            map_location="cpu",
            weights_only=False
        )

        if "model" in ckpt:
            ckpt = ckpt["model"]

        if "student" in ckpt:
            ckpt = ckpt["student"]

        # 去掉module前缀
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

        # 只加载匹配参数
        model_dict = self.backbone.state_dict()

        filtered = {
            k: v for k, v in ckpt.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        model_dict.update(filtered)
        self.backbone.load_state_dict(model_dict)

        print("Loaded parameters:", len(filtered))


    def forward(self, x):

        feat = self.backbone.forward_features(x)

        cls_token = feat[:, 0]

        return cls_token


class DinoV3Detector(nn.Module):

    def __init__(self, ckpt_path, num_classes=2):
        super().__init__()

        self.backbone = DinoV3Backbone(ckpt_path)

        self.classifier = nn.Linear(768, num_classes)


    def forward(self, x):

        feat = self.backbone(x)

        out = self.classifier(feat)

        return out