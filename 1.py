import torch
import timm
from PIL import Image
from torchvision import transforms

# 设备
device = "cuda" if torch.cuda.is_available() else "cpu"

# 1 创建模型
model = timm.create_model(
    "vit_base_patch16_224",
    pretrained=False
)

# 2 读取权重
ckpt = torch.load(
    "/data/hdd3/pw/work/checkpoint/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    map_location="cpu",
    weights_only=False
)

# 3 处理checkpoint结构
if "model" in ckpt:
    ckpt = ckpt["model"]
if "student" in ckpt:
    ckpt = ckpt["student"]

# 4 去掉 module 前缀
ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

# 5 只加载匹配参数（避免报错）
model_dict = model.state_dict()

filtered = {
    k: v for k, v in ckpt.items()
    if k in model_dict and v.shape == model_dict[k].shape
}

model_dict.update(filtered)
model.load_state_dict(model_dict)

print("Loaded parameters:", len(filtered))

# 6 放到GPU
model = model.to(device)
model.eval()

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485,0.456,0.406],
        [0.229,0.224,0.225]
    )
])

# 读取图片
img = Image.open("/data/hdd3/pw/test.JPEG").convert("RGB")
img = transform(img).unsqueeze(0).to(device)

# 提取特征
with torch.no_grad():
    feat = model.forward_features(img)

# CLS token
feature = feat[:,0]

print("Feature shape:", feature.shape)
print(feature)