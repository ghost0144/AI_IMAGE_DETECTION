import os
import random
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AIGCDataset(Dataset):

    def __init__(
        self,
        root_dir,
        split="train",
        img_size=224,
        max_samples=None,
        transform=None
    ):

        self.root_dir = root_dir
        self.split = split

        real_samples = []
        fake_samples = []

        split_dir = os.path.join(root_dir, split)

        if not os.path.exists(split_dir):
            raise ValueError(f"{split_dir} does not exist")

        # 遍历类别，例如 car cat chair
        IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".JPEG")

        for root, _, files in os.walk(split_dir):

        # 判断标签
            if "0_real" in root:
                label = 0
            elif "1_fake" in root:
                label = 1
            else:
                continue

            for f in files:
                if f.lower().endswith(IMG_EXTS):

                    path = os.path.join(root, f)

                    if label == 0:
                        real_samples.append((path, 0))
                    else:
                        fake_samples.append((path, 1))


        # =========================
        # Step1: max_samples 控制（先按类别）
        # =========================

        if max_samples is not None:

            half = max_samples // 2

            real_samples = random.sample(real_samples, min(len(real_samples), half))
            fake_samples = random.sample(fake_samples, min(len(fake_samples), half))


        # =========================
        # Step2: balance
        # =========================

        if len(real_samples) > len(fake_samples):
            real_samples = random.sample(real_samples, len(fake_samples))
        else:
            fake_samples = random.sample(fake_samples, len(real_samples))


        # =========================
        # 合并 + shuffle
        # =========================

        self.samples = real_samples + fake_samples

        random.shuffle(self.samples)


        # =========================
        # Transform
        # =========================

        if transform is None:

            self.transform = transforms.Compose([
                transforms.Resize((img_size, img_size)),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225]
                )
            ])

        else:

            self.transform = transform


        print(f"{split} real:", len(real_samples))
        print(f"{split} fake:", len(fake_samples))
        print(f"{split} total:", len(self.samples))


    def __len__(self):

        return len(self.samples)


    def __getitem__(self, idx):

        path, label = self.samples[idx]

        try:
            img = Image.open(path).convert("RGB")
        except:
            return self.__getitem__(random.randint(0, len(self.samples)-1))

        img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)

        return img, label