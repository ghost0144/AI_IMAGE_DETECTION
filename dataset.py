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

        split_dir = root_dir

        if not os.path.exists(split_dir):
            raise ValueError(f"{split_dir} does not exist")

        IMG_EXTS = (".png", ".jpg", ".jpeg", ".webp", ".JPEG")

        # =========================
        # 扫描数据
        # =========================
        for root, _, files in os.walk(split_dir):

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
        #  test：完全固定逻辑
        # =========================
        if split == "test":

            samples = real_samples + fake_samples

            # 固定采样
            if max_samples is not None:
                rng = random.Random(42)
                samples = rng.sample(samples, min(len(samples), max_samples))

            # 固定顺序
            self.samples = sorted(samples)

        # =========================
        # train / val
        # =========================
        else:

            # Step1: max_samples
            if max_samples is not None:
                half = max_samples // 2

                real_samples = random.sample(real_samples, min(len(real_samples), half))
                fake_samples = random.sample(fake_samples, min(len(fake_samples), half))

            # Step2: balance
            min_len = min(len(real_samples), len(fake_samples))

            real_samples = random.sample(real_samples, min_len)
            fake_samples = random.sample(fake_samples, min_len)

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

        print(f"{split} total:", len(self.samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):

        path, label = self.samples[idx]

        try:
            img = Image.open(path).convert("RGB")
        except:
            return self.__getitem__(random.randint(0, len(self.samples) - 1))

        img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)

        return img, label