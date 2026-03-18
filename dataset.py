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
        classes = sorted(os.listdir(split_dir))

        for cls in classes:

            cls_dir = os.path.join(split_dir, cls)

            if not os.path.isdir(cls_dir):
                continue

            real_dir = os.path.join(cls_dir, "0_real")
            fake_dir = os.path.join(cls_dir, "1_fake")

            # real images
            if os.path.exists(real_dir):

                for f in os.listdir(real_dir):

                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):

                        path = os.path.join(real_dir, f)

                        real_samples.append((path, 0))

            # fake images
            if os.path.exists(fake_dir):

                for f in os.listdir(fake_dir):

                    if f.lower().endswith((".png", ".jpg", ".jpeg", ".webp")):

                        path = os.path.join(fake_dir, f)

                        fake_samples.append((path, 1))


        # =========================
        # Step1: max_samples 控制（先按类别）
        # =========================

        if max_samples is not None:

            half = max_samples // 2

            real_samples = real_samples[:half]
            fake_samples = fake_samples[:half]


        # =========================
        # Step2: balance
        # =========================

        min_len = min(len(real_samples), len(fake_samples))

        real_samples = real_samples[:min_len]
        fake_samples = fake_samples[:min_len]


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

        img = Image.open(path).convert("RGB")

        img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)

        return img, label