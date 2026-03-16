import os
from PIL import Image

import torch
from torch.utils.data import Dataset
from torchvision import transforms


class AIGCDataset(Dataset):

    def __init__(self, root_dir, split="train", img_size=224, max_samples=None, transform=None):

        self.root_dir = root_dir
        self.split = split
        self.transform = transform

        self.samples = []

        ai_dir = os.path.join(root_dir, split, "ai")
        real_dir = os.path.join(root_dir, split, "nature")

        # AI images
        for f in os.listdir(ai_dir):
            if f.endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(ai_dir, f)
                self.samples.append((path, 1))

        # Real images
        for f in os.listdir(real_dir):
            if f.endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(real_dir, f)
                self.samples.append((path, 0))

        if max_samples is not None:
            self.samples = self.samples[:max_samples]

        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                [0.485, 0.456, 0.406],
                [0.229, 0.224, 0.225]
            )
        ])

        print(f"{split} samples:", len(self.samples))


    def __len__(self):
        return len(self.samples)


    def __getitem__(self, idx):

        path, label = self.samples[idx]

        img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        label = torch.tensor(label, dtype=torch.long)

        return img, label