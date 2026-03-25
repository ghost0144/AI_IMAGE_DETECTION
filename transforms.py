import torch
import random
import numpy as np
import cv2
from PIL import Image
from torchvision import transforms


# =========================
# JPEG Compression
# =========================
class RandomJPEGCompression:
    def __init__(self, quality_lower=30, quality_upper=90, p=0.5):
        self.quality_lower = quality_lower
        self.quality_upper = quality_upper
        self.p = p

    def __call__(self, img):
        if random.random() > self.p:
            return img

        quality = random.randint(self.quality_lower, self.quality_upper)

        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        img_np = np.array(img)

        result, encimg = cv2.imencode('.jpg', img_np, encode_param)
        decimg = cv2.imdecode(encimg, 1)

        return Image.fromarray(decimg)


# =========================
# Gaussian Noise
# =========================
class RandomGaussianNoise:
    def __init__(self, mean=0, std=10, p=0.5):
        self.mean = mean
        self.std = std
        self.p = p

    def __call__(self, img):

        if random.random() > self.p:
            return img

        img_np = np.array(img).astype(np.float32)

        noise = np.random.normal(self.mean, self.std, img_np.shape)

        noisy = img_np + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)

        return Image.fromarray(noisy)


# =========================
# Pepper Noise
# =========================
class RandomPepperNoise:
    def __init__(self, noise_ratio=0.02, p=0.5):
        self.noise_ratio = noise_ratio
        self.p = p

    def __call__(self, img):

        if random.random() > self.p:
            return img

        img_np = np.array(img)

        h, w, c = img_np.shape
        num_noise = int(h * w * self.noise_ratio)

        for _ in range(num_noise):
            x = random.randint(0, h - 1)
            y = random.randint(0, w - 1)
            img_np[x, y] = 0

        return Image.fromarray(img_np)


# =========================
# Motion Blur
# =========================
class MotionBlur:
    def __init__(self, kernel_size=5, p=0.5):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, img):

        if random.random() > self.p:
            return img

        img_np = np.array(img)

        kernel = np.zeros((self.kernel_size, self.kernel_size))
        kernel[int((self.kernel_size - 1) / 2), :] = np.ones(self.kernel_size)
        kernel = kernel / self.kernel_size

        blurred = cv2.filter2D(img_np, -1, kernel)

        return Image.fromarray(blurred)


# =========================
# Median Blur
# =========================
class MedianBlur:
    def __init__(self, kernel_size=3, p=0.5):
        self.kernel_size = kernel_size
        self.p = p

    def __call__(self, img):

        if random.random() > self.p:
            return img

        img_np = np.array(img)

        blurred = cv2.medianBlur(img_np, self.kernel_size)

        return Image.fromarray(blurred)


# =========================
# Sharpen
# =========================
class RandomSharpen:
    def __init__(self, factor=(1.0, 2.0), p=0.5):
        self.factor = factor
        self.p = p

    def __call__(self, img):

        if random.random() > self.p:
            return img

        img_np = np.array(img)

        alpha = random.uniform(self.factor[0], self.factor[1])

        kernel = np.array([
            [0, -1, 0],
            [-1, 5 + alpha, -1],
            [0, -1, 0]
        ])

        sharp = cv2.filter2D(img_np, -1, kernel)

        sharp = np.clip(sharp, 0, 255).astype(np.uint8)

        return Image.fromarray(sharp)


# =========================
# 主增强函数
# =========================
def get_train_transforms(img_size=224):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([

        transforms.RandomCrop(img_size, pad_if_needed=True),

        transforms.RandomApply([
            transforms.RandomChoice([

                RandomJPEGCompression(30, 90, p=1.0),
               #RandomGaussianNoise(std=10, p=1.0),
               #RandomPepperNoise(noise_ratio=0.02, p=1.0),
               #MotionBlur(kernel_size=5, p=1.0),
               #MedianBlur(kernel_size=3, p=1.0),
               #RandomSharpen((1.0, 2.0), p=1.0),
               transforms.RandomHorizontalFlip(),
               transforms.ColorJitter(0.2, 0.2, 0.2, 0.05),
               transforms.RandomApply([transforms.GaussianBlur(3)], p=0.3),
            ])
        ],p=0.5),

        transforms.ToTensor(),

        transforms.Normalize(mean, std)

    ])

    return transform


# =========================
# 测试集增强
# =========================
def get_test_transforms(img_size=224):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    transform = transforms.Compose([

        transforms.CenterCrop(img_size),

        transforms.ToTensor(),

        transforms.Normalize(mean, std)

    ])

    return transform