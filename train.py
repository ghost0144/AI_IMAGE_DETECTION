import os
import torch
import torch.nn as nn
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from tqdm import tqdm

from dataset import AIGCDataset
from transforms import get_train_transforms, get_test_transforms
from model import DinoV3Detector


# =========================
# 配置
# =========================

data_root = "/data/hdd3/pw/stable_diffusion_v_1_5/imagenet_ai_0424_sdv5"

ckpt_path = "/data/hdd3/pw/work/checkpoint/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

epochs = 10
batch_size = 8
lr = 1e-4

save_path = "checkpoint.pth"


# =========================
# 初始化DDP
# =========================

def setup_ddp():

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank)

    return local_rank, device


# =========================
# 主函数
# =========================

def main():

    torch.backends.cudnn.benchmark = True

    local_rank, device = setup_ddp()

    rank = dist.get_rank()


    # =========================
    # transforms
    # =========================

    train_transform = get_train_transforms(224)
    val_transform = get_test_transforms(224)


    # =========================
    # dataset
    # =========================

    train_dataset = AIGCDataset(
        data_root,
        "train",
        transform=train_transform,
        max_samples=2000
    )

    val_dataset = AIGCDataset(
        data_root,
        "val",
        transform=val_transform,
        max_samples=500
    )


    # =========================
    # sampler
    # =========================

    train_sampler = DistributedSampler(train_dataset)

    val_sampler = DistributedSampler(val_dataset, shuffle=False)


    # =========================
    # dataloader
    # =========================

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=4,
        pin_memory=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        pin_memory=True
    )


    # =========================
    # model
    # =========================

    model = DinoV3Detector(ckpt_path).to(device)
    
    for p in model.backbone.parameters():
        p.requires_grad = False

    model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=True
    )


    # =========================
    # loss optimizer
    # =========================

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.module.classifier.parameters(),
        lr=lr
    )


    # =========================
    # AMP
    # =========================

    scaler = torch.amp.GradScaler("cuda")


    best_acc = 0


    # =========================
    # training loop
    # =========================

    for epoch in range(epochs):

        train_sampler.set_epoch(epoch)

        model.train()

        total_loss = 0
        correct = 0
        total = 0

        if rank == 0:
            pbar = tqdm(train_loader)
        else:
            pbar = train_loader


        for img, label in pbar:

            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)


            with torch.amp.autocast("cuda"):

                pred = model(img)

                loss = criterion(pred, label)


            optimizer.zero_grad()

            scaler.scale(loss).backward()

            scaler.step(optimizer)

            scaler.update()


            total_loss += loss.item()

            pred_label = pred.argmax(dim=1)

            correct += (pred_label == label).sum().item()

            total += label.size(0)


        train_acc = correct / total


        if rank == 0:

            print(f"\nEpoch {epoch} Train Loss {total_loss:.4f} Acc {train_acc:.4f}")


        # =========================
        # validation
        # =========================

        model.eval()

        correct = 0
        total = 0

        with torch.no_grad():

            for img, label in val_loader:

                img = img.to(device, non_blocking=True)
                label = label.to(device, non_blocking=True)

                pred = model(img)

                pred_label = pred.argmax(dim=1)

                correct += (pred_label == label).sum().item()

                total += label.size(0)


        val_acc = correct / total


        if rank == 0:

            print(f"Epoch {epoch} Val Acc {val_acc:.4f}")


            if val_acc > best_acc:

                best_acc = val_acc

                torch.save(
                    model.module.state_dict(),
                    save_path
                )

                print("Saved best model")


# =========================

if __name__ == "__main__":
    main()