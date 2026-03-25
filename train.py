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

train_data_root = "/data/hdd3/pw/stable_diffusion_v_1_4/imagenet_ai_0419_sdv4"
test_data_root = "/data/hdd3/pw/AIGIBench"

ckpt_path = "/data/hdd3/pw/work/checkpoint/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"

epochs = 20
batch_size = 8
lr = 1e-4

save_path = "checkpoint.pth"
resume = False   # 是否从checkpoint恢复

# =========================
# 初始化DDP
# =========================

def setup_ddp():

    dist.init_process_group(backend="nccl")

    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)

    device = torch.device("cuda", local_rank)

    return local_rank, device

def evaluate(model, loader, device):

    model.eval()

    correct = torch.tensor(0, device=device)
    total = torch.tensor(0, device=device)

    with torch.no_grad():

        for img, label in loader:

            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            pred = model(img)

            pred_label = pred.argmax(dim=1)

            correct += (pred_label == label).sum()
            total += label.size(0)

    dist.all_reduce(correct)
    dist.all_reduce(total)

    acc = (correct / total).item()

    return acc

def evaluate_single(model, loader, device):

    model.eval()

    correct = 0
    total = 0

    with torch.no_grad():
        for img, label in loader:

            img = img.to(device)
            label = label.to(device)

            pred = model(img)
            pred = pred.argmax(1)

            correct += (pred == label).sum().item()
            total += label.size(0)

    return correct / total


# =========================
# 主函数
# =========================

def main():

    from seed import set_seed
    set_seed(42)

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
        train_data_root,
        "train",
        transform=train_transform,
        max_samples=10000
    )

    val_dataset = AIGCDataset(
        train_data_root,
        "val",
        transform=val_transform,
        max_samples=10000
    )

    test_dataset = AIGCDataset(
        test_data_root,
        "test",  
        transform=val_transform,
        max_samples=10000   # 可调
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
        persistent_workers=True,
        pin_memory=True,
        drop_last=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=4,
        persistent_workers=True,
        pin_memory=True,
        drop_last=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=32,  
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        drop_last=False
    )

    # =========================
    # model
    # =========================

    model = DinoV3Detector(ckpt_path).to(device)
    
    if resume and os.path.exists(save_path):
        if rank == 0:
            print(f"Loading checkpoint from {save_path}")

        # 先加载到CPU，避免多卡冲突
        checkpoint = torch.load(save_path, map_location="cpu")

        model.load_state_dict(checkpoint, strict=True)
   
    model = DDP(
    model,
    device_ids=[local_rank],
    output_device=local_rank,
    find_unused_parameters=False
    )


    # =========================
    # loss optimizer
    # =========================

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
    filter(lambda p: p.requires_grad, model.parameters()),
    lr=lr,
    weight_decay=1e-4
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
        correct = torch.tensor(0, device=device)
        total = torch.tensor(0, device=device)

#       if rank == 0:
#           print(model.module.backbone.backbone.blocks[0].attn.qkv)

        if rank == 0:
            pbar = tqdm(train_loader)
            
        if rank == 0:
            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total_params = sum(p.numel() for p in model.parameters())
            print("Trainable params:", trainable)
            print("Total params:", total_params)

        

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

            correct += (pred_label == label).sum()

            total += label.size(0)


        train_loss = total_loss / len(train_loader)


        dist.all_reduce(correct)
        dist.all_reduce(total)

        train_acc = (correct / total).item()


        if rank == 0:

            print(f"\nEpoch {epoch} Train Loss {train_loss:.4f} Acc {train_acc:.4f}")

        # =========================
        # validation
        # =========================

        val_acc = evaluate(model, val_loader, device)


        if rank == 0:

            if val_acc > best_acc:

                best_acc = val_acc

                torch.save(
                    model.module.state_dict(),
                    save_path
                )

                print("Saved best model")

        if rank == 0:

            test_acc = evaluate_single(model.module, test_loader, device)

            print(f"Epoch {epoch} Val Acc {val_acc:.4f} | Test Acc {test_acc:.4f}")
# =========================

if __name__ == "__main__":
    main()