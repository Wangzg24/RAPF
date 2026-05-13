import argparse
import os
os.environ['TOKENIZERS_PARALLELISM']='false'
# 强制离线，避免 transformers / hf 再联网
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import numpy as np

import random
from statistics import mean

import torch
from torch.optim import AdamW
from tqdm import tqdm

from dataset import build_episode_loader
from model import CLIPGCLPModel


def set_seed(seed: int):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_torch(seed=1234):
    random.seed(seed)  # python seed
    os.environ['PYTHONHASHSEED'] = str(
        seed)  # 设置python哈希种子，for certain hash-based operations (e.g., the item order in a set or a dict）。seed为0的时候表示不用这个feature，也可以设置为整数。 有时候需要在终端执行，到脚本实行可能就迟了。
    np.random.seed(
        seed)  # If you or any of the libraries you are using rely on NumPy, 比如Sampling，或者一些augmentation。 哪些是例外可以看https://pytorch.org/docs/stable/notes/randomness.html
    torch.manual_seed(seed)  # 为当前CPU设置随机种子。 pytorch官网倒是说(both CPU and CUDA)
    torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
    torch.cuda.manual_seed_all(seed)  # 使用多块GPU时，均设置随机种子
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True  # 设置为True时，cuDNN使用非确定性算法寻找最高效算法
    torch.backends.cudnn.enabled = True  # pytorch使用CUDANN加速，即使用GPU加速


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")

    # 改成你本地下载好的 CLIP 路径；也可以运行时传入
    parser.add_argument("--model_name", type=str, default="clip-vit-base-patch32")

    parser.add_argument("--mode", type=str, default="plain", choices=["plain", "add", "gclp"])
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=1)
    parser.add_argument("--train_episodes", type=int, default=2000)
    parser.add_argument("--val_episodes", type=int, default=1000)
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--topk_patches", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--save_dir", type=str, default="./checkpoints")
    parser.add_argument("--resume", type=str, default="")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def run_epoch(model, loader, optimizer, mode, train: bool):
    losses, accs = [], []

    if train:
        model.train()
    else:
        model.eval()

    pbar = tqdm(loader, desc="train" if train else "val")
    for episode in pbar:
        if train:
            optimizer.zero_grad()
            out = model.compute_episode_outputs(episode, mode=mode)
            out["loss"].backward()
            optimizer.step()
        else:
            with torch.no_grad():
                out = model.compute_episode_outputs(episode, mode=mode)

        loss_val = out["loss"].item()
        acc_val = out["acc"].item()

        losses.append(loss_val)
        accs.append(acc_val)
        pbar.set_postfix(loss=f"{loss_val:.4f}", acc=f"{acc_val:.4f}")

    if len(losses) == 0:
        return 0.0, 0.0

    return mean(losses), mean(accs)


def main(seed=1234):
    seed_torch(seed)
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)

    if not os.path.exists(args.model_name):
        raise FileNotFoundError(
            f"Local CLIP path not found: {args.model_name}\n"
            f"Please set --model_name to your local clip-vit-base-patch32 folder."
        )

    train_loader = build_episode_loader(
        root=args.data_root,
        split="train",
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        n_episodes=args.train_episodes,
        num_workers=args.num_workers,
    )

    val_loader = build_episode_loader(
        root=args.data_root,
        split="val",
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        n_episodes=args.val_episodes,
        num_workers=args.num_workers,
    )

    model = CLIPGCLPModel(
        model_name=args.model_name,
        topk_patches=args.topk_patches,
        use_adapter=True,
    ).to(args.device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    start_epoch = 1
    best_val_acc = -1.0

    if args.resume:
        if not os.path.exists(args.resume):
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume}")

        ckpt = torch.load(args.resume, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)

        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        if "epoch" in ckpt:
            start_epoch = ckpt["epoch"] + 1

        if "best_val_acc" in ckpt:
            best_val_acc = ckpt["best_val_acc"]

        print(f"Resumed from {args.resume}")
        print(f"Start epoch: {start_epoch}, best_val_acc: {best_val_acc:.4f}")

    best_path = os.path.join(args.save_dir, f"best_{args.mode}-ubpf-5.pt")
    last_path = os.path.join(args.save_dir, f"last_{args.mode}-ubpf-5.pt")

    print("Training config:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    for epoch in range(start_epoch, args.epochs + 1):
        print(f"\nEpoch {epoch}/{args.epochs} | mode={args.mode}")

        train_loss, train_acc = run_epoch(
            model=model,
            loader=train_loader,
            optimizer=optimizer,
            mode=args.mode,
            train=True,
        )

        val_loss, val_acc = run_epoch(
            model=model,
            loader=val_loader,
            optimizer=optimizer,
            mode=args.mode,
            train=False,
        )

        print(
            f"Epoch {epoch}: "
            f"train_loss={train_loss:.4f}, train_acc={train_acc:.4f}, "
            f"val_loss={val_loss:.4f}, val_acc={val_acc:.4f}"
        )

        # 保存 last
        torch.save(
            {
                "epoch": epoch,
                "mode": args.mode,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "best_val_acc": best_val_acc,
                "args": vars(args),
            },
            last_path,
        )

        # 保存 best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "mode": args.mode,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_val_acc": best_val_acc,
                    "args": vars(args),
                },
                best_path,
            )
            print(f"Saved best checkpoint to {best_path}")

    print(f"\nTraining finished. Best val acc = {best_val_acc:.4f}")
    print(f"Best checkpoint: {best_path}")
    print(f"Last checkpoint: {last_path}")


if __name__ == "__main__":
    main(1234)