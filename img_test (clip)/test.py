import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

import argparse
import random
from statistics import mean

import torch
from tqdm import tqdm
import numpy as np
from dataset import build_episode_loader
from model import CLIPGCLPModel

def setseed():
    seed = int(np.random.uniform(0, 1) * 10000000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed

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
    parser.add_argument("--model_name", type=str, default="clip-vit-base-patch32")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_plain-ubpf-5.pt")

    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"])
    parser.add_argument("--mode", type=str, default="plain", choices=["plain", "add", "gclp"])
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=1)
    parser.add_argument("--test_episodes", type=int, default=1000)
    parser.add_argument("--topk_patches", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")

    parser.add_argument("--save_logits", action="store_true")
    parser.add_argument("--save_path", type=str, default="./outputs/test_results.pt")
    return parser.parse_args()


def evaluate(model, loader, mode):
    model.eval()
    losses, accs = [], []

    all_targets = []
    all_preds = []
    all_logits = []
    all_class_names = []
    all_class_ids = []

    pbar = tqdm(loader, desc="test")
    with torch.no_grad():
        for episode in pbar:
            out = model.compute_episode_outputs(episode, mode=mode)

            loss_val = out["loss"].item()
            acc_val = out["acc"].item()

            losses.append(loss_val)
            accs.append(acc_val)

            pbar.set_postfix(loss=f"{loss_val:.4f}", acc=f"{acc_val:.4f}")

            all_targets.append(out["targets"].cpu())
            all_preds.append(out["pred"].cpu())
            all_logits.append(out["logits"].cpu())

            if hasattr(episode, "class_names"):
                all_class_names.append(episode.class_names)
            if hasattr(episode, "class_ids"):
                all_class_ids.append(episode.class_ids)

    results = {
        "mean_loss": mean(losses) if len(losses) > 0 else 0.0,
        "mean_acc": mean(accs) if len(accs) > 0 else 0.0,
        "targets": torch.cat(all_targets, dim=0) if len(all_targets) > 0 else None,
        "preds": torch.cat(all_preds, dim=0) if len(all_preds) > 0 else None,
        "logits": torch.cat(all_logits, dim=0) if len(all_logits) > 0 else None,
        "class_names": all_class_names,
        "class_ids": all_class_ids,
    }
    return results

# 固定随机种子，确保每次结果都一样
def setseed():
    seed = int(np.random.uniform(0, 1) * 10000000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    return seed

def main(seed):
    seed_torch(seed)
    args = parse_args()


    if not os.path.exists(args.model_name):
        raise FileNotFoundError(
            f"Local CLIP path not found: {args.model_name}\n"
            f"Please set --model_name to your local clip-vit-base-patch32 folder."
        )

    if not os.path.exists(args.checkpoint):
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    loader = build_episode_loader(
        root=args.data_root,
        split=args.split,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        n_episodes=args.test_episodes,
        num_workers=args.num_workers,
    )

    model = CLIPGCLPModel(
        model_name=args.model_name,
        topk_patches=args.topk_patches,
        use_adapter=True,
    ).to(args.device)

    ckpt = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    print(f"Loaded checkpoint from: {args.checkpoint}")

    if "epoch" in ckpt:
        print(f"Checkpoint epoch: {ckpt['epoch']}")
    if "best_val_acc" in ckpt:
        print(f"Checkpoint best_val_acc: {ckpt['best_val_acc']:.4f}")

    print("Test config:")
    for k, v in vars(args).items():
        print(f"  {k}: {v}")

    results = evaluate(model, loader, args.mode)

    print("\nTest finished.")
    print(f"Mean loss: {results['mean_loss']:.4f}")
    print(f"Mean acc : {results['mean_acc']:.4f}")

    if args.save_logits:
        save_dir = os.path.dirname(args.save_path)
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)

        save_obj = {
            "checkpoint": args.checkpoint,
            "split": args.split,
            "mode": args.mode,
            "mean_loss": results["mean_loss"],
            "mean_acc": results["mean_acc"],
            "targets": results["targets"],
            "preds": results["preds"],
            "logits": results["logits"],
            "class_names": results["class_names"],
            "class_ids": results["class_ids"],
            "args": vars(args),
        }
        torch.save(save_obj, args.save_path)
        print(f"Saved test results to: {args.save_path}")


if __name__ == "__main__":
    seed = setseed()
    main(seed)