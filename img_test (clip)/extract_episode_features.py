import argparse
import os

import torch

from dataset import build_episode_loader
from model import CLIPGCLPModel


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", type=str, default="./data")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"])

    # 这里不要再写 huggingface 名字，直接写本地路径
    parser.add_argument("--model_name", type=str, default="clip-vit-base-patch32")

    parser.add_argument("--mode", type=str, default="gclp", choices=["plain", "add", "gclp"])
    parser.add_argument("--n_way", type=int, default=5)
    parser.add_argument("--k_shot", type=int, default=1)
    parser.add_argument("--q_query", type=int, default=1)
    parser.add_argument("--topk_patches", type=int, default=8)
    parser.add_argument("--checkpoint", type=str, default="")
    parser.add_argument("--out_path", type=str, default="./outputs/episode_features.pt")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def main():
    args = parse_args()

    # 强制 transformers / huggingface 走离线
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"

    # 检查本地模型目录是否存在
    if not os.path.exists(args.model_name):
        raise FileNotFoundError(f"Local CLIP path not found: {args.model_name}")

    out_dir = os.path.dirname(args.out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    loader = build_episode_loader(
        root=args.data_root,
        split=args.split,
        n_way=args.n_way,
        k_shot=args.k_shot,
        q_query=args.q_query,
        n_episodes=1,
        num_workers=0,
    )
    episode = next(iter(loader))

    model = CLIPGCLPModel(
        model_name=args.model_name,   # 这里传本地目录
        topk_patches=args.topk_patches,
        use_adapter=True,
    ).to(args.device)

    if args.checkpoint:
        ckpt = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        print(f"Loaded checkpoint from {args.checkpoint}")

    model.eval()
    with torch.no_grad():
        out = model.compute_episode_outputs(episode, mode=args.mode)

    save_obj = {
        "class_names": episode.class_names,
        "support_texts": episode.support_texts,
        "support_labels": episode.support_labels,
        "query_labels": episode.query_labels,
        "targets": out["targets"].cpu(),
        "pred": out["pred"].cpu(),
        "logits": out["logits"].cpu(),
        "support_global": out["support_global"].cpu(),
        "support_local": out["support_local"].cpu(),
        "support_text": out["support_text"].cpu(),
        "query_global": out["query_global"].cpu(),
        "query_local": out["query_local"].cpu(),
        "proto_plain": out["proto_plain"].cpu(),
        "proto_add": out["proto_add"].cpu(),
        "proto_gclp": out["proto_gclp"].cpu(),
        "proto_used": out["proto_used"].cpu(),
        "mode": args.mode,
    }
    torch.save(save_obj, args.out_path)

    print(f"Saved episode features to: {args.out_path}")
    for key, value in save_obj.items():
        if torch.is_tensor(value):
            print(f"{key}: {tuple(value.shape)}")
        else:
            print(f"{key}: {type(value)}")


if __name__ == "__main__":
    main()