import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, Sampler, DataLoader


class MiniImageNetCSVDataset(Dataset):
    def __init__(self, root: str, split: str):
        self.root = root

        # 兼容两种目录:
        # 1) root/images/train
        # 2) root/train
        images_dir_1 = os.path.join(root, "images")
        images_dir_2 = root
        if os.path.exists(os.path.join(images_dir_1, split)):
            self.images_dir = images_dir_1
        elif os.path.exists(os.path.join(images_dir_2, split)):
            self.images_dir = images_dir_2
        else:
            raise FileNotFoundError(
                f"Cannot find split folder: {split} under either "
                f"{images_dir_1} or {images_dir_2}"
            )

        self.csv_file = os.path.join(root, "split", f"{split}.csv")
        if not os.path.exists(self.csv_file):
            raise FileNotFoundError(f"Split file not found: {self.csv_file}")

        self.df = pd.read_csv(self.csv_file)

        required_cols = {"filename", "label", "label_text"}
        missing = required_cols - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in {self.csv_file}: {missing}")

        # 去掉空值和奇怪空格
        self.df["label"] = self.df["label"].astype(str).str.strip()
        self.df["label_text"] = self.df["label_text"].astype(str).str.strip()
        self.df["filename"] = self.df["filename"].astype(str).str.strip()

        # 原始类别编号（例如 synset id）
        labels = sorted(self.df["label"].unique().tolist())
        self.label2id = {x: i for i, x in enumerate(labels)}
        self.id2label = {i: x for x, i in self.label2id.items()}

        # 关键：编号 -> 标签词
        self.label2text = (
            self.df.groupby("label")["label_text"]
            .first()
            .to_dict()
        )
        self.id2text = {
            self.label2id[label_name]: text_label
            for label_name, text_label in self.label2text.items()
        }

        self.samples = []
        self.class_to_indices: Dict[int, List[int]] = {}

        for _, row in self.df.iterrows():
            label_name = row["label"]          # 原始类别名 / synset id
            label_text = row["label_text"]     # 可读标签词
            label_id = self.label2id[label_name]

            image_path = os.path.join(self.images_dir, row["filename"])
            if not os.path.exists(image_path):
                raise FileNotFoundError(f"Image file not found: {image_path}")

            sample = {
                "image_path": image_path,
                "label_name": label_name,   # 例如 n01532829
                "label_id": label_id,       # 数据集内部编号 0,1,2,...
                "text_label": label_text,   # 例如 house finch
            }

            idx = len(self.samples)
            self.samples.append(sample)
            self.class_to_indices.setdefault(label_id, []).append(idx)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        image = Image.open(item["image_path"]).convert("RGB")
        return {
            "image": image,
            "label_id": item["label_id"],
            "label_name": item["label_name"],   # 原始类别编号
            "text_label": item["text_label"],   # 可读标签词
        }


class EpisodeBatchSampler(Sampler):
    def __init__(
        self,
        class_to_indices: Dict[int, List[int]],
        n_way: int,
        k_shot: int,
        q_query: int,
        n_episodes: int
    ):
        self.class_to_indices = class_to_indices
        self.classes = list(class_to_indices.keys())
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        self.n_episodes = n_episodes

    def __len__(self):
        return self.n_episodes

    def __iter__(self):
        for _ in range(self.n_episodes):
            chosen_classes = random.sample(self.classes, self.n_way)
            batch_indices = []

            for c in chosen_classes:
                candidates = self.class_to_indices[c]
                need = self.k_shot + self.q_query
                if len(candidates) < need:
                    raise ValueError(
                        f"Class {c} has only {len(candidates)} samples, but need {need}."
                    )
                batch_indices.extend(random.sample(candidates, need))

            yield batch_indices


@dataclass
class EpisodeBatch:
    support_images: List[Image.Image]
    support_labels: torch.Tensor   # [N, K]
    support_texts: List[str]       # [N]，给文本编码器用
    query_images: List[Image.Image]
    query_labels: torch.Tensor     # [N, Q]

    class_ids: List[str]           # [N]，原始类别编号，如 n01532829
    class_names: List[str]         # [N]，可读标签词，如 house finch


class EpisodeCollate:
    def __init__(self, n_way: int, k_shot: int, q_query: int):
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query

    def __call__(self, batch):
        grouped = {}
        class_id_name_of_id = {}
        text_label_of_id = {}

        for item in batch:
            y = item["label_id"]
            grouped.setdefault(y, []).append(item["image"])
            class_id_name_of_id[y] = item["label_name"]   # 原始 synset id
            text_label_of_id[y] = item["text_label"]      # 可读标签词

        if len(grouped) != self.n_way:
            raise ValueError(f"Expected {self.n_way} classes, got {len(grouped)}")

        class_ids_sorted = sorted(grouped.keys())

        support_images, query_images = [], []
        support_labels, query_labels = [], []
        support_texts = []
        class_ids = []
        class_names = []

        for new_y, old_y in enumerate(class_ids_sorted):
            imgs = grouped[old_y]
            if len(imgs) != self.k_shot + self.q_query:
                raise ValueError(
                    f"Class {old_y} episode samples = {len(imgs)}, "
                    f"expected {self.k_shot + self.q_query}"
                )

            random.shuffle(imgs)
            s_imgs = imgs[: self.k_shot]
            q_imgs = imgs[self.k_shot: self.k_shot + self.q_query]

            support_images.extend(s_imgs)
            query_images.extend(q_imgs)

            support_labels.append([new_y] * self.k_shot)
            query_labels.append([new_y] * self.q_query)

            # 文本统一使用可读标签词
            readable_name = text_label_of_id[old_y]

            support_texts.append(readable_name)
            class_names.append(readable_name)                 # 关键改动：不再放 synset id
            class_ids.append(class_id_name_of_id[old_y])      # 单独保留原始编号

        return EpisodeBatch(
            support_images=support_images,
            support_labels=torch.tensor(support_labels, dtype=torch.long),
            support_texts=support_texts,
            query_images=query_images,
            query_labels=torch.tensor(query_labels, dtype=torch.long),
            class_ids=class_ids,
            class_names=class_names,
        )


def build_episode_loader(
    root: str,
    split: str,
    n_way: int,
    k_shot: int,
    q_query: int,
    n_episodes: int,
    num_workers: int = 4
):
    dataset = MiniImageNetCSVDataset(root=root, split=split)

    sampler = EpisodeBatchSampler(
        class_to_indices=dataset.class_to_indices,
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
        n_episodes=n_episodes,
    )

    collate = EpisodeCollate(
        n_way=n_way,
        k_shot=k_shot,
        q_query=q_query,
    )

    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate,
        num_workers=num_workers,
        pin_memory=True,
    )
    return loader