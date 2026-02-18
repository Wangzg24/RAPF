import torch
import random
import os
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

class MiniImageNetFewShot(Dataset):
    def __init__(self, root, split, N, K, Q, episodes):
        super().__init__()
        self.root = os.path.join(root, split)
        self.N = N
        self.K = K
        self.Q = Q
        self.episodes = episodes

        self.transform = transforms.Compose([
            transforms.Resize((84,84)),
            transforms.ToTensor(),
        ])

        self.classes = sorted(os.listdir(self.root))
        self.class_to_images = {}

        for cls in self.classes:
            cls_path = os.path.join(self.root, cls)
            self.class_to_images[cls] = [
                os.path.join(cls_path, img)
                for img in os.listdir(cls_path)
            ]

    def __len__(self):
        return self.episodes

    def __getitem__(self, idx):
        selected_classes = random.sample(self.classes, self.N)

        support = []
        query = []
        labels = []

        for i, cls in enumerate(selected_classes):
            images = random.sample(self.class_to_images[cls], self.K + self.Q)

            support_imgs = images[:self.K]
            query_imgs = images[self.K:]

            support.append(torch.stack([
                self.transform(Image.open(img).convert('RGB'))
                for img in support_imgs
            ]))

            query.append(torch.stack([
                self.transform(Image.open(img).convert('RGB'))
                for img in query_imgs
            ]))

            labels += [i] * self.Q

        support = torch.stack(support)
        query = torch.stack(query)
        labels = torch.tensor(labels)

        return support, query, labels
