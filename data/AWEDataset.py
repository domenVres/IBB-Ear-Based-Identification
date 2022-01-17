import torch
import os
import numpy as np

from PIL import Image

class AWEDataset(torch.utils.data.Dataset):
    def __init__(self, root, transforms):
        self.root = root
        self.transforms = transforms
        # These are defined in child classes
        self.imgs = []
        self.labels = []

    def __getitem__(self, params):
        # load images, masks and bounding boxes
        img_path = params["img_path"]
        label = params["label"]
        img = Image.open(img_path).convert("RGB")
        img.load()

        # there is only one class
        label = torch.tensor(label)

        if self.transforms is not None:
            img = self.transforms(img)

        return img, label


class AWETrainSet(AWEDataset):
    def __init__(self, root, transforms):
        super().__init__(root, transforms)
        self.imgs = list(sorted(os.listdir(os.path.join(root, "train"))))
        annotations = open(self.root + "/annotations/recognition/ids.csv")

        all_labels = {}
        for line in annotations.readlines():
            img, id = line.split(",")
            if "test" in img: continue

            img = img.lstrip("train/")
            id = int(id.strip())
            # We start indexing by 0, annotations start by 1
            all_labels[img] = id - 1

        self.labels = [all_labels[img] for img in self.imgs]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "train", self.imgs[idx])
        label = self.labels[idx]
        params = {"img_path": img_path, "label": label}

        return super().__getitem__(params)

    def __len__(self):
        return len(self.imgs)

    def get_num_classes(self):
        return(len(np.unique(self.labels)))


class AWEValSet(AWEDataset):
    def __init__(self, root, transforms):
        super().__init__(root, transforms)
        self.imgs = list(sorted(os.listdir(os.path.join(root, "val"))))
        annotations = open(self.root + "/annotations/recognition/ids.csv")

        all_labels = {}
        for line in annotations.readlines():
            img, id = line.split(",")
            if "test" in img: continue

            img = img.lstrip("train/")
            id = int(id.strip())
            # We start indexing by 0, annotations start by 1
            all_labels[img] = id-1

        self.labels = [all_labels[img] for img in self.imgs]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "val", self.imgs[idx])
        label = self.labels[idx]
        params = {"img_path": img_path, "label": label}

        return super().__getitem__(params)

    def __len__(self):
        return len(self.imgs)


class AWETestSet(AWEDataset):
    def __init__(self, root, transforms):
        super().__init__(root, transforms)
        self.imgs = list(sorted(os.listdir(os.path.join(root, "test"))))
        annotations = open(self.root + "/annotations/recognition/ids.csv")

        all_labels = {}
        for line in annotations.readlines():
            img, id = line.split(",")
            if "train" in img: continue

            img = img.lstrip("test/")
            id = int(id.strip())
            # We start indexing by 0, annotations start by 1
            all_labels[img] = id - 1

        self.labels = [all_labels[img] for img in self.imgs]

    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "test", self.imgs[idx])
        label = self.labels[idx]
        params = {"img_path": img_path, "label": label}

        return super().__getitem__(params)

    def __len__(self):
        return len(self.imgs)
