"""Simple image loading dataset for inference."""

# Imports Python builtins.
import os
import os.path as osp

# Imports PyTorch packages.
from torch.utils.data import Dataset, DataLoader

# Imports other packages.
from PIL import Image

# Imports local packages.
from .coco import get_transform
from utils.misc import nested_collate


class SimpleImageDataset(Dataset):
    """A simple image loading dataset for inference without ground truth."""

    def __init__(self, imgs_dir, transform):
        self.img_names = sorted(os.listdir(imgs_dir))
        self.img_paths = [
            osp.join(imgs_dir, name) for name in self.img_names
        ]
        self.orig_sizes = [
            Image.open(img).size[::-1] for img in self.img_paths
        ]
        self.transform = transform

    def __len__(self):
        return len(self.img_names)

    def __getitem__(self, idx):
        img_name = self.img_names[idx]
        img_path = self.img_paths[idx]
        orig_size = self.orig_sizes[idx]

        img = Image.open(img_path).convert("RGB")
        img, _ = self.transform(img, None)

        return img, img_name, orig_size

def infer_loader(args):
    """Creates DataLoader for inference without ground truth."""

    transform = get_transform()

    dataset = SimpleImageDataset(args.infer_dir, transform)

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=nested_collate,
        num_workers=args.workers,
        pin_memory=True,
    )

    return loader

