"""Defines dataset, dataloader, and processing for COCO-style datasets."""

# Imports Python builtins.
import json

# Imports PyTorch packages.
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision.datasets.coco import CocoDetection as TorchvisionCocoDetection

# Imports other packages.
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# Imports local packages.
import coco_tools.transforms as T
from utils.box_ops import box_xyxy_to_xywh
from utils.misc import (
    get_balanced_sampler_weights_by_id,
    nested_collate,
)


def get_transform(aug=False):
    """Converts to tensor and normalizes to ImageNet statistics.

       Optionally performs data augmentation according to Deformable DETR.
    """

    normalize = T.Compose([
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])

    if aug:
        scales = [480, 512, 544, 576, 608, 640, 672, 704, 736, 768, 800]

        return T.Compose([
            T.RandomHorizontalFlip(),
            T.RandomResize(scales, max_size=1333),
            normalize,
        ])
    else:
        return T.Compose([
            T.RandomResize([800], max_size=1333),
            normalize,
        ])

class CocoDetection(TorchvisionCocoDetection):
    """Processing object detection datasets with COCO-style annotations.
    
       Similar to Torchvision CocoDetection with additional
       processing and formatting for weakly supervised training.
    """

    def __init__(self, data_dir, labels_json, task="train"):
        super().__init__(data_dir, labels_json)

        labels = json.load(open(labels_json, "r"))

        self.image_classes_by_id = {
            img["id"]: img["classes"] for img in labels["images"]
        }

        # Sets transform as specified by task.
        aug = True if task == "train" else False
        self.transform = get_transform(aug=aug)

    def __getitem__(self, idx):
        # Gets image and target from dataset.
        img_id = self.ids[idx]
        img = self._load_image(img_id)
        target = self._load_target(img_id)

        # Adds image id as a key to the target dict.
        target = {"image_id": img_id, "annotations": target}

        # Preprocesses targets into dict.
        target = self.prepare(target, *img.size)

        # Applies transformations to image and target.
        img, target = self.transform(img, target)

        return img, target

    def prepare(self, target, width, height):
        """Loads COCO annotations into dict of tensors."""

        # Loads image id and annotation from target.
        image_id = target["image_id"]
        ann = target["annotations"]

        # Removes crowd RLEs (keeps only single-instance bounding boxes).
        ann = [
            obj for obj in ann
            if "iscrowd" not in obj
            or not obj["iscrowd"]
        ]

        # Extracts targets of interest from annotations into lists.
        boxes = [obj["bbox"] for obj in ann]
        area = [obj["area"] for obj in ann]
        iscrowd = [0] * len(ann)

        box_classes = []
        image_classes = []
        if self.image_classes_by_id:
            image_classes = self.image_classes_by_id[image_id]
        if ann and "category_id" in ann[0]:
            box_classes = [obj["category_id"] for obj in ann]

        # If conf is in annotation keys, i.e., this json is from
        # the pseudo-label prediction, extract it.
        confs = torch.ones(len(ann))
        if ann and "conf" in ann[0]:
            confs = torch.stack(torch.tensor([obj["conf"] for obj in ann]))

        # Converts targets to tensors.
        image_id = torch.tensor([image_id])
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        image_classes = torch.tensor(image_classes, dtype=torch.int64)
        box_classes = torch.tensor(box_classes, dtype=torch.int64)
        area = torch.tensor(area, dtype=torch.float32)
        iscrowd = torch.tensor(iscrowd)
        size = torch.as_tensor([int(height), int(width)])

        if boxes.shape[0]:
            # Converts from (x1, y1, w, h) to (x1, y1, x2, y2).
            boxes[:, 2:] += boxes[:, :2]

            # Clamps boxes to image size.
            boxes[:, 0::2].clamp_(min=0., max=width)
            boxes[:, 1::2].clamp_(min=0., max=height)

            # Removes invalid boxes from targets.
            keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
            boxes = boxes[keep]
            if not self.image_classes_by_id:
                image_classes = image_classes[keep]
                box_classes = box_classes[keep]
            area = area[keep]
            iscrowd = iscrowd[keep]

        # Populates targets dict.
        target = {}
        target["image_id"] = image_id
        target["boxes"] = boxes
        target["confs"] = confs
        target["image_labels"] = image_classes
        target["box_labels"] = box_classes
        target["area"] = area
        target["iscrowd"] = iscrowd
        target["orig_size"] = size
        target["size"] = size

        return target

def coco_loader(args, task="train"):
    """Builds a dataloader for COCO-style datasets."""

    # Extracts data and labels location from args based on task.
    data_dir = vars(args)[task + "_imgs_dir"]
    labels_json = vars(args)[task + "_anns"]

    labels = json.load(open(labels_json, "r"))
    if "classes" not in labels["images"][0]:
        # Adds a classes field to each image with its image-level labels.
        print("Updating labels json")
        tmp_dict = {img["id"]: [] for img in labels["images"]}
        for ann in labels["annotations"]:
            tmp_dict[ann["image_id"]].append(ann["category_id"])
        for img_id in tmp_dict.keys():
            tmp_dict[img_id] = sorted(list(set(tmp_dict[img_id])))
        for img in labels["images"]:
            img["classes"] = tmp_dict[img["id"]]
        # Removes images with no annotations.
        labels["images"] = [img for img in labels["images"] if img["classes"]]
        json.dump(labels, open(labels_json, "w"))
        print("Updated labels json.")

    dataset = CocoDetection(data_dir, labels_json, task=task)

    # Initializes a balanced random sampler for single-GPU training only.
    sampler = None
    shuffle = True if task == "train" else False
    if task == "train" and args.sampler:
        if args.gpus == 1:
            shuffle = None
            weights_by_id = get_balanced_sampler_weights_by_id(labels, offset)
            weights_by_idx = [weights_by_id[img_id] for img_id in dataset.ids]
            num_imgs = len(weights_by_idx)
            sampler = WeightedRandomSampler(weights_by_idx, num_imgs)
        else:
            raise NotImplementedError(
                "Balanced random sampler is not"
                " implemented for multi-GPU training."
            )

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        collate_fn=nested_collate,
        num_workers=args.workers,
        pin_memory=True,
        sampler=sampler,
        shuffle=shuffle,
    )

    return loader

def prepare_coco_results(results):
    """Loads model results into COCO dict for evaluation."""

    def coco_dict(img_id, box, conf, pred):
        return {
            "image_id": img_id,
            "bbox": box,
            "score": conf,
            "category_id": pred,
        }

    coco_results = []
    for orig_id, result in results.items():
        if not result:
            continue

        # Extracts results from model outputs.
        boxes = result["boxes"]
        confs = result["confs"].tolist()
        preds = result["preds"].tolist()

        # Converts boxes to COCO format.
        boxes = box_xyxy_to_xywh(boxes).tolist()

        result_iter = zip(boxes, confs, preds)

        # Builds COCO-style dict from results.
        coco_result = [
            coco_dict(orig_id, *res)
            for res in result_iter
        ]

        # Adds dict to list of all results.
        coco_results.extend(coco_result)

    return coco_results

def coco_evaluate(results, coco_groundtruth):
    """Runs COCO evaluation and returns statistics including mAP."""

    # Processes results into COCO format.
    coco_results = prepare_coco_results(results)

    # Initializes COCO evaluator.
    coco_detections = COCO.loadRes(coco_groundtruth, coco_results)
    coco_evaluator = COCOeval(
        cocoGt=coco_groundtruth,
        cocoDt=coco_detections,
        iouType="bbox",
    )

    # Evaluates COCO results.
    coco_evaluator.evaluate()
    coco_evaluator.accumulate()
    coco_evaluator.summarize()

    stats = coco_evaluator.stats.tolist()

    return stats

