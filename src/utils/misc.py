"""Miscellaneous utility functions."""

# Imports Python builtins.
from copy import deepcopy
import os.path as osp
from typing import Optional, List

# Imports PyTorch packages.
import torch
from torch import nn
from torch import Tensor

# Imports other packages.
import numpy as np
from PIL import Image, ImageDraw, ImageFont


class NestedTensor(object):
    """Class for collection of Tensors of different sizes. From DETR."""

    def __init__(self, tensors, mask: Optional[Tensor]):
        self.tensors = tensors
        self.mask = mask

    def to(self, device, non_blocking=False):
        cast_tensor = self.tensors.to(device, non_blocking=non_blocking)
        mask = self.mask
        if mask is not None:
            assert mask is not None
            cast_mask = mask.to(device, non_blocking=non_blocking)
        else:
            cast_mask = None
        return NestedTensor(cast_tensor, cast_mask)

    def record_stream(self, *args, **kwargs):
        self.tensors.record_stream(*args, **kwargs)
        if self.mask is not None:
            self.mask.record_stream(*args, **kwargs)

    def decompose(self):
        return self.tensors, self.mask

    def __repr__(self):
        return str(self.tensors)

def _max_by_axis(the_list):
    """Helper function for creating a NestedTensor. From DETR."""

    maxes = the_list[0]
    for sublist in the_list[1:]:
        for index, item in enumerate(sublist):
            maxes[index] = max(maxes[index], item)
    return maxes

def nested_tensor_from_tensor_list(tensor_list: List[Tensor]):
    """Creates a NestedTensor from a list of Tensors. From DETR."""

    if tensor_list[0].ndim == 3:
        max_size = _max_by_axis([list(img.shape) for img in tensor_list])
        batch_shape = [len(tensor_list)] + max_size
        b, c, h, w = batch_shape
        dtype = tensor_list[0].dtype
        device = tensor_list[0].device
        tensor = torch.zeros(batch_shape, dtype=dtype, device=device)
        mask = torch.ones((b, h, w), dtype=torch.bool, device=device)
        for img, pad_img, m in zip(tensor_list, tensor, mask):
            pad_img[: img.shape[0], : img.shape[1], : img.shape[2]].copy_(img)
            m[: img.shape[1], :img.shape[2]] = False
    else:
        raise ValueError("Tensors must be 3-dimensional.")
    return NestedTensor(tensor, mask)

def inverse_sigmoid(x, eps=1e-5):
    """Transforms a sigmoid vector back into logits."""

    x = x.clamp(min=0, max=1)
    x1 = x.clamp(min=eps)
    x2 = (1 - x).clamp(min=eps)
    return torch.log(x1/x2)

def nested_collate(batch):
    """Collates batch of images as NestedTensor for use in DataLoader."""

    batch = list(zip(*batch))
    batch[0] = nested_tensor_from_tensor_list(batch[0])
    return tuple(batch)

def get_clones(module, num):
    """Duplicates modules for multi-scale learning."""

    return nn.ModuleList([deepcopy(module) for j in range(num)])

def tensor_to_pil(img):
    """De-normalizes a Tensor and converts to PIL Image for saving."""

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    # De-normalizes image.
    for c, m, s in zip(img, mean, std):
        c.mul_(s).add_(m)

    # Switches channels as required by PIL.
    img = torch.transpose(img, 1, 2)
    img = torch.transpose(img, 0, 2)
    img = Image.fromarray((img.cpu().numpy() * 255).astype(np.uint8))

    return img

def exclude_params(params, to_exclude):
    """Removes a subset of parameters from a list of parameters.

       Useful for setting learning rates for different modules.
    """

    new_params = []

    for p in params:
        exclude = False

        for param_set in to_exclude:
            if not exclude:
                for q in param_set:
                    if torch.equal(p, q):
                        exclude = True

        if not exclude:
            new_params.append(p)

    return new_params

def get_state_dict_from_checkpoint(checkpoint):
    """Finds state dict in checkpoint."""

    if "model" in checkpoint.keys():
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint.keys():
        state_dict = checkpoint["state_dict"]
    else:
        raise ValueError("No state dict found in checkpoint.")

    return state_dict

def get_balanced_sampler_weights_by_id(labels, offset):
    """Gets weights for each image for use in a balanced sampler.

       Note that the sampler is not exactly balanced as images may have more
       than one class present. The image weight is the mean weight of all
       its present classes.
    """

    images = labels["images"]
    classes = len(labels["categories"])

    def to_array(indices):
        """Expands list of indices into class-length array with offset."""
        a = np.zeros(classes)
        indices = np.asarray(indices)
        a[indices - offset] = 1
        return a

    vals = [to_array(img["classes"]) for img in images]
    totals = np.sum(np.stack(vals), axis=0)
    weights = 1. / totals

    classes_by_id = {
        img["id"]: np.asarray(img["classes"] - offset) for img in images
    }
    weights_by_id = {
        img["id"]: np.mean(weights[classes_by_id[img["id"]]]) for img in images
    }

    return weights_by_id

def compute_accuracy(results, coco_groundtruth, thresh=0.1):
    """Computes (proxy) top1 and top5 classification accuracy.

       The accuracy is the proportion of images for which a class present
       in the image is predicted in the top1/top5 most confidence boxes
       respectively. Not very rigorous; mostly useful for debugging.
    """

    groundtruth_by_img = {}
    for j in coco_groundtruth.imgs.keys():
        groundtruth_by_img[j] = {"boxes": [], "preds": []}

        inds = coco_groundtruth.getAnnIds(imgIds=[j])
        for i in inds:
            ann = coco_groundtruth.anns[i]
            groundtruth_by_img[j]["boxes"].append(ann["bbox"])
            groundtruth_by_img[j]["preds"].append(ann["category_id"])

    top1_total = 0
    top5_total = 0
    for img in results:
        res = results[img]
        gt = groundtruth_by_img[img]

        classes_in_img = set([p for p in gt["preds"]])

        # Computes top1 accuracy.
        preds = res["preds"][:1]
        confs = res["confs"][:1]
        for cls in classes_in_img:
            for pred, conf in zip(preds, confs):
                if cls == pred and conf >= thresh:
                    top1_total += 1
                    break

        # Computes top5 accuracy.
        preds = res["preds"][:5]
        confs = res["confs"][:5]
        for cls in classes_in_img:
            for pred, conf in zip(preds, confs):
                if cls == pred and conf >= thresh:
                    top5_total += 1
                    break

    top1_acc = top1_total / len(results)
    top5_acc = top5_total / len(results)

    return top1_acc, top5_acc

def gather_coco_results_across_gpus(results):
    """Collates COCO results across multiple GPUs for evaluation."""

    coco_results = {}
    for result in results:
        img_results = {}
        if len(result["image_id"]) > 1:
            for j, img_id in enumerate(result["image_id"]):
                img_results[img_id.item()] = {
                    "boxes": result["boxes"][j],
                    "confs": result["confs"][j],
                    "preds": result["preds"][j],
                }
        else:
            img_results[result["image_id"].item()] = {
                "boxes": result["boxes"],
                "confs": result["confs"],
                "preds": result["preds"],
            }
        coco_results.update(img_results)

    return coco_results

def save_infer_img(
    img,
    imgs_dir,
    name,
    classes,
    boxes,
    confs,
    preds,
    offset,
    target_boxes=None,
    target_labels=None,
):
    """Plots predictions and ground-truth boxes on an image and saves."""

    img = tensor_to_pil(img)
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype(
            "/usr/share/fonts/truetype/lato/Lato-Bold.ttf",
            size=16,
        )
        text_height, _ = font.getmetrics()
    except:
        font = ImageFont.load_default()
        text_height = 0

    # Plots predictions.
    for box, conf, pred in zip(boxes, confs, preds):
        draw.rectangle(
            ((box[0], box[1]), (box[2], box[3])),
            outline="red",
            width=3,
        )

        text_anchor = (box[0], box[1] - text_height)
        draw.text(
            text_anchor,
            f"{classes[pred - offset]} @ {conf:.2f}",
            fill="red",
            font=font,
        )

    # Plots ground-truth boxes.
    if target_boxes is not None and target_labels is not None:
        for box, label in zip(target_boxes, target_labels):
            draw.rectangle(
                ((box[0], box[1]), (box[2], box[3])),
                outline="blue",
                width=3,
            )

            text_anchor = (box[0], box[1] - text_height)
            draw.text(
                text_anchor,
                f"{classes[label - offset]}",
                fill="blue",
                font=font,
            )

    out_path = osp.join(imgs_dir, name)
    img.save(out_path, "JPEG")

