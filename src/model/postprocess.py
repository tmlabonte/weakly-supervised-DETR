"""Postprocesses WS-DETR output for validation and inference."""

# Imports PyTorch packages.
import torch
from torchvision.ops import nms

# Imports local packages.
from utils import box_ops

# Imports local model packages.
from .mil_loss import mil_score


def postprocess(
    outputs,
    target_sizes,
    joint_probability=None,
    nms_thresh=None,
    offset=0,
    sparse=None,
    supervised=None,
):
    """Scales and formats model output for validation and inference."""

    # Gets classification probabilities from supervised model.
    if supervised:
        prob = outputs["classes_logits"].sigmoid()
    # Gets MIL score from weakly supervised model.
    # Note: Even if sparsemax is used for training, it is not
    # applied during postprocessing (similarly to dropout).
    else:
        _, prob = mil_score(outputs, joint_probability=joint_probability)

    # Postprocessing from Deformable DETR; sorts boxes and preds by confidence.
    # Adds offset at the end (e.g., in case the labels are 1-indexed).
    confs, topk_indexes = torch.topk(prob.view(prob.shape[0], -1), 300, dim=1)
    topk_boxes = topk_indexes // prob.shape[2]
    preds = topk_indexes % prob.shape[2] + offset

    # Converts boxes to ((x1, y1), (x2, y2)) coordinates.
    boxes = box_ops.box_cxcywh_to_xyxy(outputs["boxes"])
    boxes = torch.gather(boxes, 1, topk_boxes.unsqueeze(-1).repeat(1, 1, 4))

    # Converts output from a tensor to a list.
    boxes = list(torch.unbind(boxes))
    confs = list(torch.unbind(confs))
    preds = list(torch.unbind(preds))

    # Performs non-maximum suppression on the model output.
    # Not strictly necessary and sometimes makes model output worse,
    # but can be useful for visualization and inference.
    if nms_thresh:
        it = zip(boxes, confs, preds)
        for j, (img_boxes, img_confs, img_preds) in enumerate(it):
            inds = nms(img_boxes, img_confs, iou_threshold=nms_thresh)
            boxes[j] = img_boxes[inds]
            confs[j] = img_confs[inds]
            preds[j] = img_preds[inds]

    # Converts from relative [0, 1] to absolute [0, height] coordinates.
    img_h, img_w = target_sizes.unbind(1)
    scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
    for j, img_boxes in enumerate(boxes):
        boxes[j] = img_boxes * scale_fct[j, None, :]

        # Clamps box width and height to image.
        x_coords = boxes[j][:, 0::2]
        y_coords = boxes[j][:, 1::2]
        boxes[j][:, 0::2] = torch.clamp(x_coords, min=0., max=float(img_w[j]))
        boxes[j][:, 1::2] = torch.clamp(y_coords, min=0., max=float(img_h[j]))

    # Clamps confs to [0, 1].
    for j, _ in enumerate(confs):
        confs[j] = torch.clamp(confs[j], min=0., max=1.)

    # Builds list of output dicts.
    results = [
        {"boxes": b, "confs": c, "preds": p}
        for b, c, p in zip(boxes, confs, preds)
    ]

    return results

