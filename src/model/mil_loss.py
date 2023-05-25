"""Defines loss for multiple instance learning (MIL)."""

# Imports PyTorch packages.
import torch
import torch.nn.functional as F

# Imports other packages.
from sparsemax import Sparsemax

from utils.misc import inverse_sigmoid


def mil_score(outputs, joint_probability=None, sparse=None):
    """Computes box-wise MIL score.

       The MIL score is the elementwise product of the detection and
       classification softmax scores. We optionally sparsemax over
       the detections dimension, which typically increases performance.
    """

    if joint_probability:
        dets_logits = inverse_sigmoid(outputs["obj_confs"])
    else:
        dets_logits = outputs["dets_logits"]

    # Softmaxes over the classes dimension.
    classes_logits = outputs["classes_logits"]
    classes = F.softmax(classes_logits, dim=2)

    # Computes detection sigmoid as proxy for detection confidence.
    dets_sigmoid = dets_logits.sigmoid()

    # Softmaxes or sparsemaxes over the detections dimension.
    if sparse:
        dets = Sparsemax(dim=1)(dets_logits)
    else:
        dets = F.softmax(dets_logits, dim=1)

    if joint_probability:
        num_classes = classes.shape[-1]
        dets = dets.unsqueeze(-1).repeat(1, 1, num_classes)

    # Computes element-wise product of the two scores.
    scores = classes * dets

    return dets_sigmoid, scores

def mil_label(batch_size, num_classes, targets, offset=0):
    """Gets the weak supervision label for MIL.

       There is a 1 in the class slot if there is
       at least one instance of that class in the image.
    """

    # Creates empty tensor for MIL labels.
    mil_labels = torch.zeros((batch_size, num_classes))
    mil_labels = mil_labels.type_as(targets[0]["boxes"])

    # Populates MIL label from targets.
    for j, img_target in enumerate(targets):
        for cls in img_target["image_labels"]:
            # Subtracts offset (e.g., if the labels are 1-indexed).
            cls -= offset

            # Sets class slot to 1 in the label.
            mil_labels[j][cls] = 1

    return mil_labels

def mil_nll(mil_scores, mil_labels, eps=1e-5):
    """Computes negative log-likelihood between MIL scores and MIL labels.
    
       eps argument prevents loss from becoming NaN.
    """

    # Computes class-wise log-likelihoods.
    class_likelihoods = mil_labels * torch.log(mil_scores + eps) \
                        + (1 - mil_labels) * torch.log(1 - mil_scores + eps)

    # Computes mean NLL loss across batch.
    nll = -torch.sum(class_likelihoods, 1)
    mil_loss = torch.mean(nll, 0)

    return mil_loss

def objectness_mse_loss(dets_sigmoid, obj_confs):
    """Computes MSE regularization loss on detections and objectness.
    
       Aligns the maximum detection confidence of the weakly supervised
       model with the objectness confidence of the class-agnostic model.
    """

    # Computes maximum detection confidences and MSE loss.
    max_det_confs, _ = torch.max(dets_sigmoid, dim=2)
    objectness_loss = F.mse_loss(max_det_confs, obj_confs)

    return objectness_loss

def mil_loss(
    outputs,
    targets,
    joint_probability=None,
    objectness_scale=1,
    offset=0,
    sparse=None,
):
    """Computes MIL score and label, then returns NLL loss with objectness."""

    batch_size, _, classes = outputs["classes_logits"].shape
    mil_labels = mil_label(batch_size, classes, targets, offset=offset)

    dets_sigmoid, scores = mil_score(
        outputs,
        joint_probability=joint_probability,
        sparse=sparse,
    )
    mil_scores = torch.sum(scores, 1)
    mil_loss = mil_nll(mil_scores, mil_labels)

    if joint_probability:
        objectness_loss = 0. * torch.sum(outputs["dets_logits"])
    else:
        objectness_loss = objectness_mse_loss(
            dets_sigmoid,
            outputs["obj_confs"],
        )
        objectness_loss *= objectness_scale
        
    return mil_loss, objectness_loss

