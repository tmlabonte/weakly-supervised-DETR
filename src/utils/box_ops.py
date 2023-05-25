"""Operations for transforming bounding boxes."""

# Import PyTorch packages.
import torch


def box_cxcywh_to_xyxy(x):
    """Converts boxes from center-x center-y width height to xyxy."""

    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=-1)

def box_cxcywh_to_xywh(x):
    """Converts boxes from center-x center-y width height to xywh."""

    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), w, h]
    return torch.stack(b, dim=-1)

def box_xyxy_to_cxcywh(x):
    """Converts boxes from xyxy to center-x center-y width height."""

    x0, y0, x1, y1 = x.unbind(-1)
    b = [(x0 + x1) / 2, (y0 + y1) / 2, (x1 - x0), (y1 - y0)]
    return torch.stack(b, dim=-1)

def box_xyxy_to_xywh(x):
    """Converts boxes from xyxy to x y width height."""

    x0, y0, x1, y1 = x.unbind(-1)
    b = ((x0, y0, x1 - x0, y1 - y0))
    return torch.stack(b, dim=-1)
