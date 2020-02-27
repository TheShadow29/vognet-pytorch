"""
Helper functions for boxes
Adapted from
https://github.com/facebookresearch/maskrcnn-benchmark/
blob/master/maskrcnn_benchmark/structures/boxlist_ops.py
"""
import torch

TO_REMOVE = 0


def get_area(box):
    """
    box: [N, 4]
    torch.tensor of
    type x1y1x2y2
    """
    area = (
        (box[:, 2] - box[:, 0] + TO_REMOVE) *
        (box[:, 3] - box[:, 1] + TO_REMOVE)
    )
    return area


def box_iou(box1, box2):
    """
    box1: [N, 4]
    box2: [M, 4]
    both of type torch.tensor
    Assumes both of type x1y1x2y2
    output: [N,M]
    """
    if len(box1.shape) == 1 and len(box1) == 4:
        box1 = box1.unsqueeze(0)
    if len(box2.shape) == 1 and len(box2) == 4:
        box2 = box2.unsqueeze(0)

    N = len(box1)
    M = len(box2)

    area1 = get_area(box1)
    area2 = get_area(box2)

    lt = torch.max(box1[:, None, :2], box2[:, :2])  # [N,M,2]
    rb = torch.min(box1[:, None, 2:], box2[:, 2:])  # [N,M,2]

    wh = (rb - lt + TO_REMOVE).clamp(min=0)  # [N,M,2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N,M]

    iou = inter / (area1[:, None] + area2 - inter)
    return iou
