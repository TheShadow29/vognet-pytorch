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


def bbox_overlaps(rois, gt_box, frm_mask):

    overlaps = bbox_overlaps_batch(rois[:, :, :5], gt_box[:, :, :5], frm_mask)

    return overlaps


def bbox_overlaps_batch(anchors, gt_boxes, frm_mask=None):
    """
    Source:
    https://github.com/facebookresearch/grounded-video-description/blob/
    master/misc/bbox_transform.py#L176
    anchors: (b, N, 4) ndarray of float
    gt_boxes: (b, K, 5) ndarray of float
    frm_mask: (b, N, K) ndarray of bool

    overlaps: (b, N, K) ndarray of overlap between boxes and query_boxes
    """
    batch_size = gt_boxes.size(0)

    N = anchors.size(1)
    K = gt_boxes.size(1)

    anchors = anchors[:, :, :5].contiguous()
    gt_boxes = gt_boxes[:, :, :5].contiguous()

    gt_boxes_x = (gt_boxes[:, :, 2] - gt_boxes[:, :, 0] + 1)
    gt_boxes_y = (gt_boxes[:, :, 3] - gt_boxes[:, :, 1] + 1)
    gt_boxes_area = (gt_boxes_x * gt_boxes_y).view(batch_size, 1, K)

    anchors_boxes_x = (anchors[:, :, 2] - anchors[:, :, 0] + 1)
    anchors_boxes_y = (anchors[:, :, 3] - anchors[:, :, 1] + 1)
    anchors_area = (anchors_boxes_x *
                    anchors_boxes_y).view(batch_size, N, 1)

    gt_area_zero = (gt_boxes_x == 1) & (gt_boxes_y == 1)
    anchors_area_zero = (anchors_boxes_x == 1) & (anchors_boxes_y == 1)

    boxes = anchors.view(batch_size, N, 1, 5).expand(batch_size, N, K, 5)
    query_boxes = gt_boxes.view(
        batch_size, 1, K, 5).expand(batch_size, N, K, 5)

    iw = (torch.min(boxes[:, :, :, 2], query_boxes[:, :, :, 2]) -
          torch.max(boxes[:, :, :, 0], query_boxes[:, :, :, 0]) + 1)
    iw[iw < 0] = 0

    ih = (torch.min(boxes[:, :, :, 3], query_boxes[:, :, :, 3]) -
          torch.max(boxes[:, :, :, 1], query_boxes[:, :, :, 1]) + 1)
    ih[ih < 0] = 0
    ua = anchors_area + gt_boxes_area - (iw * ih)

    if frm_mask is not None:
        # proposal and gt should be on the same frame to overlap
        # print('Percentage of proposals that are in the annotated frame: {}'.format(torch.mean(frm_mask.float())))

        overlaps = iw * ih / ua
        overlaps *= frm_mask.type(overlaps.type())

        # mask the overlap here.
        overlaps.masked_fill_(gt_area_zero.view(
            batch_size, 1, K).expand(batch_size, N, K), 0)
        overlaps.masked_fill_(anchors_area_zero.view(
            batch_size, N, 1).expand(batch_size, N, K), -1)

    return overlaps
