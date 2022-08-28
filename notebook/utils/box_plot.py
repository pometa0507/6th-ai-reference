######################
# Based on https://github.com/traveller59/second.pytorch
# Licensed under The MIT License
# Author Yan Yan
######################

import cv2
import numpy as np
import torch


def cv2_draw_3d_bbox(img, bboxes, colors, thickness=1, line_type=cv2.LINE_8):
    # assume bboxes has right format(N, 8, 2).
    bboxes = bboxes.astype(np.int32)
    for box, color in zip(bboxes, colors):
        color = tuple(int(c) for c in color)
        box_a, box_b = box[:4], box[4:]
        for pa, pb in zip(box_a, box_a[[1, 2, 3, 0]]):
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     color, thickness, line_type)
        for pa, pb in zip(box_b, box_b[[1, 2, 3, 0]]):
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     color, thickness, line_type)
        for pa, pb in zip(box_a, box_b):
            cv2.line(img, (int(pa[0]), int(pa[1])), (int(pb[0]), int(pb[1])),
                     color, thickness, line_type)
    return img


def boxes_to_corners_3d(boxes3d):
    """
        7 -------- 4
       /|         /|
      6 -------- 5 .
      | |        | |
      . 3 -------- 0
      |/         |/
      2 -------- 1
    Args:
        boxes3d:  (N, 7) [x, y, z, dx, dy, dz, heading], (x, y, z) is the box center
    Returns:
        boxes3d:  shape=(N, 8, 3)
    """
    boxes3d, is_numpy = check_numpy_to_torch(boxes3d)

    template = boxes3d.new_tensor((
        [1, 1, -1], [1, -1, -1], [-1, -1, -1], [-1, 1, -1],
        [1, 1, 1], [1, -1, 1], [-1, -1, 1], [-1, 1, 1],
    )) / 2

    corners3d = boxes3d[:, None, 3:6].repeat(1, 8, 1) * template[None, :, :]
    corners3d = rotate_points_along_z(corners3d.view(-1, 8, 3),
                                      boxes3d[:, 6]).view(-1, 8, 3)
    corners3d += boxes3d[:, None, 0:3]

    return corners3d.numpy() if is_numpy else corners3d


def check_numpy_to_torch(x):
    if isinstance(x, np.ndarray):
        return torch.from_numpy(x).float(), True
    return x, False


def rotate_points_along_z(points, angle):
    """
    Args:
        points: (B, N, 3 + C)
        angle: (B), angle along z-axis, angle increases x ==> y
    Returns:
    """
    points, is_numpy = check_numpy_to_torch(points)
    angle, _ = check_numpy_to_torch(angle)

    cosa = torch.cos(angle)
    sina = torch.sin(angle)
    zeros = angle.new_zeros(points.shape[0])
    ones = angle.new_ones(points.shape[0])
    rot_matrix = torch.stack((
        cosa,  sina, zeros,
        -sina, cosa, zeros,
        zeros, zeros, ones
    ), dim=1).view(-1, 3, 3).float()
    points_rot = torch.matmul(points[:, :, 0:3], rot_matrix)
    points_rot = torch.cat((points_rot, points[:, :, 3:]), dim=-1)
    return points_rot.numpy() if is_numpy else points_rot

