import torch


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def filter_high_iou_bboxes(output, iou_threshold):
    num_bboxes = output.shape[0]
    is_removed = [False] * num_bboxes

    for i in range(num_bboxes):
        if is_removed[i]:
            continue

        current_bbox = output[i, :4]
        current_conf = output[i, 4]

        # Calculate IoU between the current bbox and other bboxes
        iou = box_iou(current_bbox.view(1, -1), output[:, :4])
        if iou.sum() == 1:
            continue

        # Find indices of bboxes with IoU greater than the threshold
        high_iou_indices = (iou > iou_threshold).nonzero(as_tuple=False).squeeze(1)

        # Find the indices of bboxes to be removed (lower confidence scores)
        lower_conf_indices = high_iou_indices[output[high_iou_indices, 4] < current_conf]

        # Remove bboxes with lower confidence scores
        for index in lower_conf_indices:
            is_removed[index] = True

    filtered_indices = [i for i, removed in enumerate(is_removed) if not removed]
    filtered_output = output[filtered_indices]
    return filtered_output


# Example usage
iou_threshold = 0.8
output = torch.tensor([[3.69375e+01, 1.21625e+02, 5.39688e+02, 5.61125e+02, 5.86854e-01, 1.00000e+00, 3.96500e+02,
                        3.81500e+02, 9.95117e-01],
                       [2.09938e+02, 4.61875e+01, 5.51938e+02, 4.56688e+02, 4.10043e-01, 1.00000e+00, 3.92500e+02,
                        3.78000e+02, 9.79492e-01],
                       [1.43625e+02, 3.04172e+02, 5.12625e+02, 5.61172e+02, 2.58463e-01, 1.00000e+00, 3.93750e+02,
                        3.82750e+02, 9.52637e-01]])  # Your output tensor here

filtered_output = filter_high_iou_bboxes(output, iou_threshold)
print(filtered_output)
