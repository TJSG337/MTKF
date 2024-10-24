# A Unified Object and Keypoint Detection Framework for Personal Protective Equipment Use

This repository is a PyTorch implementation of our paper: A Unified Object and Keypoint Detection Framework for Personal Protective Equipment Use.

## Installation
This code is based on [YOLOv5](https://github.com/ultralytics/yolov5) and [YOLO-pose](https://arxiv.org/abs/2204.06806). Please install the required dependencies, and you can also refer to their preparation tutorials as a reference.

## Data prepation
Some data prepation scripts can be found in `utils/data_process`.  
The annotated data can be found in this [google drive]().

## MTKF Implementation
The updated code for implementing MTKF is displayed below:

1. Updated DC-NMS in [general.py](./utils/general.py)
2. Updated C3MA in [comment.py](./models/common.py)
3. Updated framework configs in [models/hub](./models/hub/)


## Reference
- https://github.com/ultralytics/yolov5
- D. Maji, S. Nagori, M. Mathew, D. Poddar, YOLO-Pose: Enhancing YOLO for Multi Person Pose Estimation Using Object Keypoint Similarity Loss, in: Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops, 2022: pp. 2637–2646.
- M. Park, D.Q. Tran, J. Bak, S. Park, Small and overlapping worker detection at construction sites, Automation in Construction 151 (2023) 104856. https://doi.org/10.1016/j.autcon.2023.104856. 
...
