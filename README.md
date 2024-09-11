# A Unified Object and Keypoint Detection Framework for Personal Protective Equipment Use

This repository is a PyTorch implementation of our paper: A Unified Object and Keypoint Detection Framework for Personal Protective Equipment Use.

> We will release the corresponding source after acceptance.

## Installation
This code is based on [YOLOv5](https://github.com/ultralytics/yolov) and [YOLO-pose](https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose). Please install the required dependencies, and you can also refer to their preparation tutorials as a reference.

## Data prepation
Some data prepation scripts can be found in `utils/data_process`.  
The annotated data can be found in this [google drive]().

## MTKF Implementation
The updated code for implementing MTKF is displayed below:

1. Updated DC-NMS in [general.py]()
2. Updated C3MA in [comment.py]()
3. Updated framework configs in [models/hub]()

Implement commend
```bash
sh train.sh
sh test.sh
```

## Reference
- https://github.com/ultralytics/yolov5
- https://github.com/TexasInstruments/edgeai-yolov5/tree/yolo-pose  
...
