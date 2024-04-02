## Get models and weight
You can get yolox model and default model weight from [yolox official](https://github.com/Megvii-BaseDetection/YOLOX) and get bytetrack trained model weight from [bytetrack official](https://github.com/ifzhang/ByteTrack)

## Modify your own model config
detector.py中配置了yolox和bytetrack_x_mot17两个模型的设置，可以根据自己的需求添加新配置。

The settings of the yolox and bytetrack_x_mot17 models are configured in detector.py. You can add new configurations according to your own needs.

## Import module
Usage:

```python
from yoloxdetector import YOLOX
model = YOLOX(
    test_size=(640, 1088),
    model_weighs="../bytetrack_x_mot17.pt",
    model_config="bytetrack_x_mot17",
    # test_size=(640, 640),
    # model_weighs="../yolox_x.pt",
    # model_config="yolox_x",
    device="cuda",
    half=False,
    fuse=False,
)
outputs = model(img)
```
得到的outputs形式如下：

The obtained outputs are of the following form: 

```
outputs (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
```