yolox官方仓库：
yolox official:
https://github.com/Megvii-BaseDetection/YOLOX
bytetrack官方仓库：
bytetrack official:
https://github.com/ifzhang/ByteTrack

detector.py中配置了yolox和bytetrack_x_mot17两个模型的设置，可以根据自己的需求添加新配置。
The detector.py is configured with the settings for both models, and you can add new configurations to suit your needs.

使用方法：
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
    outputs (x1, y1, x2, y2, obj_conf, class_conf, class_pred)