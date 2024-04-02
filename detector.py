#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii, Inc. and its affiliates.

import os
import time
from loguru import logger
import cv2
import torch


from yolox.exp import Exp as MyExp
from yolox.data.data_augment import ValTransform
from yolox.data.datasets import COCO_CLASSES
from yolox.utils import fuse_model, get_model_info, postprocess, vis


# bytetrack_x_mot17 detector settings
class Exp_bytetrack_x_mot17(MyExp):
    def __init__(self):
        super(Exp_bytetrack_x_mot17, self).__init__()
        self.num_classes = 1
        self.depth = 1.33
        self.width = 1.25
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]
        self.train_ann = "train.json"
        self.val_ann = "test.json"    
        self.input_size = (800, 1440)
        self.test_size = (800, 1440)
        self.random_size = (18, 32)
        self.max_epoch = 80
        self.print_interval = 20
        self.eval_interval = 5
        self.test_conf = 0.4
        self.nmsthre = 0.45
        self.no_aug_epochs = 10
        self.basic_lr_per_img = 0.001 / 64.0
        self.warmup_epochs = 1

# default yolox_x detector settings
class Exp_yolox_x(MyExp):
    def __init__(self):
        super(Exp_yolox_x, self).__init__()
        self.depth = 1.33
        self.width = 1.25
        self.test_conf = 0.4
        self.nmsthre = 0.45
        self.exp_name = os.path.split(os.path.realpath(__file__))[1].split(".")[0]

class YOLOX(object):
    """
        YOLOX detection model.
        
        Attributes:
        test_size: The size of the image that the model will be tested.
        model_weighs: Checkpoint file path for the model. 
        model_config: Model configuration from class Exp. Replaceable.
        device: "cuda" or "cpu".
        half:Adopt mix precision evaluating.
        fuse:Fuse conv and bn for testing.

    """
    
    def __init__(
        self,
        test_size=(640, 640),
        model_weighs="yolox_x.pt",
        model_config="bytetrack_x_mot17",
        device="cuda",
        half=False,
        fuse=False,
    ):
        

         # exp can be repladed other class Exp config
        if model_config == "bytetrack_x_mot17":
            self.exp = Exp_bytetrack_x_mot17() 
            self.preproc = ValTransform(legacy=True)
        elif model_config == "yolox_x":
            self.exp = Exp_yolox_x()
            self.preproc = ValTransform(legacy=False)
        else:
            raise ValueError("Unsupported model configuration: {}".format(model_config))
        
        self.device = device
        self.test_size = test_size
        self.model_weighs = model_weighs
        self.fp16 = half
        self.fuse = fuse

        self.exp.test_size = self.test_size
        # load model
        model = self.exp.get_model()
        logger.info("Model Summary: {}".format(get_model_info(model, self.exp.test_size)))
        model.to(self.device)
        model.eval()
        logger.info("loading checkpoint")
        ckpt = torch.load(self.model_weighs, map_location="cpu")
        # load the model state dict
        model.load_state_dict(ckpt["model"])
        logger.info("loaded checkpoint done.")
        # fuse and half
        if self.fuse:
            logger.info("\tFusing model...")
            model = fuse_model(model)
        if self.fp16:
            model = self.exp.model.half()


    def inference(self, img):
        img_info = {"id": 0}
        # if get a img path
        if isinstance(img, str):
            img_info["file_name"] = os.path.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None
        # get img info
        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img
        img_info["ratio"] = min(self.exp.test_size[0] / img.shape[0], self.exp.test_size[1] / img.shape[1])
     
        # img preprocessing
        img, _ = self.preproc(img, None, self.exp.test_size)
        img = torch.from_numpy(img).unsqueeze(0)
        img = img.float()
        img = img.to(self.device)
        if self.fp16:
            img = img.half()  # to FP16

        # inference
        with torch.no_grad():
            t0 = time.time()
            outputs = self.exp.model(img)
            outputs = postprocess(
                outputs, self.exp.num_classes, self.exp.test_conf, self.exp.nmsthre,
                class_agnostic=True
            )
            logger.info("Infer time: {:.4f}s".format(time.time() - t0))
            outputs[0][:, 0:4] /= img_info["ratio"]
        # (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        return outputs, img_info
    
    def __call__(self, img):
        detection, _ = self.inference(img)
        return(detection)
    
    # visualize a image
    def visual(self, output, img_info, cls_conf=0.2):
        output = output.cpu()
        bboxes = output[:, 0:4]
        cls = output[:, 6]
        scores = output[:, 4] * output[:, 5]
        vis_res = vis(img_info["raw_img"], bboxes, scores, cls, cls_conf, COCO_CLASSES)
        cv2.imshow('Image', vis_res)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return vis_res





    
