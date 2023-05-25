import argparse
import os
import platform
import sys
from pathlib import Path
import cv2
import torch
sys.path.append('/home/isaac/Isaac/Xira/CFE_data_extraction/yolov5')

from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

imgsz=(640, 640)
path = 'val2/IMG_20230522_111213.jpg'
image = cv2.imread(path)
device = select_device('')

 # Load model
model = DetectMultiBackend('yolov5/weights/yolov5s_telmex.pt', device=device, dnn=False, data='yolov5/data/coco128.yaml', fp16=False)
stride, names, pt = model.stride, model.names, model.pt
imgsz = check_img_size(imgsz, s=stride) 
print(stride, names, pt)
#image = check_img_size(image, stride)  # check image size
print(imgsz)

dataset = LoadImages(path, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)

# Run inference
model.warmup(imgsz=(1 if pt or model.triton else 1, 3, *imgsz))  # warmup
seen, windows, dt = 0, [], (Profile(), Profile(), Profile())
for path, im, im0s, vid_cap, s in dataset:
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim

    # Inference
    with dt[1]:
        pred = model(im, augment=False, visualize=False)

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, 0.25, 0.45, None, False, max_det=1000)

    for i, det in enumerate(pred):
        print(det)
        im0 = im0s.copy()
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]
        if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_boxes(im.shape[2:], det[:, :4], im0.shape).round()
                print(det)

cv2.circle(image,(1821,113),5,(255,0,0),7)
cv2.circle(image,(2318,408),5,(255,0,0),7)
cv2.imshow('Image',image)
cv2.waitKey(0)
cv2.destroyAllWindows()

