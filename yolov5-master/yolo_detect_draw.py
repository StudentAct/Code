import argparse
import os
import platform
import sys
from pathlib import Path
import cv2, numpy as np
import torch
from tqdm import tqdm

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative
from models.common import DetectMultiBackend
from utils.dataloaders import IMG_FORMATS, VID_FORMATS, LoadImages, LoadScreenshots, LoadStreams
from utils.general import (LOGGER, Profile, check_file, check_img_size, check_imshow, check_requirements, colorstr, cv2,
                           increment_path, non_max_suppression, print_args, scale_boxes, strip_optimizer, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import select_device, smart_inference_mode

device = select_device('cpu')
#weights = 'last_trainold_testold.pt'

weights = 'best_trainHM_testHM.pt'

model = DetectMultiBackend(weights, device=device, dnn=False, data='data/coco128.yaml', fp16=False)
# Run inference
stride, names, pt = model.stride, model.names, model.pt

imgsz=(640, 640)
imgsz = check_img_size(imgsz, s=stride)  # check image size

#img_path='/ext_data2/home/pdung/yolov5-master/runs/ch01_20210329105417_0035.jpg'
#img = cv2.imread(img_path)
model.warmup(imgsz=(1 if pt or model.triton else bs, 3, *imgsz))  # warmup
seen, windows, dt = 0, [], (Profile(), Profile(), Profile())

source = './runs/1.mp4'
dataset = LoadImages(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=1)
conf_thres = 0.25
iou_thres = 0.45
classes = None
agnostic_nms = False
max_det = 1000
save_path = './runs/imgs_1mp4/'
color = (0, 0, 255)

count = 0
for path, im, im0s, vid_cap, s in tqdm(dataset):
    save_path = f'./runs/imgs_1mp4_draw/{str(count).zfill(4)}.jpg'
    h0, w0 = im0s.shape[:2]
    h1, w1 = im.shape[1:]
    scale_x = w0/w1
    scale_y = h0/h1
    # print (im.shape, im0s.shape)
    # print(save_path)
    # cv2.imwrite(save_path, im0s)
    with dt[0]:
        im = torch.from_numpy(im).to(model.device)
        im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
        im /= 255  # 0 - 255 to 0.0 - 1.0
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        # print (im.shape)

    # Inference
    with dt[1]:
        pred = model(im, augment=False, visualize=False)

    # NMS
    with dt[2]:
        pred = non_max_suppression(pred, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
        det = pred[0]
        # print (len(det))
        pred_np = det.cpu().detach().numpy()
        
        if len(det)>0:
            #out_path = 'last_trainold_testold.txt'
            out_path = 'best_trainHM_testHM_1mp4_draw.txt'
            writer = open(out_path,'at')
            
            pred_np[:,0] *= scale_x
            pred_np[:,1] *= scale_y
            pred_np[:,2] *= scale_x
            pred_np[:,3] *= scale_y
            coords = str(pred_np[:,:-1])

            # visualize
            for i, d in enumerate(pred_np):
                 im0s = cv2.rectangle(im0s,
                      (int(d[0]), int(d[1])), (int(d[2]), int(d[3])),
                      color=color, thickness=2)

            #print(coords)
            for i in range(10): 
                coords = coords.replace('  ',' ')
            coords = coords.replace('[ ','[')
            writer.write(f'{save_path} {coords}\n')
            writer.close()
    # break
    cv2.imwrite(save_path, im0s)


    #     # for i, det in enumerate(pred):  # per image
    #     #     # print (det)
        #     print (len(det))
        #     pred_np = det.cpu().detach().numpy()
        #     print ('pred_np',pred_np)
        # print (len(pred),pred)
        # if len(pred)>1:
            # break
    
    count +=1
    # break


# h,w = img.shape[:2]
# scale_x = imgsz[0]/w
# scale_y = imgsz[1]/h
# scale = min(scale_x, scale_y)
# h2,w2  = int(scale*h), int(scale*w)
# img = cv2.resize(img,(h2,w2))
# im = torch.from_numpy(img).to(model.device)

# print(im.shape)
# im = torch.swapaxes(im, 0, 2) # 3,1920,1080
# im = torch.swapaxes(im, 1, 2)# 3,1080,1920
# print(im.shape)
# im = im.half() if model.fp16 else im.float()  # uint8 to fp16/32
# im /= 255  # 0 - 255 to 0.0 - 1.0
# if len(im.shape) == 3:
#     im = im[None]  # expand for batch dim
# print(im.shape)
# pred = model(im, augment=False, visualize=False)
# print(pred)
