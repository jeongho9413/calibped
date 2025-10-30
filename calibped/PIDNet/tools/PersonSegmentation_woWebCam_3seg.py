from PIL import Image
import joblib
import argparse
import matplotlib.pyplot as plt
import os
import numpy as np
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torchvision import transforms
import torch.nn.functional as F

import _init_paths

# from PIDNet.models import pidnet
from models import pidnet
# from PIDNet import datasets
import datasets

import cv2
from datetime import datetime

def load_model(args, num_classes):
    device = torch.device(args.device)

    print(f'Loading weights from {args.model}')
    tmp_dict = torch.load(args.model, map_location='cpu')
    
    model = pidnet.PIDNet(m=3, n=4, num_classes=num_classes, planes=64, ppm_planes=112, head_planes=256, augment=False).to(device)
    model_dict = model.state_dict()
    
    seg_dict = {}
    for k,v in tmp_dict.items():
        key = k[6:]
        if key in model_dict and v.shape == model_dict[key].shape:
            name = key
            seg_dict[name]=v
    model.load_state_dict(seg_dict)
    model.eval()
    
    return model


def load_images(args, return_original=False):
    resize_info = (args.height, args.width)
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(resize_info),
        transforms.Normalize((0.469, 0.500, 0.398), (0.235, 0.238, 0.275)),
        ])

    print(f'Loading {len(args.filenames)} images')
    images = []
    if return_original:
        originals = []
    for fn in args.filenames:
        image = Image.open(fn)
        if return_original:
            originals.append(image)
        images.append(tf(image))

    if return_original:
        return torch.stack(images), np.array(originals)
    
    return torch.stack(images)

def create_overlay_image(original_frame, segmentation, args):
    """セグメンテーション結果を元画像に重畳表示"""
    # opencvの形式に画像を変換
    print(f'{original_frame.shape=}, {original_frame.max()=}, {original_frame.min()=}, {original_frame.dtype=}')
    original_frame = cv2.cvtColor(original_frame, cv2.COLOR_RGB2BGR)
    
    # セグメンテーション結果を元画像のサイズにリサイズ
    seg_resized = cv2.resize(segmentation.astype(np.uint8), (original_frame.shape[1], original_frame.shape[0]))
    
    # 人（クラス0）を白，
    mask = np.zeros_like(original_frame)
    mask[seg_resized == 0] = [255, 255, 255]  # 人を白で表示
    mask[seg_resized != 0] = [0, 0, 0] # それ以外を黒で表示
    
    # 元画像とマスクを重畳（透明度50%）
    alpha = 0.5
    overlay = cv2.addWeighted(original_frame, 1-alpha, mask, alpha, 0)

    # opencvの形式をPILの形式に変換
    overlay = cv2.cvtColor(overlay.astype(np.uint8), cv2.COLOR_BGR2RGB)
    print(f'{overlay.shape=}, {overlay.max()=}, {overlay.min()=}')
    # overlay = np.clip((overlay/overlay.max())*255, 0, 255).astype(np.uint8)
    # print(f'{overlay.shape=}, {overlay.max()=}, {overlay.min()=}')
    return overlay

def predict(args, model, imgs):
    device = torch.device(args.device)
    bs = args.batch_size

    print('Predicting segmentations')
    segs = []
    with torch.no_grad():
        for i in range(0, len(imgs), bs):
            batch = imgs[i: i+bs]
            batch = batch.to(device)
            
            pred = model(batch)
            pred = F.interpolate(input=pred, size=batch.size()[-2:], mode='bilinear', align_corners=True)
            segs.extend(torch.argmax(pred, dim=1).cpu().numpy())
            
    return segs
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Estimate segmentation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--filenames', nargs='*', type=str, help='list of images', required=True)
    parser.add_argument('--filenames', nargs='*', type=str, help='list of images', default=['../data/kusakari/kusakari_ds/2023-09-13-15-57-05.png'])
    parser.add_argument('--width', default=1344, type=int, help='image width after resize')
    parser.add_argument('--height', default=768, type=int, help='image height after resize')
    parser.add_argument('--out', default='result', type=str, help='output directory')
    parser.add_argument('--device', default='cuda:0', type=str, help='device for PyTorch')
    parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    # parser.add_argument('--model', default='./ckpt/best.pt', type=str, help='pretrained model filepath')
    parser.add_argument('--model', default='../pretrained_models/kusakari/best_kawahara.pt', type=str, help='pretrained model filepath')
    parser.add_argument('--overlay', action='store_true', default=True, help='overlay on the original image')
    parser.add_argument('--cmap', default='tab20', type=str, help='color map name')
    args = parser.parse_args()
    NUM_CLASSES = 3
    
    model = load_model(args, NUM_CLASSES)
    if args.overlay:
        images, originals = load_images(args, return_original=True)
    else:
        images = load_images(args)
    segmentations = predict(args, model, images)

    color_map = plt.get_cmap(args.cmap, NUM_CLASSES)
    print(f'Saving {len(segmentations)} segmentation images')
    if not os.path.exists(args.out):
        os.mkdir(args.out)

    shape = images[0].numpy().shape
    for i, seg in enumerate(segmentations):
        """
        # 河原先生のコード（20250203-3seg-segmentation.py）
        result = np.zeros(shape).astype(np.uint8)
        for j, rgb in enumerate(color_map.colors):
            color = [int(rgb[0]*256), int(rgb[1]*256), int(rgb[2]*256)]
            color = np.clip(color, 0, 255)
            for k in range(3):
                result[k,:,:][seg==j] = color[k]
        """
        if args.overlay:
            result = create_overlay_image(originals[i], seg, args)
        else:
            # 人（クラス0）を白，それ以外を黒に
            result = np.zeros(shape).astype(np.uint8)
            result[:, seg == 0] = 255
            result = result.transpose(1, 2, 0)
        result = Image.fromarray(result)
        image_name, image_ext = args.filenames[i].split('/')[-1].split('.')
        result.save(f'{args.out}/overlay_{image_name}_{datetime.now().strftime("%Y%m%dT%H%M%S")}.{image_ext}')
