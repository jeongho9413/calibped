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
import cv2
import time

import _init_paths

# from PIDNet.models import pidnet
from models import pidnet
# from PIDNet import datasets
import datasets


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
    model.load_state_dict(seg_dict, strict=False)
    model.eval()
    
    return model


def preprocess_frame(frame, args):
    """OpenCVのBGR画像をPIL画像に変換し、前処理を行う"""
    # BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # PIL画像に変換
    pil_image = Image.fromarray(frame_rgb)
    
    # 前処理
    tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize((args.height, args.width)),
        transforms.Normalize((0.469, 0.500, 0.398), (0.235, 0.238, 0.275)),
    ])
    
    return tf(pil_image).unsqueeze(0)  # バッチ次元を追加


def predict_single_frame(args, model, frame_tensor):
    """単一フレームのセグメンテーション予測"""
    device = torch.device(args.device)
    
    with torch.no_grad():
        frame_tensor = frame_tensor.to(device)
        pred = model(frame_tensor)
        pred = F.interpolate(input=pred, size=frame_tensor.size()[-2:], mode='bilinear', align_corners=True)
        seg = torch.argmax(pred, dim=1).cpu().numpy()[0]  # バッチ次元を削除
        
    return seg


def create_overlay_image(original_frame, segmentation, args):
    """セグメンテーション結果を元画像に重畳表示"""
    # セグメンテーション結果を元画像のサイズにリサイズ
    seg_resized = cv2.resize(segmentation.astype(np.uint8), (original_frame.shape[1], original_frame.shape[0]))
    
    # 人（kawahara:クラス0、cityscapes:クラス11）を白、それ以外を黒に
    mask = np.zeros_like(original_frame)
    if 'kawahara' in args.model:
        mask[seg_resized == 0] = [255, 255, 255]
    elif 'cityscapes' in args.model:
        mask[seg_resized == 11] = [255, 255, 255]
    
    # 元画像とマスクを重畳（透明度50%）
    alpha = 0.5
    overlay = cv2.addWeighted(original_frame, 1-alpha, mask, alpha, 0)
    
    return overlay


def run_webcam_segmentation(args):
    """ウェブカメラからのリアルタイムセグメンテーション"""
    # カメラを初期化
    cap = cv2.VideoCapture(0)  # デフォルトカメラ
    
    if not cap.isOpened():
        print("カメラを開けませんでした")
        return
    
    # カメラの設定
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # モデルを読み込み
    num_classes = 3 if 'kawahara' in args.model else 19
    model = load_model(args, num_classes)
    print("リアルタイムセグメンテーションを開始します。'q'キーで終了します。")
    try:
        while True:
            # フレームを読み込み
            ret, frame = cap.read()
            if not ret:
                print("フレームを読み込めませんでした")
                break
            # フレームを前処理
            frame_tensor = preprocess_frame(frame, args)
            # セグメンテーション予測
            segmentation = predict_single_frame(args, model, frame_tensor)
            # 結果を重畳表示
            overlay = create_overlay_image(frame, segmentation, args)
            # 結果を表示
            cv2.imshow('Real-time Person Segmentation', overlay)
            # cv2.imshow('Real-time Person Segmentation', (segmentation*13).astype(np.uint8))
            
            # 'q'キーで終了
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("セグメンテーションを停止します")
    
    finally:
        # リソースを解放
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # parser = argparse.ArgumentParser(description='Estimate segmentation', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser = argparse.ArgumentParser(description='Real-time Person Segmentation with WebCam', formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    # parser.add_argument('--filenames', nargs='*', type=str, help='list of images', required=True)
    # parser.add_argument('--filenames', nargs='*', type=str, help='list of images', default=['../data/kusakari/kusakari_ds/2023-09-13-15-57-05.png'])
    parser.add_argument('--width', default=1344, type=int, help='image width after resize')
    parser.add_argument('--height', default=768, type=int, help='image height after resize')
    # parser.add_argument('--out', default='result', type=str, help='output directory')
    parser.add_argument('--device', default='cuda:0', type=str, help='device for PyTorch')
    # parser.add_argument('--batch_size', default=1, type=int, help='batch size')
    # parser.add_argument('--model', default='./ckpt/best.pt', type=str, help='pretrained model filepath')
    # parser.add_argument('--model', default='../pretrained_models/kusakari/best_kawahara.pt', type=str, help='pretrained model filepath')
    parser.add_argument('--model', default='../pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt', type=str, help='pretrained model filepath')
    # parser.add_argument('--cmap', default='tab20', type=str, help='color map name')
    args = parser.parse_args()
    
    # リアルタイムセグメンテーションを実行
    run_webcam_segmentation(args)
