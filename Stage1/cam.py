# -*- coding: utf-8 -*-
"""
任务：热图导出
需要修改参数：args：输入文件夹、热图方法、尺寸
            模型+参数
            最后的输出路径
@Time    : 2022/3/28
@Author  : Lin Zhenzhe, Zhang Shuyi
@modified by: https://github.com/jacobgil/pytorch-grad-cam
"""
import argparse
import os.path

import cv2
import numpy as np
import torch
from pytorch_grad_cam import GradCAM, ScoreCAM, GradCAMPlusPlus, AblationCAM, XGradCAM, EigenCAM, EigenGradCAM, \
    LayerCAM, FullGrad
from pytorch_grad_cam import GuidedBackpropReLUModel
from pytorch_grad_cam.utils.image import show_cam_on_image, deprocess_image, preprocess_image
from torchvision import models
# from data.utils import get_allfile
from tqdm import tqdm
import torch.nn as nn
from torch.utils.data import DataLoader
from dataloaders.dataset import VideoDataset
from glob import glob
from utils import list2excel


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--use-cuda', action='store_true', default=True, help='Use NVIDIA GPU acceleration')
    parser.add_argument('--image-path', type=str, default='./data/dataset', help='Input image path')
    # checkpoints / 123
    parser.add_argument('--aug_smooth', action='store_true', default=False,
                        help='Apply test time augmentation to smooth the CAM')
    parser.add_argument('--eigen_smooth', action='store_true', default=True,
                        help='Reduce noise by taking the first principle componenet of cam_weights*activations')
    parser.add_argument('--method', type=str, default='eigengradcam',
                        choices=['gradcam', 'gradcam++', 'scorecam', 'xgradcam', 'ablationcam',
                                 'eigencam', 'eigengradcam', 'layercam', 'fullgrad'],
                        help='Can be gradcam/gradcam++/scorecam/xgradcam/ablationcam/eigencam/eigengradcam/layercam')
    parser.add_argument('--resize', action='store_true', default=(512, 512))
    parser.add_argument('--batch_size', action='store_true', default=1, help='input_batchsize')
    parser.add_argument('--gpu_id', action='store_true', default=[2, 3], help='which gpu use')
    args = parser.parse_args()
    args.use_cuda = args.use_cuda and torch.cuda.is_available()
    if args.use_cuda:
        torch.cuda.set_device(0)
        print('Using GPU for acceleration')
    else:
        print('Using CPU for computation')
    return args


if __name__ == '__main__':
    """ python cam.py -image-path <path_to_image>
    Example usage of loading an image, and computing:
        1. CAM
        2. Guided Back Propagation
        3. Combining both
    """

    args = get_args()
    i = 4
    e = 2
    task = ['zhushi', 'saoshi', 'fanxiangsaoshi', 'shuipinggensui', 'shuzhigensui'][i]
    # datadir = ['./data/finaldata/1zhushi', './data/finaldata/2saoshi',
    #              './data/finaldata/3fanxiangsaoshi', './data/finaldata/4shuipinggensui',
    #              './data/finaldata/5shuzhigensui'][i]
    exter = ['huiai_data', 'eryuan_data', 'zoc318_data', ''][e]
    datadir = './data/exter_val/'+exter+['/1zhushi', '/2saoshi', '/3fanxiangsaoshi', '/4shuipinggensui', '/5shuzhigensui'][i]
    model_path = ['./model/1zhushi/P3D-zhushi_epoch-175.pth',
                  './model/2saoshi/P3D-saoshi_epoch-81.pth',
                  './model/3fanxiangsaoshi/P3D-fanxiangsaoshi_epoch-435.pth',
                  './model/4shuipinggensui/P3D-shuipinggensui_epoch-127.pth',
                  './model/5shuzhigensui/P3D-shuzhigensui_epoch-70.pth'][i]

    val_dataloader = DataLoader(VideoDataset(dataset=datadir, split='val'), batch_size=args.batch_size,
                                num_workers=0, drop_last=False)
    methods = \
        {"gradcam": GradCAM,
         "scorecam": ScoreCAM,
         "gradcam++": GradCAMPlusPlus,
         "ablationcam": AblationCAM,
         "xgradcam": XGradCAM,
         "eigencam": EigenCAM,
         "eigengradcam": EigenGradCAM,
         "layercam": LayerCAM,
         "fullgrad": FullGrad
         }

    model = torch.load(model_path).eval()

    # Choose the target layer you want to compute the visualization for.
    # Usually this will be the last convolutional layer in the model.
    # Some common choices can be:
    # Resnet18 and 50: model.layer4
    # VGG, densenet161: model.features[-1]
    # mnasnet1_0: model.layers[-1]
    # You can print the model to help chose the layer
    # You can pass a list with several target layers,
    # in that case the CAMs will be computed per layer and then aggregated.
    # You can also try selecting all layers of a certain type, with e.g:
    # from pytorch_grad_cam.utils.find_layers import find_layer_types_recursive
    # find_layer_types_recursive(model, [torch.nn.ReLU])

    # different net has different layer

    target_layers = [model.module.layer4[2].bn4]  # P3D

    filename_list, label_list, pred_list = ['filename'], ['label'], ['prediction']
    for inputs, labels, filename in tqdm(val_dataloader):
        # move inputs and labels to the device the training is taking place on

        input_tensor = inputs.cuda()
        output = model(input_tensor)
        pred = torch.max(nn.Softmax(dim=1)(output), 1)[1]
        input_mean_img = 0
        for img in glob(os.path.join(filename[0], '*.jpg')):
            input_mean_img += cv2.imdecode(np.fromfile(img, dtype=np.uint16), -1) / len(
                glob(os.path.join(filename[0], '*.jpg')))
        if '/0/' in filename[0]:
            img_name = filename[0].split('/0/')[-1]
        else:
            img_name = filename[0].split('/1/')[-1]
        # We have to specify the target we want to generate
        # the Class Activation Maps for.
        # If targets is None, the highest scoring category (for every member in the batch) will be used.
        # You can target specific categories by
        # targets = [e.g ClassifierOutputTarget(281)]
        targets = None

        # Using the with statement ensures the context is freed, and you can
        # recreate different CAM objects in a loop.

        cam_algorithm = methods[args.method]
        with cam_algorithm(model=model,
                           target_layers=target_layers,
                           use_cuda=args.use_cuda) as cam:
            # AblationCAM and ScoreCAM have batched implementations.
            # You can override the internal batch size for faster computation.
            cam.batch_size = 1
            grayscale_cam = cam(input_tensor=input_tensor,
                                targets=targets,
                                aug_smooth=args.aug_smooth,
                                eigen_smooth=args.eigen_smooth)

            # Here grayscale_cam has only one image in the batch
            grayscale_cam = grayscale_cam[0, :]

            cam_image = show_cam_on_image(input_mean_img / 255, grayscale_cam, use_rgb=True)

            # cam_image is RGB encoded whereas "cv2.imwrite" requires BGR encoding.
            cam_image = cv2.cvtColor(cam_image, cv2.COLOR_RGB2BGR)

        # gb_model = GuidedBackpropReLUModel(model=model, use_cuda=args.use_cuda)
        # gb = gb_model(input_tensor, target_category=None)
        #
        # cam_mask = cv2.merge([grayscale_cam, grayscale_cam, grayscale_cam])
        # cam_gb = deprocess_image(cam_mask * gb)
        # gb = deprocess_image(gb)
        # cv2.imencode('.jpg', input_mean_img)[1].tofile(f'./cam_visualize/{task}/{img_name}.jpg')
        # cv2.imencode('.jpg', cam_image)[1].tofile(f'./cam_visualize/{task}/{img_name}_{str(labels.tolist()[0])}_cam.jpg')


        filename_list.append(img_name)
        label_list.append(str(labels.tolist()[0]))
        pred_list.append(str(pred.tolist()[0]))
    list2excel(exter[:-5]+'_' + task, [filename_list, label_list, pred_list])
