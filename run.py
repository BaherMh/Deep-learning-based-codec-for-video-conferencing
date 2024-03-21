
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
import cv2
from skimage.transform import resize
import socket, pickle,struct,imutils, yaml
import numpy as np
from argparse import ArgumentParser
import matplotlib
matplotlib.use('Agg')
import os, sys
from tqdm import tqdm
from skimage import img_as_ubyte
from codec import apply_codec

from golomb_coding import golomb_coding, golomb_decoding


from scipy.spatial import ConvexHull
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.avd_network import AVDNetwork
from _thread import *
import threading

import math




from PSNR import PSNR, psnr_videos

import torch
import torch.nn as nn
from torchvision.transforms import transforms
import numpy as np
from torch.autograd import Variable
from torchvision.models import squeezenet1_1
import torch.functional as F
from io import open
from PIL import Image
import pathlib
import glob
import imageio

       

if __name__ == "__main__":
    print(torch.cuda.is_available())
    if sys.version_info[0] < 3:
        raise Exception("You must use Python 3 or higher. Recommended version is Python 3.9")

    parser = ArgumentParser()
    parser.add_argument("--quantization", default=100.0, help="quantization parameter for golomb coding")
    parser.add_argument("--alpha", default=0.2, help= "alpha parameter that determines the thresold for the feedback")
    parser.add_argument("--output_path", default='compressed.mp4', help="output file path")
    parser.add_argument("--video_file", help="the video to be compressed")
    parser.add_argument("--gap", default=5, help="check the quality of the image every $gap$ frame")

    opt = parser.parse_args()

   
    alpha = float(opt.alpha)
    quantization = float(opt.quantization)
    video_file = opt.video_file
    output_path = opt.output_path
    gap = int(opt.gap)
    video = video_file
    compression_rate, bitrate, psnr_vid, total_bits, total_len  = apply_codec(video, alpha, output_path, quantization, gap)
   
    print("psnr videos is: ", psnr_videos(output_path, video))
    print("bpf", total_bits/total_len)
    print("compression_rate", compression_rate)
   