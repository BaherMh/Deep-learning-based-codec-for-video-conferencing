import os
import torch
import imageio
from skimage.transform import resize
import cv2
from PSNR import PSNR, psnr_videos
from tqdm import tqdm
import numpy as np
from scipy.spatial import ConvexHull
from modules.inpainting_network import InpaintingNetwork
from modules.keypoint_detector import KPDetector
from modules.dense_motion import DenseMotionNetwork
from modules.avd_network import AVDNetwork
import yaml
from KeypointsCompressor import compress_keypoints, decompress_keypoints

from tpsmmDemo import my_make_animation
from torchvision.transforms import transforms
from feedback.FeedbackNet import FeedbackNet
from torch.autograd import Variable
from skimage import img_as_ubyte

# a function to load the Generator and Keypoint Detector

def load_checkpoints(config_path, checkpoint_path, device):
    with open(config_path) as f:
        config = yaml.safe_load(f)

    inpainting = InpaintingNetwork(**config['model_params']['generator_params'],
                                        **config['model_params']['common_params'])
    kp_detector = KPDetector(**config['model_params']['common_params'])
    dense_motion_network = DenseMotionNetwork(**config['model_params']['common_params'],
                                              **config['model_params']['dense_motion_params'])
    avd_network = AVDNetwork(num_tps=config['model_params']['common_params']['num_tps'],
                             **config['model_params']['avd_network_params'])

    kp_detector.to('cuda')
    dense_motion_network.to('cuda')
    inpainting.to('cuda')
    avd_network.to('cuda')

    checkpoint = torch.load(checkpoint_path, map_location=device)

    inpainting.load_state_dict(checkpoint['inpainting_network'])
    kp_detector.load_state_dict(checkpoint['kp_detector'])
    dense_motion_network.load_state_dict(checkpoint['dense_motion_network'])
    if 'avd_network' in checkpoint:
        avd_network.load_state_dict(checkpoint['avd_network'])

    inpainting.eval()
    kp_detector.eval()
    dense_motion_network.eval()
    avd_network.eval()

    return inpainting, kp_detector, dense_motion_network, avd_network




def apply_codec(video, alpha, output_path, quantization, gap):
    inpainting, kp_detector, dense_motion_network, avd_network = load_checkpoints(config_path = "config/vox-256.yaml", checkpoint_path = "checkpoints/vox.pth.tar", device = torch.device('cuda'))
    transformer=transforms.Compose([
         transforms.ToTensor(),  #0-255 to 0-1, numpy to tensors
 ])
    checkpoint=torch.load('checkpoints/feedback_checkpoints.model')
    model=FeedbackNet()
    model.load_state_dict(checkpoint)
    model.eval()
    video_size = os.path.getsize(video) * 8
    psnr = 0
    with torch.no_grad():
        reader = imageio.get_reader(video)
        
        
        
        fps = reader.get_meta_data()['fps']
        real_percentage = 0
        video_resized = []
        try:
            for im in reader:
                video_resized.append(im)
        except RuntimeError:
            pass
        reader.close()
        video = video_resized
        video_resized = [resize(frame, (256, 256))[..., :3] for frame in video_resized]
        resulted_frames = []
        num_of_bits = 0
        for frame_idx in tqdm(range(len(video_resized))):
            frame_0_1 = video_resized[frame_idx]
            frame_0_255 = cv2.resize(video[frame_idx], (256, 256))
            
            
            
            #encoder_code
            
            if real_percentage > alpha or frame_idx%gap != 0:
                driving = torch.tensor(frame_0_1[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                driving = driving.cuda()
                kp_driving = kp_detector(driving)
                keys_compressed, key_bits = compress_keypoints(kp_driving, kp_source, quantization)
                num_of_bits += key_bits
            else:
                img_encode = cv2.imencode('.jpg', frame_0_255)[1]
                image_size = len(img_encode) * 8
                if frame_idx != 0:
                    num_of_bits += image_size
                source = torch.tensor(frame_0_1[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                source = source.cuda()
                kp_source = kp_detector(source)
                
            #end of encoder code
            
            
            #decoder code
            #if there is no need to update reference frame 
            if real_percentage > alpha or frame_idx%gap != 0:
                decoder_kp_driving = decompress_keypoints(keys_compressed, decoder_kp_source, quantization)
                reconstructed = my_make_animation(decoder_source, decoder_kp_driving, inpainting, kp_detector, dense_motion_network, avd_network, device = 'cuda', mode = 'relative')
                image_tensor = transformer(cv2.cvtColor(reconstructed, cv2.COLOR_BGR2RGB)).float()
                image_tensor = image_tensor.unsqueeze_(0)
                image_tensor.cuda()
                input = Variable(image_tensor)
                ROD = model(input)  #Rate Of Distortion
                real_percentage = (ROD.cpu().detach().numpy()[0][0])
                resulted_frames.append(reconstructed)
                psnr += PSNR(reconstructed*255, frame_0_255)
                
            #else if the reconstructed image is highly distorted, update the refernce frame
            else:
                decoder_source = cv2.imdecode(img_encode, cv2.IMREAD_COLOR)
                psnr += PSNR(decoder_source, frame_0_255)
                decoder_source = decoder_source/255.0
                src = torch.tensor(decoder_source[np.newaxis].astype(np.float32)).permute(0, 3, 1, 2)
                src = src.cuda()
                decoder_kp_source = kp_detector(src)
                real_percentage = 1
                resulted_frames.append(decoder_source)
                
        #compression rate is the ratio between the bits nedded to transfer video if streamed with no kind of compression divided by the 
        # the bits needed to transfer video in the proposed codec
        uncompressed_size = 256*256*3*8*len(video_resized)
        compression_rate = uncompressed_size/num_of_bits
        bit_rate = (num_of_bits/len(video_resized))*fps
        psnr_avg = psnr/len(video_resized)
        print("bits is: ", num_of_bits)
        print("psnr is: ", psnr/len(video_resized))
        print("compression increase = ", video_size/num_of_bits)
        print("compression rate = ", uncompressed_size/num_of_bits)
        print("bitrate is: ", (num_of_bits/len(video_resized))*fps)
        #end of decoder code
        
    imageio.mimsave(output_path, [img_as_ubyte(frame) for frame in resulted_frames], fps=fps)
                
    return compression_rate, bit_rate, psnr_avg, num_of_bits, len(video_resized)
        