import numpy as np
import math
import cv2
def PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * math.log10(max_pixel / math.sqrt(mse))
    return psnr
    
    
    
    
    


def psnr_videos(video1, video2):

    # Read the video from specified path 
    vid1 = cv2.VideoCapture(video1)  
    vid2 = cv2.VideoCapture(video2)  
    
    # Find the frame rate of the video 
    fps1 = vid1.get(cv2.CAP_PROP_FPS)  
    fps2 = vid2.get(cv2.CAP_PROP_FPS)  
    
    # Find the number of frames in the video file 
    frameCount1 = int(vid1.get(cv2.CAP_PROP_FRAME_COUNT))  
    frameCount2 = int(vid2.get(cv2.CAP_PROP_FRAME_COUNT))  
    print(frameCount1)
    print(frameCount2)
    # Read the first frame from the videos 
    ret1, frame1 = vid1.read()  
    ret2, frame2 = vid2.read()  
    
    # Calculate PSNR between frames of two videos 
    psnrVals = [] 
    
    for i in range(min([frameCount1, frameCount2])):     
    
        psnrVal = PSNR(frame1, frame2)     
    
        psnrVals += [psnrVal]     
    
        ret1, frame1 = vid1.read()     
    
        ret2, frame2 = vid2.read()
    return np.mean(psnrVals)