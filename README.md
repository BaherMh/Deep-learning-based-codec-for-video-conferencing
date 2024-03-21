# Deep-learning-based-codec-for-video-conferencing
we implemented a deep learning based video compression system for video conferencing over the internet, allowing smooth chat even when the internet connection is slow. At first, when connection is made, a reference image is sent to the receiver, in the next frames we only send the keypoints of the face, which is much smaller in size compared to the entire frame, and the receiver reconstructs the image using these landmarks and the reference image, which is the key idea of the compression in our system. To maintain the accuracy of the image being reconstructed, we proposed a feedback network in decoder side, which takes the reconstructed image, and outputs a number predicting the rate of distortion in that image and send it back to the encoder. The encoder then decides whether to update the reference image or continue sending landmarks, by comparing this value with a predetermined threshold.
Evaluation experiments demonstrates that we can save up to 73% of bitrate compared to HEVC with almost the same image quality. 
the system is implemented in python and pytorch following this schema.

![codec - Copy](https://github.com/BaherMh/Deep-learning-based-codec-for-video-conferencing/assets/105556066/c93570f7-2986-4309-aa8d-eab1f39582e9)

# pretrained chackpoints
we build our system using the repository Thin-Plate-Spline-Motion-Model, and you can found the pretrained weights in their project link <a href="https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model" target="_blank">Thin-Plate-Spline-Motion-Model</a>
# Animation Demo

the parameters you can specify before running are "video_file" which corresponds to the path of the video you want to compress, "quantization" which is the quantization factor used in the algorithm of compressing keypoints before sending which is golomb coding, "output_path" which is the path to the result video, and "gap" which specify every how many frames the feedback network work and send back the rate of distortion in the reconstructed image.

```console
python run.py --video_file path/to/video/mp4 --quantization 100 --output_path path/to/result.mp4 --gap 5
```
or you can shortly use this command, because other parameters have default value

```console
python run.py --video_file path/to/video/mp4
```

# Feedback Netwrok
we have already trained the feed back netwoek on our self-made dataset (we provide the saved weights in the checkpoints folder) , if you have questions about it please feel free to ask, or you can train it on your own using this command

```console
python feedback/TrainFeedbackNet.py --train_path path/to/train/folder --test_path path/to/test/folder 
```

# Acknowledgments
this code is build upon <a href="https://github.com/yoyo-nb/Thin-Plate-Spline-Motion-Model" target="_blank">Thin-Plate-Spline-Motion-Model</a>
Thanks for your amazing work!




