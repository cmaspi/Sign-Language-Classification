Run build.ipynb to see the code working

# How does it work?
1. A CNN is trained on sign detection dataset to classify the sign
2. Webcam video is taken using opencv
3. frame rate is set to 0.5 fps to allow the user to change the hand gesture
4. center square portion of the video feed is cropped to be processed.
5. using mediapipe, get the coordinates for the hand
6. crop out only the hand portion from the image
7. Use chan vese image segmentation to create a binary mask for hand
8. use the trained model to classify the sign.

## Failed method
I tried to use the oxford hand detection dataset to train a model to return bounding box instead of using mediapipe api. This failed because in the oxford hand segmentation dataset there are only about 254 samples in training set which have only one annotation. More images could be obtained which have only one annotation by cropping images, however in the interest of time and lack of compute I used the pretrained model offered by mediapipe.

# Is the model Robust?
No, if you watch the `sample_video.webm`, the model often fails on some signs. Moreover, if some portion of the hand is not in the frame, mediapipe api would return an error. Chan Vese image segmentation works good only is it is "easy" to separate the hand from the background. The hand should also be well lit to avoid incorrect segmentation.