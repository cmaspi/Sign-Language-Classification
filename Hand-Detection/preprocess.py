import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from skimage import img_as_float
from skimage.segmentation import chan_vese


def sq_crop(img: np.ndarray):
    h, w = img.shape[:2]
    if h > w:
        img = img.transpose([1, 0]+list(range(2, len(img.shape))))
        img = sq_pad(img)
        img = img.transpose([1, 0]+list(range(2, len(img.shape))))
        return img

    crop = (w-h)//2
    return img[:, crop:crop+h]


def sq_pad(img: np.ndarray):
    h, w = img.shape[:2]
    if h > w:
        img = img.transpose([1, 0]+list(range(2, len(img.shape))))
        img = sq_pad(img)
        img = img.transpose([1, 0]+list(range(2, len(img.shape))))
        return img

    pad = (w-h)//2
    pad_img = np.zeros((w, w)+img.shape[2:])
    pad_img[pad:pad+h] = img
    return pad_img


class HandMarker:
    def __init__(self):
        self.base_options = python.BaseOptions(
            model_asset_path='hand_landmarker.task')
        self.options = vision.HandLandmarkerOptions(
            base_options=self.base_options,
            num_hands=2)
        self.detector = vision.HandLandmarker.create_from_options(
            self.options)
        self.pad = 10

    def give_box(self, img: np.ndarray):
        mpImg = mp.Image(image_format=mp.ImageFormat.SRGB,
                         data=img)
        detection_result = self.detector.detect(mpImg)
        xs, ys = [], []
        for coord in detection_result.hand_landmarks[0]:
            xs.append(coord.x)
            ys.append(coord.y)
        sh = img.shape[0]
        xmin = int(min(xs)*sh-self.pad)
        xmax = int(max(xs)*sh+self.pad)
        ymin = int(min(ys)*sh-self.pad)
        ymax = int(max(ys)*sh+self.pad)
        return (xmin, xmax, ymin, ymax)


def bin_mask(img):
    if img.shape[-1] == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img = img_as_float(img)

    cv = chan_vese(img, mu=0.25, lambda1=.6, lambda2=.6, tol=1e-5,
                   max_num_iter=500, dt=0.5, init_level_set="checkerboard",
                   extended_output=True)
    if cv[0].mean() < .5:
        ret = 1-cv[0]
    ret = cv[0]
    return ret
