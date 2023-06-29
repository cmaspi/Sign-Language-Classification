from PIL import Image
import numpy as np
import os


def get_data():
    Images = []
    imgs_path = './filtered_data/images/'
    for name in range(len(os.listdir(imgs_path))):
        im_path = imgs_path+str(name)+'.png'
        img = Image.open(im_path)
        img = np.asarray(img)
        Images.append(img)

    Images_np = np.array(Images)/255
    # Images_1axis = (0.2126*Images_np[..., 0] + 0.7152*Images_np[..., 1] +
    #                 0.0722*Images_np[..., 2])[..., np.newaxis]

    annotations = np.load('./filtered_data/annotations.npy').astype(float)
    annotations /= 200
    return (Images_np, annotations)
