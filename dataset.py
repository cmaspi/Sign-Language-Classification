import numpy as np
from PIL import Image
from tqdm import tqdm
import os


Images = []
labels = []
Data_dir = './Dataset/train/'
for i in tqdm(os.listdir(Data_dir)):
    path = f'{Data_dir}/{i}/'
    for img_name in os.listdir(path):
        img_path = f'{path}/{img_name}'
        img = Image.open(img_path)
        img = np.asarray(img)
        Images.append(img)
    labels.extend([int(i)]*os.listdir(path).__len__())

Images = np.array(Images)
labels = np.array(labels)

num_Images_per_label = labels.size//np.unique(labels).size

def get_random_uniform(n: int):
    ret_Images, ret_labels = [], []
    
    for label in tqdm(np.unique(labels)):
        indices = np.random.choice(num_Images_per_label, n, replace=False)
        ret_Images.extend(Images[labels == label][indices])
        ret_labels.extend([label]*n)
    return np.array(ret_Images), np.array(ret_labels)
