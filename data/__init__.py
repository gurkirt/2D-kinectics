import numpy as np
import cv2
import torch


def base_transform(image, size, mean):
    #print(image.shape)
    x = cv2.resize(image, (size, size)).astype(np.float32)
    # x = cv2.resize(np.array(image), (size, size)).astype(np.float32)
    x -= mean
    x = x.astype(np.float32)
    x = x[:, :, (2, 1, 0)]
    x = torch.from_numpy(x).permute(2, 0, 1)
    return x

class BaseTransform:
    def __init__(self, size, mean):
        self.size = size
        self.mean = np.array(mean, dtype=np.float32)

    def __call__(self, image):
        return base_transform(image, self.size, self.mean)