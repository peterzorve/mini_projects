import cv2
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import torch 
import torch.nn as nn
import torch.nn.functional as F 
import time 
from torchvision import datasets, transforms

import functions as fn 


""" ANNOTATIONS """


# path11 = "C:/Users/Omistaja/Desktop/deep_learning_images/green_screen.png"
# fn.green_background(path11)

path = 'C:/Users/Omistaja/Desktop/deep_learning_images/hand_written_two_2.png'

rgb_image, aa, rgb_digit, cropped_resized = fn.preprocess_plot_resize(path)


fig = plt.figure()
plt.figure(figsize=(15, 8))

plt.subplot(1, 4, 1)
plt.imshow(rgb_image)
plt.title('Original Image', fontsize=20)

plt.subplot(1, 4, 2)
plt.imshow(aa)
plt.title('Highlight Digit', fontsize=20)

plt.subplot(1, 4, 3)
plt.imshow(rgb_digit, cmap='gray')
plt.title('Cropped Area', fontsize=20)

plt.subplot(1, 4, 4)
plt.imshow(cropped_resized, cmap='gray')
plt.title('Cropped digit', fontsize=20)

plt.show()