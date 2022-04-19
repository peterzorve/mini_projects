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
fn.preprocess_plot_resize(path)


# path = "C:/Users/Omistaja/Desktop/deep_learning_images/green_screen.png"
# fn.green_background(path)