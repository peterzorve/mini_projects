import cv2
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 

import torch 
import torch.nn as nn
import torch.nn.functional as F 
import time 
from torchvision import datasets, transforms


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################

""" ANNOTATIONS """

path = "C:/Users/Omistaja/Desktop/deep_learning_images/peterzorve.JPG"

def annotation(path):
     
     bgr_image = cv2.imread(path)
     rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
     gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

     rgb_image_copy1 = rgb_image.copy()
     rgb_image_copy2 = rgb_image.copy()

     gray_image_copy3 = gray_image.copy()
     gray_image_copy3 = cv2.cvtColor(gray_image_copy3, cv2.COLOR_BGR2RGB)

     

     cv2.rectangle( rgb_image_copy1, (200, 400), (2500, 3000), (255, 0, 0), 80) 
     cv2.putText(   rgb_image_copy1, 'My Face', (200, 3500), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 255), 30)

     

     cv2.line(  rgb_image_copy2, (400, 2500),  (2400, 500),  (200, 200, 0), 60)
     cv2.line(  rgb_image_copy2, (2400, 500),  (2400, 2500), (0, 200, 200), 60)
     cv2.line(  rgb_image_copy2, (2400, 2500), (400, 2500),  (200, 0, 200), 60)
     cv2.circle(rgb_image_copy2, (1300, 1400), 1100, (0, 0, 255), 60)

     

     cv2.rectangle( gray_image_copy3, (200, 3000), (2700, 3300), (0, 255, 0), cv2.FILLED)
     cv2.rectangle( gray_image_copy3, (200, 3400), (2700, 3700), (0, 255, 0), cv2.FILLED)
     cv2.rectangle( gray_image_copy3, (200, 3800), (2700, 4100), (0, 255, 0), cv2.FILLED)

     cv2.rectangle( gray_image_copy3, (200, 3000), (2700, 3300), (0, 0, 0), 30)
     cv2.rectangle( gray_image_copy3, (200, 3400), (2700, 3700), (0, 0, 0), 30)
     cv2.rectangle( gray_image_copy3, (200, 3800), (2700, 4100), (0, 0, 0), 30)

     cv2.putText(   gray_image_copy3, 'Dear Dad,',                    (260, 3210), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 0), 25)
     cv2.putText(   gray_image_copy3, 'Incase you are reading this,', (260, 3610), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 0), 25)
     cv2.putText(   gray_image_copy3, 'I miss you',                   (260, 4010), cv2.FONT_HERSHEY_PLAIN, 10, (0, 0, 0), 25)


     fig = plt.figure()
     plt.figure(figsize=(30, 20))

     plt.subplot(1, 4, 1)
     plt.imshow(rgb_image)
     plt.title('RGB', fontsize=30)

     plt.subplot(1, 4, 2)
     plt.imshow(rgb_image_copy1)
     plt.title('ANNOTATION 1', fontsize=20)

     plt.subplot(1, 4, 3)
     plt.imshow(rgb_image_copy2)
     plt.title('ANNOTATION 2', fontsize=20)

     plt.subplot(1, 4, 4)
     plt.imshow(gray_image_copy3) #, cmap='gray')
     plt.title('ANNOTATION 2', fontsize=20)

     plt.show()

     return 


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################

path = "C:/Users/Omistaja/Desktop/deep_learning_images/peterzorve.JPG"

def bgr_rgb_gray(path):
     bgr_image = cv2.imread(path)
     rgb_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)
     gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)

     
     fig = plt.figure()
     plt.figure(figsize=(30, 20))

     plt.subplot(1, 5, 1)
     plt.imshow(bgr_image)
     plt.title('BGR', fontsize=30)

     plt.subplot(1, 5, 2)
     plt.imshow(rgb_image)
     plt.title('RGB', fontsize=30)

     plt.subplot(1, 5, 3)
     plt.imshow(gray_image, cmap='gray')
     plt.title('GRAY', fontsize=30)

     plt.show()



#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################


def color_spaces():
     bgr_color_space_1 = cv2.imread("C:/Users/Omistaja/Desktop/deep_learning_images/color_space_1.png")
     bgr_color_space_2 = cv2.imread("C:/Users/Omistaja/Desktop/deep_learning_images/color_space_2.png")
     bgr_color_space_3 = cv2.imread("C:/Users/Omistaja/Desktop/deep_learning_images/color_space_3.png")

     

     rgb_color_space_1 = cv2.cvtColor(bgr_color_space_1, cv2.COLOR_BGR2RGB)
     rgb_color_space_2 = cv2.cvtColor(bgr_color_space_2, cv2.COLOR_BGR2RGB)
     rgb_color_space_3 = cv2.cvtColor(bgr_color_space_3, cv2.COLOR_BGR2RGB)

     

     fig = plt.figure()
     plt.figure(figsize=(30, 20))

     plt.subplot(1, 5, 1)
     plt.imshow(rgb_color_space_1)
     plt.title('COLOR BOX', fontsize=30)

     plt.subplot(1, 5, 2)
     plt.imshow(rgb_color_space_2)
     plt.title('COLOR SPACES (HSV SCALE)', fontsize=20)

     plt.subplot(1, 5, 3)
     plt.imshow(rgb_color_space_3)
     plt.title('COLOR SPACES (HSV SCALE)', fontsize=20)

     plt.show()

     return


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################

path = "C:/Users/Omistaja/Desktop/deep_learning_images/peterzorve.JPG"

def thresholding(path):

     def im_show(image):
          return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

     bgr_peter       = cv2.imread(path)
     rgb_peter       = cv2.cvtColor(bgr_peter, cv2.COLOR_BGR2RGB)
     gray_peter      = cv2.cvtColor(bgr_peter, cv2.COLOR_BGR2GRAY)

     

     retval_1, dst_1 = cv2.threshold(gray_peter, 127, 255, cv2.THRESH_BINARY)
     retval_2, dst_2 = cv2.threshold(gray_peter, 127, 255, cv2.THRESH_BINARY_INV)
     retval_3, dst_3 = cv2.threshold(gray_peter, 127, 255, cv2.THRESH_TOZERO)
     retval_4, dst_4 = cv2.threshold(gray_peter, 127, 255, cv2.THRESH_TOZERO_INV)
     retval_5, dst_5 = cv2.threshold(gray_peter, 127, 255, cv2.THRESH_TRUNC)
     retval_6, dst_6 = cv2.threshold(gray_peter, 50,  255, cv2.THRESH_BINARY)
     retval_7, dst_7 = cv2.threshold(gray_peter, 50,  255, cv2.THRESH_BINARY_INV)

     

     gray_peter = im_show(gray_peter)
     dst_1 = im_show(dst_1)
     dst_2 = im_show(dst_2)
     dst_3 = im_show(dst_3)
     dst_4 = im_show(dst_4)
     dst_5 = im_show(dst_5)
     dst_6 = im_show(dst_6)
     dst_7 = im_show(dst_7)

     

     all_images  = [bgr_peter,  rgb_peter,  gray_peter,   dst_1,  dst_2,   dst_3,   dst_4,   dst_5, dst_6, dst_7]
     type_images = ['BGR', 'RGB', 'GRAY', 'THRESH_BINARY - (127, 255)', 'THRESH_BINARY_INV - (127, 255)', 'THRESH_TOZERO', 'THRESH_TOZERO_INV', 'THRESH_TRUNC', 'THRESH_BINARY - (50, 255)', 'THRESH_BINARY_INV - (50, 255)']

     

     fig = plt.figure()
     plt.figure(figsize=(20, 12))

     for i in range(1, len(all_images) + 1 ):
          plt.subplot(2, 5, i)
          plt.imshow(all_images[i-1])
          plt.title(type_images[i-1])
     
     plt.show()
          
     return 


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################


path1 = 'C:/Users/Omistaja/Desktop/deep_learning_images/hand_written_notes.png'
path2 = 'C:/Users/Omistaja/Desktop/deep_learning_images/leaf.jpg'

def more_thresholding(path1, path2):
     def im_show(image):
          return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

     
     bgr_notes = cv2.imread(path1)
     rgb_notes = cv2.cvtColor(bgr_notes, cv2.COLOR_BGR2RGB)
     gray_notes = cv2.cvtColor(rgb_notes, cv2.COLOR_BGR2GRAY)
     adaptive_thresh_mean_c = cv2.adaptiveThreshold(gray_notes, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 7, 3)
     adaptive_thresh_gaussian = cv2.adaptiveThreshold(gray_notes, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
     ret_1, otsu_notes = cv2.threshold(gray_notes, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

     

     bgr_leaf     = cv2.imread(path2)
     rgb_leaf    = cv2.cvtColor(bgr_leaf, cv2.COLOR_BGR2RGB)
     gray_leaf    = cv2.cvtColor(bgr_leaf, cv2.COLOR_BGR2GRAY)
     ret_1, otsu_leaf = cv2.threshold(gray_leaf, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

     

     gray_notes = im_show(gray_notes)
     adaptive_thresh_mean_c = im_show(adaptive_thresh_mean_c)
     adaptive_thresh_gaussian = im_show(adaptive_thresh_gaussian)
     otsu_notes = im_show(otsu_notes)
     gray_leaf = im_show(gray_leaf)
     otsu_leaf    = im_show(otsu_leaf)

     

     fig = plt.figure()
     plt.figure(figsize=(22, 13))

     plt.subplot(3, 3, 1)
     plt.imshow(rgb_notes)
     plt.title('RGB', fontsize=20)

     plt.subplot(3, 3, 2)
     plt.imshow(gray_notes)
     plt.title('GRAY', fontsize=20)

     plt.subplot(3, 3, 3)
     plt.imshow(adaptive_thresh_mean_c)
     plt.title('ADAPTIVE THRESH (MEAN_C)', fontsize=20)

     plt.subplot(3, 3, 4)
     plt.imshow(adaptive_thresh_gaussian)
     plt.title('ADAPTIVE THRESH (GAUSSIAN)', fontsize=20)

     plt.subplot(3, 3, 6)
     plt.imshow(otsu_notes)
     plt.title('OTSU', fontsize=20)

     plt.subplot(3, 3, 7)
     plt.imshow(rgb_leaf)
     plt.title('RGB', fontsize=20)

     plt.subplot(3, 3, 8)
     plt.imshow(gray_leaf)
     plt.title('GRAY', fontsize=20)

     plt.subplot(3, 3, 9)
     plt.imshow(otsu_leaf)
     plt.title('OTSU', fontsize=20)

     plt.show()

     return 

#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################


path = "C:/Users/Omistaja/Desktop/deep_learning_images/peterzorve.JPG"

def morphological_transformations(path):
     bgr_peter       = cv2.imread(path)
     rgb_peter       = cv2.cvtColor(bgr_peter, cv2.COLOR_BGR2RGB)
     gray_peter      = cv2.cvtColor(bgr_peter, cv2.COLOR_BGR2GRAY)

     

     def im_show(image):
          return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

     

     kernel = np.ones((3, 3))
     kernel_1 = np.ones((13, 13))



     ret, threshold = cv2.threshold(gray_peter, 200, 255, cv2.THRESH_BINARY)


     dilation = cv2.dilate(threshold, kernel, iterations = 5)
     erosion  = cv2.erode(threshold, kernel, iterations = 5)
     opening = cv2.morphologyEx(threshold, cv2.MORPH_OPEN,  kernel_1, iterations = 3)
     closing = cv2.morphologyEx(threshold, cv2.MORPH_CLOSE, kernel_1, iterations = 3)



     gray_peter = im_show(gray_peter)
     threshold  = im_show(threshold)
     dilation   = im_show(dilation)
     erosion    = im_show(erosion)
     opening    = im_show(opening)
     closing    = im_show(closing)


     fig = plt.figure()
     plt.figure(figsize=(30, 20))

     plt.subplot(2, 4, 1)
     plt.imshow(rgb_peter)
     plt.title('RGB', fontsize=30)

     plt.subplot(2, 4, 2)
     plt.imshow(gray_peter)
     plt.title('GRAY', fontsize=30)

     plt.subplot(2, 4, 3)
     plt.imshow(threshold)
     plt.title('THRESHOLD', fontsize=30)

     plt.subplot(2, 4, 5)
     plt.imshow(dilation)
     plt.title('DILATION', fontsize=30)

     plt.subplot(2, 4, 6)
     plt.imshow(erosion)
     plt.title('EROSION', fontsize=30)

     plt.subplot(2, 4, 7)
     plt.imshow(opening)
     plt.title('OPENING', fontsize=30)

     plt.subplot(2, 4, 8)
     plt.imshow(closing)
     plt.title('CLOSING', fontsize=30)

     plt.show()
     
     return 




#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################


path = "C:/Users/Omistaja/Desktop/deep_learning_images/peterzorve.JPG"

def filters_kernels(path):

     def im_show(image):
          return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    

     bgr_image  = cv2.imread(path)
     rgb_image  = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2RGB)                             
     gray_image = cv2.cvtColor(bgr_image, cv2.COLOR_BGR2GRAY)                               

  

     kernel_blur = np.ones((51, 51), np.float32)/(51*51)
     blur_image_with_kernel  = cv2.filter2D(rgb_image, -1, kernel_blur)                      



     blur_image    = cv2.blur(rgb_image, (51, 51))                                             
     blur_gaussian = cv2.GaussianBlur(rgb_image, (51, 51), 0)                                 
     blur_median   = cv2.medianBlur(rgb_image, 51)                                           


                                                                                
     kernel_sharp = np.ones((51, 51), np.float32) * -1                          
     kernel_sharp[26, 26] = 51*51                                               
     image_sharp  = cv2.filter2D(rgb_image, -1, kernel_sharp)                                  



     kernel_edge = np.ones((51, 51), np.float32) * -1
     kernel_edge[26, 26] = (51*51) - 1
     image_edge  = cv2.filter2D(gray_image, -1, kernel_edge)                                 

     kernel_sobel_x = np.array([   [-30, 0, 30], [-60, 0, 60], [-30, 0, 30]],   np.float32)
     kernel_sobel_y = np.array([   [30, 60, 30],  [0,  0,  0], [-30, -60, -30]], np.float32)
     image_sobel_x  = cv2.filter2D(gray_image, -1, kernel_sobel_x)                             
     image_sobel_y  = cv2.filter2D(gray_image, -1, kernel_sobel_y)                            

     canny = cv2.Canny(gray_image, 100, 200)                                                

     gray_image     = im_show(gray_image)
     image_edge     = im_show(image_edge)
     image_sobel_x  = im_show(image_sobel_x)
     canny          = im_show(canny)

     fig = plt.figure()
     plt.figure(figsize=(30, 18))

     plt.subplot(2, 5, 1)
     plt.imshow(rgb_image)
     plt.title('RGB', fontsize=30)

     plt.subplot(2, 5, 2)
     plt.imshow(gray_image)
     plt.title('GRAY', fontsize=30)

     plt.subplot(2, 5, 3)
     plt.imshow(blur_image_with_kernel)
     plt.title('BLUR USING KERNELS', fontsize=30)

     plt.subplot(2, 5, 4)
     plt.imshow(blur_image)
     plt.title('BLUR IMAGE', fontsize=30)

     plt.subplot(2, 5, 5)
     plt.imshow(blur_gaussian)
     plt.title('GAUSSIAN BLUR', fontsize=30)

     plt.subplot(2, 5, 6)
     plt.imshow(blur_median)
     plt.title('MEDIAN BLUR', fontsize=30)

     plt.subplot(2, 5, 7)
     plt.imshow(image_sharp)
     plt.title('SHARP IMAGE', fontsize=30)

     plt.subplot(2, 5, 8)
     plt.imshow(image_edge)
     plt.title('EDGE IMAGE', fontsize=30)

     plt.subplot(2, 5, 9)
     plt.imshow(image_sobel_x)
     plt.title('SOBEL (X)', fontsize=30)

     plt.subplot(2, 5, 10)
     plt.imshow(canny)
     plt.title('CANNY', fontsize=30)

     plt.show()

     return 


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################


# def web_camera():
     
#      cap = cv2.VideoCapture(0)

#      while True:

#           width  = int(cap.get(3))
#           height = int(cap.get(4))

#           ret, frame = cap.read()

#           image = np.zeros(frame.shape, np.uint8)

#           smaller_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)

#           image[ : height//2,  : width//2] = smaller_frame
#           image[ height//2 :,  : width//2] = smaller_frame
#           image[ : height//2,  width//2 :] = smaller_frame
#           image[ height//2 :,  width//2 :] = smaller_frame

#           cv2.imshow('frame', image)

#           if cv2.waitKey(1) == ord('q'):
#                break 

#      cap.release()
#      cv2.destroyAllWindows()

#      return 


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################


path = 'C:/Users/Omistaja/Desktop/deep_learning_images/candy_1.jpg' 

def candy_ball(path):
     bgr_candy = cv2.imread(path)
     hsv_candy = cv2.cvtColor(bgr_candy, cv2.COLOR_BGR2HSV)
     rgb_candy = cv2.cvtColor(bgr_candy, cv2.COLOR_BGR2RGB)

     lower_range_blue  = np.array([95, 50,  20])
     upper_range_blue  = np.array([140, 255, 255])
     lower_range_green = np.array([40, 50, 20])
     upper_range_green = np.array([80, 255, 255])

     mask_blue  = cv2.inRange(hsv_candy, lower_range_blue,  upper_range_blue)
     mask_green = cv2.inRange(hsv_candy, lower_range_green, upper_range_green)

     rgb_candy_blue  = rgb_candy.copy()
     rgb_candy_green = rgb_candy.copy()

     rgb_candy_blue[mask_blue == 0]   = [0, 0, 0]
     rgb_candy_green[mask_green == 0] = [0, 0, 0]

     combine = rgb_candy_blue + rgb_candy_green

     fig = plt.figure()
     plt.figure(figsize=(30, 12))

     plt.subplot(2, 4, 1)
     plt.imshow(bgr_candy)
     plt.title('BGR', fontsize=20)

     plt.subplot(2, 4, 2)
     plt.imshow(rgb_candy)
     plt.title('RGB', fontsize=20)

     plt.subplot(2, 4, 3)
     plt.imshow(hsv_candy)
     plt.title('HSV', fontsize=20)

     plt.subplot(2, 4, 4)
     plt.imshow(rgb_candy_blue)
     plt.title('BLUE CANDY BALLS', fontsize=20)

     plt.subplot(2, 4, 5)
     plt.imshow(rgb_candy_green)
     plt.title('GREEN CANDY BALLS', fontsize=20)

     plt.subplot(2, 4, 6)
     plt.imshow(combine)
     plt.title('BLUE + GREEN CANDIES', fontsize=20)

     plt.show()

     return 



#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################

path = 'C:/Users/Omistaja/Desktop/deep_learning_images/hand_written_two_2.png'

def preprocess(path):
     bgr_digit  = cv2.imread(path)
     rgb_digit = cv2.cvtColor(bgr_digit, cv2.COLOR_BGR2RGB)
     gray_digit = cv2.cvtColor(bgr_digit, cv2.COLOR_BGR2GRAY)

     ret, threshold  = cv2.threshold(gray_digit, 100, 255, cv2.THRESH_BINARY)
     ret_inv, threshold_inv  = cv2.threshold(gray_digit, 100, 255, cv2.THRESH_BINARY_INV)

     fig = plt.figure()
     plt.figure(figsize=(30, 10))

     plt.subplot(1, 5, 1)
     plt.imshow(rgb_digit)
     plt.title('RGB', fontsize=20)

     plt.subplot(1, 5, 2)
     plt.imshow(threshold, cmap='gray')
     plt.title('THRESHOLD', fontsize=20)

     plt.subplot(1, 5, 3)
     plt.imshow(threshold_inv, cmap='gray')
     plt.title('THRESHOLD', fontsize=20)

     plt.show()

     return 



#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################


path = 'C:/Users/Omistaja/Desktop/deep_learning_images/color_map.png'

def color_map(path):
     color_map = cv2.imread(path)
     color_map_original = cv2.cvtColor(color_map, cv2.COLOR_BGR2RGB)

     plt.figure(figsize=(16, 12))
     plt.imshow(color_map_original)
     plt.title('COLOR MAP', fontsize=40)

     plt.show()
     return 


#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################

# path = 'C:/Users/Omistaja/Desktop/deep_learning_images/hand_written_two_0.png'
# path = 'C:/Users/Omistaja/Desktop/deep_learning_images/hand_written_two_1.png'
path = 'C:/Users/Omistaja/Desktop/deep_learning_images/hand_written_two_2.png'


def preprocess_plot_resize(path):

     """ Read the images and convert it to the various color formats """
     bgr_number  = cv2.imread(path)
     rgb_number  = cv2.cvtColor(bgr_number, cv2.COLOR_BGR2RGB)
     gray_number = cv2.cvtColor(bgr_number, cv2.COLOR_BGR2GRAY)
     

     """ Use the gray color to convert the image to a binary color using the threshold function """
     retvalue, binary_threshold = cv2.threshold(gray_number, 80, 255, cv2.THRESH_BINARY_INV)
     

     """ With the binary colors, find the contours which includes the number """
     coutours, associate = cv2.findContours(binary_threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
     

     """ Sort the contours from biggest to smallest. The biggest contours is probably the one with the digit in it  """
     sorted_contous = sorted(coutours, key=cv2.contourArea, reverse=True)
     

     """ Find the rectangular coordinates of the first contour with the digit written in it """
     x, y, w, h = cv2.boundingRect(sorted_contous[0]) 
     

     """ Find a new coordinates by which to crop the image later """
     boundary_list = [x, y, (gray_number.shape[0] - (x+w)), (gray_number.shape[0] - (y+h))]
     num = int(min(boundary_list)/2)
     

     """ Find the rectangle around the coordinates to be cropped """
     rgb_number_copy3 = rgb_number.copy()
     cv2.rectangle(rgb_number_copy3, (x-num, y-num), (x+w+num, y+h+num), (255, 0, 0), 6)
     

     """ Using the binary image, crop the digit with a little space around it """
     crop_later = binary_threshold.copy()
     cropped_number = crop_later[y-num : y+h+num,  x-num : x+w+num]
     

     """ Resize the cropped image. This image will be passed through our model to make predictions """
     resized_numnber = cv2.resize(cropped_number, (100, 100))
     


     """ """
     rgb_number_copy = rgb_number.copy()
     cv2.drawContours(rgb_number_copy, sorted_contous[0], -1, (255, 0, 0), 6)
     


     rgb_number_copy2 = rgb_number.copy()
     cv2.rectangle(rgb_number_copy2, (x, y), (x+w, y+h), (255, 0, 0), 6)


     fig = plt.figure()
     plt.figure(figsize=(30, 15))

     plt.subplot(2, 4, 1)
     plt.imshow(rgb_number)
     plt.title('Original Image', fontsize=20)

     plt.subplot(2, 4, 2)
     plt.imshow(rgb_number_copy3)
     plt.title('Highlight Image', fontsize=20)

     plt.subplot(2, 4, 3)
     plt.imshow(cropped_number, cmap='gray')
     plt.title('Cropped digit', fontsize=20)

     plt.subplot(2, 4, 4)
     plt.imshow(resized_numnber, cmap='gray')
     plt.title('Resized (100, 100)', fontsize=20)

     plt.show()


     # plt.subplot(2, 4, 5)
     # plt.imshow(gray_number, cmap='gray')
     # plt.title('GRAY', fontsize=20)


     # plt.subplot(2, 4, 6)
     # plt.imshow(binary_threshold)
     # plt.title('BINARY', fontsize=20)


     # plt.subplot(2, 4, 7)
     # plt.imshow(rgb_number_copy)
     # plt.title('', fontsize=20)

     # plt.subplot(2, 4, 8)
     # plt.imshow(rgb_number_copy2)
     # plt.title('', fontsize=20)

     return  # rgb_number, rgb_number_copy3, cropped_number, resized_numnber



#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################



path = "C:/Users/Omistaja/Desktop/deep_learning_images/green_screen.png"

def green_background(path): 
     """ Load Images """
     bgr_fight1 = cv2.imread(path)

     """ Resize the images """
     bgr_fight1 = cv2.resize(bgr_fight1, (500, 500))

     """ Change from BGR to HSV """
     hsv_fight1 = cv2.cvtColor(bgr_fight1, cv2.COLOR_BGR2HSV)

     """ Change from BGR to RGB """
     rgb_fight1 = cv2.cvtColor(bgr_fight1, cv2.COLOR_BGR2RGB)

     """ Set the boundaries """
     lower_bound_green = (0,   0, 0)         # (X, Y, V)
     upper_bound_green = (180, 1, 5)       # (X, Y, V)  

     """ Define the mask"""
     mask_green = cv2.inRange(hsv_fight1, lower_bound_green, upper_bound_green)

     """ Make a copy of the original Data """
     green_back_fight1_mask = rgb_fight1.copy()

     """ Apply the Mask to the image """
     green_back_fight1_mask[mask_green==0] = [0, 255, 0]

     

     """ NEW COLOR """
     lower_bound_blue = (45, 0, 0)         # (X, Y, V)
     upper_bound_blue = (65, 255, 255)       # (X, Y, V)  

     """ Define the mask"""
     mask_blue = cv2.inRange(hsv_fight1, lower_bound_green, upper_bound_green)

     """ Make a copy of the original Data """
     blue_back_fight1_mask = rgb_fight1.copy()

     """ Apply the Mask to the image """
     blue_back_fight1_mask[mask_blue == 0] = [255, 0, 255]

     
     fig = plt.figure()
     plt.figure(figsize=(30, 10))

     plt.subplot(1, 3, 1)
     plt.imshow(rgb_fight1)
     plt.title('ORIGINAL DATA')


     plt.subplot(1, 3, 2)
     plt.imshow(green_back_fight1_mask)
     plt.title('GREEN BACKGROUND')


     plt.subplot(1, 3, 3)
     plt.imshow(blue_back_fight1_mask)
     plt.title('BLUE BACKGROUND')

     plt.show()

     return 



#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################





#################################################################################################################################################
#################################################################################################################################################
#################################################################################################################################################










