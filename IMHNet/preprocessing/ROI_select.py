import os
import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import math
import matplotlib.image as mpimg
total_image = None
folder_path = r"./data/total_label" 

for filename in os.listdir(folder_path):
    if filename.endswith('.png') or filename.endswith('.jpg'):
        image_path = os.path.join(folder_path, filename)
        gray_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        gray_image = gray_image/255
        if total_image is None:
            total_image = gray_image
        else:
            total_image = cv2.add(total_image, gray_image)

cv2.imwrite("./split_train.png", total_image)


image=Image.open(r"./split_train.png")
image=image.convert("L")
width,height=image.size
cor_value_x=0
cor_value_y=0
pixel_total=0
for i in range(width):
    for j in range (height):
        
        pixel_value=image.getpixel((i,j))
        
        cor_value_x=cor_value_x+i*pixel_value
        cor_value_y = cor_value_y + j * pixel_value
        pixel_total=pixel_total+pixel_value
x_centroid=math.ceil(cor_value_x/pixel_total)
y_centroid=math.ceil(cor_value_y/pixel_total)
with open("./centroid.txt", "a") as file:
    file.write("split_train_x: {}".format(x_centroid))
    file.write("split_train_y: {}".format(y_centroid))
