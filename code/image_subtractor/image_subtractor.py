import cv2
import numpy as np
import os

avg_width = 20
avg_height = 20

image1_name = './images/images_63796657_20180119143035_IMAG0089-100-89.JPG'
image2_name = './images/images_63796888_20180119143134_IMAG0090-100-90.JPG'
#image2_name = './images/images_63798188_20180119143733_IMAG0096-100-96.JPG'
#image2_name = './images/images_63825977_20180119171133_IMAG0250-100-250.JPG'
img1 = cv2.imread(image1_name)
img2 = cv2.imread(image2_name)

# Verify images have the same size
# TODO: Maybe resize the larger image to the smaller image??

print("Y, X, Channels: " + str(img1.shape))

if img1.shape != img2.shape:
    print("Unable to process these two images, they are of different dimensions")
    print(image1_name)
    print(image2_name)
    exit(1)

# Create a blank image to write to
blank_image = np.zeros((img1.shape[0], img1.shape[1], 3), np.uint8)

# Loop over the entire image, section by section
for y in range(0, img1.shape[0]-avg_height):
    for x in range(0, img1.shape[1]-avg_width):
        # Get the 20x20 area (for image 1)
        region1 = img1[y:y+avg_height, x:x+avg_width]  # img1[y1:y2, x1:x2]

        # Get the average color for all pixels in the region1
        avg_color_per_row1 = np.average(region1, axis=0)
        avg_color1 = np.average(avg_color_per_row1, axis=0)

        # Get the 20x20 area (for image 2)
        region2 = img2[y:y + avg_height, x:x + avg_width]  # img1[y1:y2, x1:x2]

        # Get the average color for all pixels in the region2
        avg_color_per_row2 = np.average(region2, axis=0)
        avg_color2 = np.average(avg_color_per_row2, axis=0)

        # Subtract the two colors and that becomes the new color
        avg_subtracted = np.absolute(np.subtract(avg_color1, avg_color2))
        blank_image[y+10, x+10] = avg_subtracted
        #print(avg_color_per_row)
        #print(avg_color)

cv2.imshow('image', blank_image)
cv2.waitKey(0)

cv2.destroyAllWindows()
