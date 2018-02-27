import cv2
import numpy as np
from matplotlib import pyplot as plt

# For the connected components, must be greyscale
averaged_image = cv2.imread("../image_subtractor/all_combined.png", cv2.IMREAD_GRAYSCALE)
averaged_image_bgr = cv2.imread("../image_subtractor/all_combined.png")

print("Starting shape", averaged_image.shape)

# Remove the boarder "artifact" created by the previous process
# Which is because we can't average the pixels off the side of the image to make the last 10 pixels.
averaged_image = averaged_image[10:-50, 10:-10]
averaged_image_bgr = averaged_image_bgr[10:-50, 10:-10]
# TODO: Find dynamic way to do this, since the values will be in the previous program
#       perhaps command line arugments or shared preferences file?

print("After border removal", averaged_image.shape)



print(np.max(averaged_image))
print(averaged_image.shape)


def show_hist_gray(image):
    avg_hist = cv2.calcHist([image], [0], None, [16], [0, 16])

    print(avg_hist)

    plt.plot(avg_hist)
    plt.show()
    cv2.waitKey()


#averaged_image = cv2.equalizeHist(averaged_image)
# res = np.hstack((averaged_image_bgr,equ)) #stacking images side-by-side

def try_manual_paint_image(image, psudo_mask, split):
    ## Assign a different colorf to each image
    for i in range(split):
        color = int(255 / split) * i
        image[psudo_mask == i] = (24, 24, color)
        image[psudo_mask == i + 4] = (24, color, 24)
        image[psudo_mask == i + 8] = (color, 24, 24)
        image[psudo_mask == i + 12] = (color, 24, color)

        image[psudo_mask == 16] = (128, 128, 128)

    cv2.imshow('image', image)
    cv2.waitKey()

    return image

def try_cc_with_watershed():
    ret, thresh = cv2.threshold(averaged_image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,5)
    ret, sure_fg = cv2.threshold(dist_transform,0.7*dist_transform.max(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    outarray = []
    outarray = np.asarray(outarray)
    ret, markers = cv2.connectedComponents(averaged_image, outarray, 8)

    markers = markers + 1
    markers[unknown == 255] = 0

    markers = cv2.watershed(averaged_image_bgr, markers)
    averaged_image_bgr[markers == -1] = [255, 0, 0]

    print(markers)
    cv2.imshow('image', markers)
    cv2.waitKey()

def try_cc_alone():
    # Threshold it so it becomes binary
    ret, thresh = cv2.threshold(averaged_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # You need to choose 4 or 8 for connectivity type
    connectivity = 8
    # Perform the operation
    output = cv2.connectedComponentsWithStats(thresh, connectivity, cv2.CV_32S)
    # Get the results
    # The first cell is the number of labels
    num_labels = output[0]
    # The second cell is the label matrix
    labels = output[1]
    # The third cell is the stat matrix
    stats = output[2]
    # The fourth cell is the centroid matrix
    centroids = output[3]

    # Scale from uint32 to uint8
    labels = cv2.convertScaleAbs(labels)
    print("NumLabels", num_labels)
    print(np.max(labels))
    print(labels.dtype)
    show_hist_gray(labels)

    cv2.imshow('image', labels)
    cv2.waitKey()


try_cc_alone()

try_manual_paint_image(averaged_image_bgr, averaged_image, 4)
try_manual_paint_image(averaged_image_bgr, averaged_image, 2)
try_manual_paint_image(averaged_image_bgr, averaged_image, 3)
try_manual_paint_image(averaged_image_bgr, averaged_image, 1)


# Cleanup
cv2.destroyAllWindows()
