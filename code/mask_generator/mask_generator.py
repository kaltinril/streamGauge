import cv2
import numpy as np
from matplotlib import pyplot as plt

# For the connected components, must be greyscale
filename = "../image_subtractor/all_combined2.png"
#filename = "source_image.JPG"
averaged_image = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
averaged_image_bgr = cv2.imread(filename)

print("Starting shape", averaged_image.shape)


# load image and determine threadhold that will sepeate data
# create bands of averaged rows
# generate mask from the banded sections
# HOG code: take top most block and top most block and generate histograms
# grab the same section from all images for the ROI for HOG generation
# Run them through an ANN with 0 for top band and 1 for bottom band



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



def try_manual_paint_image(image, psudo_mask, split):
    img = np.asarray(list(image))
    mask = np.asarray(list(psudo_mask))

    ## Assign a different colorf to each image
    for i in range(split):
        color = int(255 / split) * i
        img[mask == i] = (24, 24, color)
        img[mask == i + 4] = (24, color, 24)
        img[mask == i + 8] = (color, 24, 24)
        img[mask == i + 12] = (color, 24, color)

        img[mask == 16] = (128, 128, 128)

    cv2.imshow('image', img)
    cv2.waitKey()
    cv2.destroyAllWindows()

    return img



def try_cc_with_watershed():
    # https://docs.opencv.org/3.1.0/d3/db4/tutorial_py_watershed.html
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

    cv2.imshow('image', markers)
    cv2.waitKey()
    cv2.destroyAllWindows()

def try_cc_alone():
    # https://stackoverflow.com/questions/35854197/how-to-use-opencvs-connected-components-with-stats-in-python
    # https://docs.opencv.org/3.1.0/d3/dc0/group__imgproc__shape.html#gac2718a64ade63475425558aa669a943a
    # Threshold it so it becomes binary
    ret, thresh = cv2.threshold(averaged_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # You need to choose 4 or 8 for connectivity type
    connectivity =8
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
    #show_hist_gray(labels)

    cv2.imshow('image', labels)
    cv2.waitKey()
    cv2.destroyAllWindows()

def try_k_means(img_color, K = 4):
    # https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html

    Z = img_color.reshape((-1, 3))

    # convert to np.float32
    Z = np.float32(Z)

    # define criteria, number of clusters(K) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img_color.shape))

    # Adjust the colors so we can see them
    print("Max Value", np.max(res2))
    max = np.max(res2)
    res2 = res2 * int(256 / max)
    print("Max Value", np.max(res2))

    cv2.imshow('res2', res2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return res2

def try_equalized_Histogram(in_image):
    out_image = cv2.equalizeHist(in_image)
    cv2.imshow('image', out_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def try_thresholding():
    #https://docs.opencv.org/3.4.0/d7/d4d/tutorial_py_thresholding.html
    return None


def try_blob_detection(im):
    # https://www.learnopencv.com/blob-detection-using-opencv-python-c/
    # https://stackoverflow.com/questions/8076889/how-to-use-opencv-simpleblobdetector
    # Setup SimpleBlobDetector parameters.
    params = cv2.SimpleBlobDetector_Params()

    # Change thresholds
    params.minThreshold = 10
    params.maxThreshold = 200

    # Filter by Area.
    params.filterByArea = True
    params.minArea = 1500

    # Filter by Circularity
    params.filterByCircularity = True
    params.minCircularity = 0.1

    # Filter by Convexity
    params.filterByConvexity = True
    params.minConvexity = 0.87

    # Filter by Inertia
    params.filterByInertia = True
    params.minInertiaRatio = 0.01

    # Create a detector with the parameters
    detector = cv2.SimpleBlobDetector_create(params)

    # Detect blobs.
    keypoints = detector.detect(im)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures
    # the size of the circle corresponds to the size of blob

    im_with_keypoints = cv2.drawKeypoints(im, keypoints, np.array([]), (0, 0, 255),
                                          cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # Show blobs
    cv2.imshow("Keypoints", im_with_keypoints)
    cv2.waitKey(0)

def dialation(img):
    # Maybe we can use dialation to exaggerate the regions?
    # https://docs.opencv.org/master/d9/d61/tutorial_py_morphological_ops.html
    kernel = np.ones((5, 5), np.uint8)
    d = cv2.dilate(img, kernel, iterations=5)

    cv2.imshow('image', d)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return d

def try_them_all():
    try_k_means(averaged_image_bgr, 8)
    k_image = try_k_means(averaged_image_bgr, 4)  # This looks like the best of the k_means options
    try_k_means(averaged_image_bgr, 2)
    try_cc_alone() # Almost?
    show_hist_gray(averaged_image)

    # Doesn't appear to find any blobs
    try_blob_detection(averaged_image_bgr)

    # Works sort of, but too many regions
    try_equalized_Histogram(averaged_image)

    new_image4 = try_manual_paint_image(averaged_image_bgr, averaged_image, 4)
    new_image3 = try_manual_paint_image(averaged_image_bgr, averaged_image, 3)
    new_image2 = try_manual_paint_image(averaged_image_bgr, averaged_image, 2)
    new_image1 = try_manual_paint_image(averaged_image_bgr, averaged_image, 1)
    new_image5 = try_manual_paint_image(new_image4, averaged_image, 2)

    d_image = dialation(k_image)


def create_banding_gray(image):
    img = np.asarray(list(image))

    if len(image.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # sample an Widght by N area
    for i in range(0, img.shape[0], 20):
        roi = img[i:i+20, :]
        #largest_occurrence = max(roi, key=roi.)
        #largest_occurrence = np.bincount(roi)
        unique, counts = np.unique(roi, return_counts=True)
        array_position = counts.argmax()
        largest_occurrence = unique[array_position]
        print(np.asarray((unique, counts)).T)
        img[i:i+20, :] = largest_occurrence


    cv2.imshow('Banded Grayscale image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img


def create_banding_color(image):
    img = np.asarray(list(image))

    # sample an Widght by N area
    for i in range(0, img.shape[0], 20):
        roi = img[i:i+20, :]

        color = []
        for c in range(3):
            hist = cv2.calcHist([roi], [c], None, [256], [0, 256])
            #unique, counts = np.unique(hist, return_counts=True)
            array_position = hist.argmax()
            # largest_occurrence = unique[array_position]
            color.append(array_position)

        combined_color = (color[0], color[1], color[2])

        img[i:i+20, :] = combined_color


    cv2.imshow('Banded Color image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    return img

def overlay_image(overlay, alpha, background):
    output = np.asarray(list(background))
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    cv2.imshow('Overlayed', output)
    cv2.waitKey(0)

    return output

# This looks like the best of the k_means options for our source image
k_image = try_k_means(averaged_image_bgr, 3)
banded_image = create_banding_color(k_image)

k_image_small = cv2.resize(k_image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
banded_image_small = cv2.resize(banded_image,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

sbs = np.hstack((k_image_small,banded_image_small)) #stacking images side-by-side
cv2.imshow('Side-by-side', sbs)
cv2.waitKey(0)

# Create an overlay and a greyscale version
overlayed = overlay_image(banded_image, 0.7, k_image)
banded_gray = create_banding_gray(k_image)

# Convert banded_gray to 3 channels so we can slap them together
banded_gray_3channel = np.stack((banded_gray,)*3, -1)

#Shrink those down
overlayed_small = cv2.resize(overlayed,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)
banded_gray_small = cv2.resize(banded_gray_3channel,None,fx=0.5, fy=0.5, interpolation = cv2.INTER_CUBIC)

# Display them
sbs2 = np.hstack((overlayed_small,banded_gray_small)) #stacking images side-by-side
cv2.imshow('Side-by-side 2', sbs2)
cv2.waitKey(0)

vertical = np.vstack((sbs,sbs2)) #stacking images side-by-side
cv2.imshow('Vertical (All 4)', vertical)
cv2.waitKey(0)


# Cleanup
cv2.destroyAllWindows()
