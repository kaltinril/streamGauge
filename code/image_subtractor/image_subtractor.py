import cv2
import numpy as np
import os

avg_width = 21
avg_height = 21
DEBUG = False

image1_name = './images/images_63796657_20180119143035_IMAG0089-100-89.JPG'
image2_name = './images/images_63796888_20180119143134_IMAG0090-100-90.JPG'
directory = "C:\\Users\\thisisme1\\Downloads\\Spartan - Cell-20180124T191933Z-001\\Spartan - Cleaned\\"

# TODO: Need to look into using this, maybe do the same thing but faster:
# https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html

# Verify images have the same size
# TODO: Maybe resize the larger image to the smaller image??


def valid_images(image1, image2):
    if image1.shape != image2.shape:
        print("Unable to process these two images, they are of different dimensions")
        print("(Y, X, Channels): ", str(image1.shape), str(image2.shape))
        return False

    return True


def average_then_subtract_images(image_directory):
    out_image = None
    image1 = None
    image2 = None
    pairs = 0

    # Load images from folders in loop
    for filename in os.listdir(image_directory):
        combined_filename = os.path.join(image_directory, filename)

        if image1 is None:
            image1 = cv2.imread(combined_filename)
            if image1 is None:
                print("ERROR: Invalid file, skipping:", combined_filename)
                continue

            print(pairs, "Working on new image1", combined_filename) if DEBUG else None
            image1 = cv2.blur(src=image1, ksize=(avg_width, avg_height))
            continue

        if image2 is None:
            image2 = cv2.imread(combined_filename)
            if image2 is None:
                print("ERROR: Invalid file, skipping:", combined_filename)
                continue

            print(pairs, "Working on new image2", combined_filename) if DEBUG else None
            image2 = cv2.blur(src=image2, ksize=(avg_width, avg_height))

        if not valid_images(image1, image2):
            image2 = None
            continue

        # Create a blank image to write to
        if out_image is None:
            out_image = np.zeros((image1.shape[0], image1.shape[1], 3), np.uint64)

        pairs += 1

        avg_subtracted = cv2.absdiff(image1, image2)
        out_image = np.add(out_image, avg_subtracted)

        # Set image1 as the blurred (Averaged) image2 so we don't have to re-calculate it
        # when we compare against "image3"
        image1 = image2
        image2 = None

    out_image = out_image / pairs  # average the values
    out_image = out_image.astype(np.uint8)  # convert back to uint8
    return out_image


# Get the result of the complete averaged series
blank_image = average_then_subtract_images(directory)

cv2.imwrite('all_combined2.png', blank_image)

# Display
# cv2.imshow('image', blank_image)
# cv2.waitKey(0)

# cleanup
cv2.destroyAllWindows()
