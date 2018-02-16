import cv2
import numpy as np
import os

avg_width = 20
avg_height = 20

image1_name = './images/images_63796657_20180119143035_IMAG0089-100-89.JPG'
image2_name = './images/images_63796888_20180119143134_IMAG0090-100-90.JPG'
#image2_name = './images/images_63798188_20180119143733_IMAG0096-100-96.JPG'
#image2_name = './images/images_63825977_20180119171133_IMAG0250-100-250.JPG'
directory = "C:\\Users\\thisisme1\\Downloads\\Spartan - Cell-20180124T191933Z-001\\Spartan - Cleaned\\"


# Verify images have the same size
# TODO: Maybe resize the larger image to the smaller image??


def valid_images(image1, image2):
    if image1.shape != image2.shape:
        print("Unable to process these two images, they are of different dimensions")
        print("(Y, X, Channels): ", str(image1.shape), str(image2.shape))
        return False

    return True


def extract_average_from_region(image, x, y, height, width):
    # Get the 20x20 area (for image 1)
    region = image[y:y+height, x:x+width]  # img1[y1:y2, x1:x2]
    
    # Get the average color for all pixels in the region1
    avg_color_per_row = np.average(region, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color


def average_image(source_image, roi_width, roi_height):
    out_image = np.zeros((source_image.shape[0], source_image.shape[1], 3), np.uint8)

    # We can't get the average of the left 10, top 10, right 10, and bottom 10 pixels
    y_offset = int(roi_height / 2)
    x_offset = int(roi_width / 2)

    rows = source_image.shape[0] - roi_height
    cols = source_image.shape[1] - roi_width

    for y in range(0, rows):
        for x in range(0, cols):
            average_color = extract_average_from_region(source_image, x, y, roi_height, roi_width)
            out_image[y + y_offset, x + x_offset] = average_color

    return out_image


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

            print(pairs, "Working on new image1", combined_filename)
            image1 = average_image(image1, avg_width, avg_height)
            continue

        if image2 is None:
            image2 = cv2.imread(combined_filename)
            if image2 is None:
                print("ERROR: Invalid file, skipping:", combined_filename)
                continue

            print(pairs, "Working on new image2", combined_filename)
            image2 = average_image(image2, avg_width, avg_height)

        if not valid_images(image1, image2):
            image2 = None
            continue

        # Create a blank image to write to
        if out_image is None:
            out_image = np.zeros((image1.shape[0], image1.shape[1], 3), np.uint64)

        pairs += 1

        #avg_subtracted = np.zeros((image1.shape[0], image1.shape[1], 3), np.int16)

        #cv2.subtract(image1, image2, dst=avg_subtracted, dtype=3)
        avg_subtracted = cv2.absdiff(image1, image2)
        #avg_subtracted = image1 - image2
        # print(image1[10,10])
        # print(image2[10,10])
        # print(avg_subtracted[10,10])

        out_image = np.add(out_image, avg_subtracted)



        # cv2.imshow('image', image1)
        # cv2.waitKey(0)
        # cv2.imshow('image', image2)
        # cv2.waitKey(0)
        # cv2.imshow('image', avg_subtracted)
        # cv2.waitKey(0)
        # cv2.imshow('image', out_image)
        # cv2.waitKey(0)

        # Set image1 as the blurred (Averaged) image2 so we don't have to re-calculate it
        # when we compare against "image3"
        image1 = image2
        image2 = None

        if pairs == 1:
            break

    out_image = out_image / pairs  # average the values
    out_image = out_image.astype(np.uint8)  # convert back to uint8
    return out_image


# Get the result of the complete averaged series
blank_image = average_then_subtract_images(directory)

cv2.imwrite('all_combined.png', blank_image)

# Display
# cv2.imshow('image', blank_image)
# cv2.waitKey(0)

# cleanup
cv2.destroyAllWindows()
