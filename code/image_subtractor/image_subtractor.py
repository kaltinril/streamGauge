import cv2
import numpy as np
import os

avg_width = 20
avg_height = 20

image1_name = './images/images_63796657_20180119143035_IMAG0089-100-89.JPG'
image2_name = './images/images_63796888_20180119143134_IMAG0090-100-90.JPG'
#image2_name = './images/images_63798188_20180119143733_IMAG0096-100-96.JPG'
#image2_name = './images/images_63825977_20180119171133_IMAG0250-100-250.JPG'
img1 = None
img2 = None


# Verify images have the same size
# TODO: Maybe resize the larger image to the smaller image??


def valid_images(image1, image2):
    print("Y, X, Channels: " + str(image1.shape))

    if image1.shape != image2.shape:
        print("Unable to process these two images, they are of different dimensions")
        return False

    print("Images the same size, can continue.")
    return True


def extract_average_from_region(image, x, y, height, width):
    # Get the 20x20 area (for image 1)
    region = image[y:y+height, x:x+width]  # img1[y1:y2, x1:x2]
    
    # Get the average color for all pixels in the region1
    avg_color_per_row = np.average(region, axis=0)
    avg_color = np.average(avg_color_per_row, axis=0)
    return avg_color


# Create a blank image to write to
blank_image = None

# loop over all images in directory
#   Grab 2 images
#   Subtract them and add to main array
#   if only 1 image left, ignore for now
# divide the blank_image by the number of pairs
directory = './images'
pairs = 0
# Load images from folders in loop
for filename in os.listdir(directory):
    combined_filename = os.path.join(directory, filename)
    
    if img1 is None:
        img1 = cv2.imread(combined_filename)
        if img1 is None:
            print("ERROR: Invalid file, skipping:", combined_filename)
            continue
        print(pairs, "Working on new image1", combined_filename)
        continue

    if img2 is None:
        img2 = cv2.imread(combined_filename)
        if img2 is None:
            print("ERROR: Invalid file, skipping:", combined_filename)
            continue
        print(pairs, "Working on new image2", combined_filename)

    pairs += 1
    if not valid_images(img1, img2):
        continue

    # Create a blank image to write to
    if blank_image is None:
        blank_image = np.zeros((img1.shape[0], img1.shape[1], 3), np.uint64)

    # We can't get the average of the left 10, top 10, right 10, and bottom 10 pixels
    y_offset = int(avg_height / 2)
    x_offset = int(avg_width / 2)

    rows = img1.shape[0] - avg_height
    cols = img1.shape[1] - avg_width

    # Loop over the entire image, section by section
    for y in range(0, rows):
        for x in range(0, cols):
            avg_color1 = extract_average_from_region(img1, x, y, avg_height, avg_width)
            avg_color2 = extract_average_from_region(img2, x, y, avg_height, avg_width)

            # Subtract the two colors and that becomes the new color
            avg_subtracted = np.absolute(np.subtract(avg_color1, avg_color2))
            blank_image[y+y_offset, x+x_offset] = blank_image[y+y_offset, x+x_offset] + avg_subtracted

    # Since we are comparing adjacent images, we need img2 and img3
    # so, move img2 to img1, and force directory processing to grab img3 and toss it in img2.
    # TODO: Instead, calculate IMG1 average and then afterwards, save that instead of recalculating each loop
    # TODO: this would mean we do N averages instead of (N*2)-2
    img1 = img2
    img2 = None

    # if pairs == 4:
    #     break

# average the values
blank_image = blank_image / pairs

# convert back to uint8
blank_image = blank_image.astype(np.uint8)

cv2.imwrite('all_combined.png', blank_image)

# Display
# cv2.imshow('image', blank_image)
# cv2.waitKey(0)

# cleanup
cv2.destroyAllWindows()
