import cv2
import numpy as np
import os

avg_width = 20
avg_height = 20

directory = "C:\\Users\\thisisme1\\Downloads\\Spartan - Cell-20180124T191933Z-001\\Spartan - Cleaned\\"
output_directorty = "C:\\Users\\thisisme1\\Downloads\\Spartan - Cell-20180124T191933Z-001\\Spartan - Averaged\\"


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


def average_all_images(image_directory):
    image = None

    # Load images from folders in loop
    for filename in os.listdir(image_directory):
        combined_filename = os.path.join(image_directory, filename)
        output_filename = os.path.join(output_directorty, filename)

        image = cv2.imread(combined_filename)
        if image is None:
            print("ERROR: Invalid file, skipping:", combined_filename)
            continue

        print("Averaging image:", combined_filename)
        image = average_image(image, avg_width, avg_height)
        
        print("Saving image:", output_filename)
        cv2.imwrite(output_filename, image)


# Get the result of the complete averaged series
average_all_images(directory)

# cleanup
cv2.destroyAllWindows()
