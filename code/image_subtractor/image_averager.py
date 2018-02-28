import cv2
import os

avg_width = 20
avg_height = 20
DEBUG = False
directory = "C:\\Users\\thisisme1\\Downloads\\Spartan - Cell-20180124T191933Z-001\\Spartan - Cleaned\\"
output_directorty = "C:\\Users\\thisisme1\\Downloads\\Spartan - Cell-20180124T191933Z-001\\Spartan - Averaged\\"


def average_all_images(image_directory):

    # Load images from folders in loop
    for filename in os.listdir(image_directory):
        combined_filename = os.path.join(image_directory, filename)
        output_filename = os.path.join(output_directorty, filename)

        image = cv2.imread(combined_filename)
        if image is None:
            print("ERROR: Invalid file, skipping:", combined_filename)
            continue

        print("Averaging image:", combined_filename) if DEBUG else None
        image = cv2.blur(src=image, ksize=(20, 20))

        print("Saving image:", output_filename) if DEBUG else None
        cv2.imwrite(output_filename, image)


# Get the result of the complete averaged series
average_all_images(directory)

# cleanup
cv2.destroyAllWindows()
