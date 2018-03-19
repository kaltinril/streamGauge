import cv2
import numpy as np
import os
import sys

'''
Overlay the mask image with all the images in a folder to produce masked version of the images
'''

DEBUG = False

def overlay_image(background, overlay, alpha):
    output = np.asarray(list(background))
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output


# Makes sure the file extension is an image
# that the image loaded
# that the image shape matches the overlay shape
# that the filename doesn't contain "_mask".
def valid_images(name, ext, image, image_overlay):

    # Move tot he next file if it is not an image
    if ext.lower() not in ['.png', '.jpg', '.gif', '.bmp', '.tif', '.tiff'] or \
            image is None or \
            image_overlay.shape != image.shape or \
            name[-5:] == '_mask':
        return False

    return True


def overlay_all(image_folder, output_folder, overlay_filename, alpha):

    print("Loading mask:", overlay_filename)
    image_overlay = cv2.imread(overlay_filename)
    if image_overlay is None:
        print("ERROR: Overlay image is invalid.", file=sys.stderr)
        sys.exit(1)

    # Load list of files, so we don't accidentally find the mask files (Happened during testing)
    files = os.listdir(image_folder)

    for filename in files:
        combined_filename = os.path.join(image_folder, filename)

        print("Working on:", filename) if DEBUG else None

        # Split the filename into name and extension so we can add the word "Mask" to the output filename
        name, ext = os.path.splitext(filename)
        output_filename = os.path.join(output_folder, name + '_mask' + ext)

        # Load the image, skip to the next image if it wasn't loaded
        image = cv2.imread(combined_filename)

        # If the the image or extension have an issue, skip the file
        if not valid_images(name, ext, image, image_overlay):
            print("Skipping:", filename) if DEBUG else None
            continue

        # Produce the overlaid version
        output = overlay_image(image, image_overlay, alpha)

        # Save the image to a new file
        cv2.imwrite(output_filename, output)


def main():
    image_folder = '../image_subtractor/images/'
    output_folder = './overlaid/'
    overlay = '../mask_generator/mask.png'
    alpha = 0.6

    overlay_all(image_folder, output_folder, overlay, alpha)


if __name__ == "__main__":
    main()
