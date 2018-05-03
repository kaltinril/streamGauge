import cv2                  # Used to load, blur, and save the images
import numpy as np          # Used for image and array manipulation
import os                   # Used to combine the image paths, create the output directory of save_blur
import sys                  # Used to get the sys.argv options
import getopt               # Friendly command line options

'''
Image Subtractor
    Find the temporal differences between a set of images in a directory
    Created by:     Jeremy Swartwood
Usage:
    To use this in another python library:
        import cv2
        import image_subtractor
        
        all_combined = image_subtractor.average_then_subtract_images("./images/")
        cv2.imwrite("all_combined.png", all_combined)
'''

# Global Values and Defaults
DEBUG = False
AVERAGE_WIDTH_DEFAULT = 21
AVERAGE_HEIGHT_DEFAULT = 21
SAVE_BLUR_DEFAULT = False


def create_output_directory(image_directory):
    """
    Create a new folder/directory if it does not already exist

    :param image_directory:     Directory to put the output folder in
    :return:                    Return the image_directory combined with the output folder
    """
    output_directory = os.path.join(image_directory, "output")
    if not os.path.exists(output_directory):
        print("Making output directory to save the Blurred Images...")
        print("  output_directory: " + output_directory)
        os.makedirs(output_directory)

    return output_directory


def save_blurred_image(output_directory, filename, image):
    """
    Given an output directory and filename, save the passed in imager to that filename in that directory

    :param output_directory:    Where the output filename should be saved
    :param filename:            What the output filename should be
    :param image:               The output image to save
    """
    output_filename = os.path.join(output_directory, filename)

    print("Saving image:", output_filename) if DEBUG else None
    cv2.imwrite(output_filename, image)


def load_and_blur_image(input_image, pairs, average_width, average_height):
    """
    Load an image in, run the OpenCV BLUR method on that image to perform low pass filtering.

    :param input_image:     Image to blur
    :param pairs:           Debug print information
    :param average_width:   Size to blur together Horizontally
    :param average_height:  Size to blur together Vertically
    :return:                The blurred image
    """
    image = cv2.imread(input_image)
    if image is None:
        print("ERROR: Invalid file, skipping:", input_image)
        return None

    print(pairs, "Working on new image:", input_image) if DEBUG else None

    return cv2.blur(src=image, ksize=(average_width, average_height))


def resize_image_to_mask(image, mask):
    """
    If an image size does not match the mask, resize the image to match the mask.

    :param image:   Image to run through Gabor Filter activations
    :param mask:    The mask that is used to determine which bands each ROI belongs to
    :return:        The resized image
    """
    r = mask.shape[1] / image.shape[1]
    dim = (mask.shape[1], int(image.shape[0] * r))
    resized = cv2.resize(image, dim, interpolation=cv2.INTER_AREA)

    return resized


def average_then_subtract_images(image_directory,
                                 average_width=AVERAGE_WIDTH_DEFAULT,
                                 average_height=AVERAGE_HEIGHT_DEFAULT,
                                 save_blur=SAVE_BLUR_DEFAULT):
    """
    Find the temporal differences between a set of images in a directory
     Steps:    1. Go through all images in the image_directory 2 at a time.
               2. Blur both images average_width and average_height (reduce small differences in the images)
               3. Subtract each pair of images color values to produce the absolute value difference.
               4. Add all these pair values together and divide by the total number of pairs (Average)
               5. Return an image of the final result
     Link:
               https://docs.opencv.org/master/d4/d13/tutorial_py_filtering.html

    :param image_directory:     Source directory that contains the images to be blurred, subtracted, and averaged.
    :param average_width:       Width of blur region
    :param average_height:      Height of blur region
    :param save_blur:           Boolean: Should the image be saved?
    :return:                    Combined image output of all image pairs blurred, subtracted, and averaged.
    """
    out_image = None
    image1 = None
    image2 = None
    output_directory = ""
    pairs = 0

    # If the user wants to save the blurred images, make sure the output directory exists
    if save_blur:
        output_directory = create_output_directory(image_directory)

    # Load images from folders in loop
    for filename in os.listdir(image_directory):
        combined_filename = os.path.join(image_directory, filename)

        # Load first image if we have not yet loaded it
        if image1 is None:
            image1 = load_and_blur_image(combined_filename, pairs, average_width, average_height)

            # Save the blurred image1 (This should only happen once)
            if save_blur and image1 is not None:
                save_blurred_image(output_directory, filename, image1)
            continue

        # Load second image
        if image2 is None:
            image2 = load_and_blur_image(combined_filename, pairs, average_width, average_height)
            if image2 is None:
                continue

        # Make sure the images are the correct dimensions, if not, resize to mask size
        if image1.shape[0:2] != image2.shape[0:2]:
            print('Image size mismatch, resizing:', combined_filename)
            image2 = resize_image_to_mask(image2, image1)

        # Save the blurred image2
        if save_blur:
            save_blurred_image(output_directory, filename, image2)

        # Create a blank image to write to
        # We don't have the dimensions until we've entered the loop, so this will only be None once.
        if out_image is None:
            out_image = np.zeros((image1.shape[0], image1.shape[1], 3), np.uint64)

        pairs += 1  # Keep track of how many pairs so we can divide by this later

        # (Subtract) the color values of the 2 images and add the result to the output_image color values
        avg_subtracted = cv2.absdiff(image1, image2)
        out_image = np.add(out_image, avg_subtracted)

        # Set image1 as the blurred (Averaged) image2 so we don't have to re-calculate it
        # Clear image2 so we load the next image on the next loop
        image1 = image2
        image2 = None

    # (Average) the output_image and convert back to 0-255 image size
    out_image = out_image / pairs  # average the values
    out_image = out_image.astype(np.uint8)  # convert back to uint8
    return out_image


def print_help(script_name):
    """
    Print out command line usage and arguments

    :param script_name: Name of the script, used just to make the help print more specific to this file
    """
    print("Usage:   " + script_name + " -o <output_image> -i <input_directory> -w <width> -e <height> -s <save>")
    print("")
    print(" -h, --help")
    print("    This message is printed only")
    print(" -o, --outfile")
    print("    Output image from the subtracted and averaged indir images")
    print("    default: all_combined.png")
    print(" -i, --indir")
    print("    Input directory containing files to pair-wise subtract and average")
    print(" -w, --width")
    print("    Width of the region to blur")
    print("    default: 21")
    print(" -e, --height")
    print("    Height of the region to blur")
    print("    default: 21")
    print(" -s, --save")
    print("    Save each blurred image to a sub-directory")
    print("    WARNING: This will take longer.")
    print(" -d, --debug")
    print("    Turn debug mode on")
    print("    WARNING: This will slow down the process")
    print("")
    print("Example: " + script_name + ' -o outputfile.png -i "C:\\images\\"')


def load_arguments(argv):
    """
    Load all arguments that were passed in on the command line, and set parameters used in other locations
    
    :param argv:    The arguments from the command l ine
    :return:        The set values or defaults for: input_directory, output_filename, average_width, average_height, save_blur
    """
    global DEBUG
    script_name = argv[0]  # Snag the first argument (The script name)

    # Default values for parameters/arguments
    input_directory = "C:\\Users\\thisisme1\\Downloads\\Spartan - Cell-20180124T191933Z-001\\Spartan - Cleaned\\"
    output_filename = 'all_combined.png'
    average_width = AVERAGE_WIDTH_DEFAULT
    average_height = AVERAGE_HEIGHT_DEFAULT
    save_blur = SAVE_BLUR_DEFAULT

    # No reason to parse the options if there are none, just use the defaults
    if len(argv) > 1:
        try:
            single_character_options = "ho:i:w:e:ds"  # : indicates a required value with the value
            full_word_options = ["help", "outfile=", "indir=", "width=", "height=", "debug", "save"]

            opts, remainder = getopt.getopt(argv[1:], single_character_options, full_word_options)
        except getopt.GetoptError:
            print("ERROR: Unable to get command line arguments!")
            print_help(script_name)
            sys.exit(2)

        for opt, arg in opts:
            if opt in ("-h", "--help"):
                print_help(script_name)
                sys.exit(0)
            elif opt in ("-o", "--outfile"):
                output_filename = arg
            elif opt in ("-i", "--indir"):
                input_directory = arg
            elif opt in ("-w", "--width"):
                average_width = arg
            elif opt in ("-e", "--height"):
                average_height = int(arg)
            elif opt in ("-d", "--debug"):
                DEBUG = True  # Global variable for printing out debug information
            elif opt in ("-s", "--save"):
                save_blur = True

    print("Using parameters:")
    print("Output File:     ", output_filename)
    print("Intput Directory:", input_directory)
    print("Blur Width:      ", average_width)
    print("Blur Height:     ", average_height)
    print("Save blur images?", save_blur)
    print("Debug:           ", DEBUG)
    print("")

    return input_directory, output_filename, average_width, average_height, save_blur


def main(argv):
    print("Image Temporal Blurring, Averaging, and Subtraction.")
    print("")

    # Load all the arguments and return them
    input_directory, output_filename, average_width, average_height, save_blur = load_arguments(argv)

    # Get the result of the complete averaged series of files and save it
    blank_image = average_then_subtract_images(input_directory, average_width, average_height, save_blur)
    cv2.imwrite(output_filename, blank_image)

    # Display image if debug is on
    if DEBUG:
        cv2.imshow('All Combined Image', blank_image)
        cv2.waitKey(0)

    # cleanup
    cv2.destroyAllWindows()

    print("")
    print("Successfully completed, look for file", output_filename)


if __name__ == "__main__":
    main(sys.argv)
