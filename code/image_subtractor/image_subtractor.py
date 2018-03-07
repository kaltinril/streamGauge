import cv2                  # Used to load, blur, and save the images
import numpy as np          # Used for image and array manipulation
import os                   # Used to combine the image paths
import sys                  # Used to get the sys.argv options
import getopt               # Friendly command line options

'''
Image Subtractor
    Find the temporal differences between a set of images in a directory
    Created by:     Jeremy Swartwood
'''

# Global Values
DEBUG = False


# Method:   valid_image
# Purpose:  Verify images have the same size
# Input:
#           image1 - First image to check file_size, used as the assumed CORRECT file_size
#           image2 - Second image to check the file_size
#
# Output:
#           Boolean (True/False) if the images are valid (The same size)
# TODO: Maybe resize the larger image to the smaller image, instead of tossing it??
# TODO: Maybe ignore images that are "TOO different"
def valid_images(image1, image2, image2_name):
    if image1.shape != image2.shape:
        print("Mismatching Dimensions, skipping: [" + image2_name + "].")
        print("(Y, X, Channels) = Image1:", str(image1.shape), " Image2:", str(image2.shape))
        return False

    return True


# Method:   average_then_subtract_images
# Purpose:  Find the temporal differences between a set of images in a directory
# Input:
#           image_directory - Source directory that contains the images to be blurred, subtracted, and averaged.
#           average_width - Width of blur region
#           average_height - Height of blur region
#
# Output:
#           Combined image output of all image pairs blurred, subtracted, and averaged.
#
# Steps:    1. Go through all images in the image_directory 2 at a time.
#           2. Blur both images average_width and average_height (reduce small differences in the images)
#           3. Subtract each pair of images color values to produce the absolute value difference.
#           4. Add all these pair values together and divide by the total number of pairs (Average)
#           5. Return an image of the final result
def average_then_subtract_images(image_directory, average_width, average_height):
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
            image1 = cv2.blur(src=image1, ksize=(average_width, average_height))
            continue

        if image2 is None:
            image2 = cv2.imread(combined_filename)
            if image2 is None:
                print("ERROR: Invalid file, skipping:", combined_filename)
                continue

            print(pairs, "Working on new image2", combined_filename) if DEBUG else None
            image2 = cv2.blur(src=image2, ksize=(average_width, average_height))

        if not valid_images(image1, image2, combined_filename):
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


def print_help(script_name):
    print("Usage:   " + script_name + " -f <filename> -a <serverAddress> -p <port> -e <error%>")
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
    print(" -d, --debug")
    print("    Turn debug mode on")
    print("Example: " + script_name + ' -o outputfile.png -i "C:\\images\\"')


def load_arguments(argv):
    global DEBUG
    script_name = argv[0]  # Snag the first argument (The script name)

    # Default values for parameters/arguments
    input_directory = "C:\\Users\\thisisme1\\Downloads\\Spartan - Cell-20180124T191933Z-001\\Spartan - Cleaned\\"
    output_filename = 'all_combined.png'
    average_width = 21
    average_height = 21

    # No reason to parse the options if there are none, just use the defaults
    if len(argv) > 1:
        try:
            single_character_options = "ho:i:w:e:d"  # : indicates a required value with the value
            full_word_options = ["help", "outfile=", "indir=", "width=", "height=", "debug"]

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

    print("Using parameters:")
    print("Output File:     ", output_filename)
    print("Intput Directory:", input_directory)
    print("Blur Width:      ", average_width)
    print("Blur Height:     ", average_height)
    print("Debug:           ", DEBUG)
    print("")

    return input_directory, output_filename, average_width, average_height


def main(argv):
    print("Image Temporal Blurring, Averaging, and Subtraction.")
    print("")

    # Load all the arguments and return them
    input_directory, output_filename, average_width, average_height = load_arguments(argv)

    # Get the result of the complete averaged series of files and save it
    blank_image = average_then_subtract_images(input_directory, average_width, average_height)
    cv2.imwrite(output_filename, blank_image)

    # Display image if debug is on
    if DEBUG:
        cv2.imshow('All Combined Image', blank_image)
        cv2.waitKey(0)

    # cleanup
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
