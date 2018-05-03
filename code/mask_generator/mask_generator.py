import cv2                  # Used to load, blur, and save the images
import numpy as np          # Used for image and array manipulation
import sys                  # Used to get the sys.argv options
import getopt               # Friendly command line options

'''
Mark Generator
    Take a temporal subtracted aggregate image in and produce a mask image from it
    Created by:     Jeremy Swartwood
Usage:
    To use this in another python library:
        import cv2
        import mask_generator
        
        mask = mask_generator.build_mask("temporal_image.png")
        cv2.imwrite("mask.png", mask)
'''

# Global Defaults
DEBUG = False
DEFAULT_SOURCE_FILENAME = "../image_subtractor/all_combined.png"
DEFAULT_OUTPUT_FILENAME = "mask.png"
DEFAULT_K_VALUE = 3
DEFAULT_BAND_SIZE = 20


# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
def try_k_means(img_color, k_value=DEFAULT_K_VALUE):
    """
    Take an input image, and run the OpenCV kmeans implementation against it to reduce the image into K_VALUE colors.

    :param img_color:   The color image to run the kmeans on
    :param k_value:     The number of centroids/categories
    :return:            The Clustered result image
    """
    reshaped_image = img_color.reshape((-1, 3))

    # convert to np.float32
    reshaped_image = np.float32(reshaped_image)

    # define criteria, number of clusters(k_value) and apply kmeans()
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(reshaped_image, k_value, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # Now convert back into uint8, and make original image
    center = np.uint8(center)
    res = center[label.flatten()]
    res2 = res.reshape((img_color.shape))

    # Adjust the colors so we can see them
    max_value = np.max(res2)
    res2 = res2 * int(256 / max_value)
    print("Max Value", np.max(res2)) if DEBUG else None

    return res2


def create_banding_gray(image, band_size=DEFAULT_BAND_SIZE):
    """
    Take an input mask image and create a image with bands of the max value for that row in grayscale

    :param image:       The source image to run the banding on
    :param band_size:   The size to check horizontally for similar color values
    :return:
                        img = The banded image values 20, 40, 80, etc
                        img2 = The banded image values, but using the original color values from kmeans
    """
    img = np.asarray(list(image))
    img2 = np.asarray(list(image))

    if len(image.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # sample an width by band_size area
    old_color = 0
    band_color = 10
    for i in range(0, img.shape[0], band_size):
        roi = img[i:i+band_size, :]
        unique, counts = np.unique(roi, return_counts=True)
        array_position = counts.argmax()
        largest_occurrence = unique[array_position]

        if old_color != largest_occurrence:
            old_color = largest_occurrence
            band_color *= 2

        img[i:i + band_size, :] = band_color
        img2[i:i + band_size, :] = largest_occurrence
        print(np.asarray((unique, counts)).T) if DEBUG else None

    return img, img2


def create_banding_color(image, band_size=DEFAULT_BAND_SIZE):
    """
    Take an input mask image and create a image with bands of the max value for that row in Color

    :param image:       The source image to run the banding on
    :param band_size:   The size to check horizontally for similar color values
    :return:
                        img = The banded image values 20, 40, 80, etc
                        img2 = The banded image values, but using the original color values from kmeans
    """
    img = np.asarray(list(image))
    img2 = np.asarray(list(image))

    # sample an width by band_size area
    color_old = []
    band_color = np.asarray([10, 10, 10])

    for i in range(0, img.shape[0], band_size):
        roi = img[i:i+band_size, :]

        color = []
        for c in range(3):
            hist = cv2.calcHist([roi], [c], None, [256], [0, 256])
            array_position = hist.argmax()
            color.append(array_position)

        combined_color = (color[0], color[1], color[2])

        if color_old != combined_color:
            color_old = combined_color
            band_color *= 2

        img[i:i+band_size, :] = band_color
        img2[i:i+band_size, :] = combined_color

    print(band_color)
    return img, img2


def overlay_image(overlay, alpha, background):
    """
    Take two images and put 1 "ontop" of the other by making the top one slightly transparent

    :param overlay:         Image to put ontop
    :param alpha:           Transparency value
    :param background:      Image to put behind
    :return:                Combined image
    """
    output = np.asarray(list(background))
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output


def convert_banded_to_unique_colors(band_img, band_img_orig, mask):
    """
    Take the banded image, the banded with original color values, and the original kmeans mask
        Produce a final image with the banded colors, except when there is overlap between bands
        in which case make the color BLACK so we can exclude it during the Gabor filter generation.

    :param band_img:        Banded image with values like 20, 40, 80
    :param band_img_orig:   Banded image with values like 0, 1, 2, 3
    :param mask:            Kmeans mask with values like 0, 1, 2, 3
    :return:                Banded mask with cut out BLACK area's that are overlapping from the other bands
    """
    img = np.asarray(list(band_img))

    if len(band_img.shape) == 3:
        black = np.zeros(band_img.shape[2])
    else:
        black = 0

    for y in range(band_img.shape[0]):
        for x in range(band_img.shape[1]):

            # If the image colors match, use the band_img color,
            # otherwise, use BLACK.
            if np.array_equal(band_img_orig[y, x], mask[y, x]):
                color = band_img[y, x]
            else:
                color = black

            img[y, x] = color

    return img


def build_mask(source_filename=DEFAULT_SOURCE_FILENAME, 
               k_value=DEFAULT_K_VALUE,
               output_filename=DEFAULT_OUTPUT_FILENAME):
    """
    1. Take in the source all_averaged.png file
    2. run kmeans with the K-value clusters
    3. produce the bands based on the highest k-value in that area
    4. Create a bitwise_and image to remove contradicting sections
    5. Save the final mask

    :param source_filename: Image to use as the input to the kmeans mask generation
    :param k_value:         Number of clusters the kmeans should use
    :param output_filename: The output final mask
    :return:                The output final mask
    """
    # Load image in
    source_image_gray = cv2.imread(source_filename, cv2.IMREAD_GRAYSCALE)
    source_image = cv2.imread(source_filename)

    # Display debug information if requested
    if DEBUG:
        print("Starting temporal image shape", source_image_gray.shape)
        print("Max BGR Value:", np.max(source_image))
        print("Max Gray scale Value:", np.max(source_image_gray))

    # Remove bottom boarder
    # TODO: Find dynamic way to do this
    source_image_gray = source_image_gray[0:-35, :]
    source_image = source_image[0:-35, :]

    # Display debug information if requested
    if DEBUG:
        print("After Removing bottom shape", source_image_gray.shape)
        print("Max BGR Value:", np.max(source_image))
        print("Max Gray scale Value:", np.max(source_image_gray))

    # Create the K_IMAGE and then turn it into a banded image, and finally bitwise "AND" the images together
    k_image = try_k_means(source_image, k_value)
    banded_image, banded_image2 = create_banding_color(k_image)
    final_mask = cv2.bitwise_and(banded_image2, k_image)
    final_mask = convert_banded_to_unique_colors(banded_image, banded_image2, final_mask)


    # Do the same thing for greyscale
    k_image_gray = cv2.cvtColor(k_image, cv2.COLOR_BGR2GRAY)
    banded_gray, banded_gray2 = create_banding_gray(k_image_gray)
    final_mask_gray = cv2.bitwise_and(banded_gray2, k_image_gray)
    final_mask_gray = convert_banded_to_unique_colors(banded_gray, banded_gray2, final_mask_gray)

    # Add the bottom 35 rows of pixels back as a black bar of values of "0"
    black_bar_color = np.zeros((35, k_image.shape[1], 3))
    black_bar_gray = np.zeros((35, k_image.shape[1]))
    k_image = np.vstack((k_image, black_bar_color))
    banded_image = np.vstack((banded_image, black_bar_color))
    final_mask = np.vstack((final_mask, black_bar_color))
    final_mask_gray = np.vstack((final_mask_gray, black_bar_gray))

    # Save the images
    cv2.imwrite("kmean-" + output_filename, k_image)
    cv2.imwrite("banding-" + output_filename, banded_image)
    cv2.imwrite(output_filename, final_mask)
    cv2.imwrite("gray-" + output_filename, final_mask_gray)

    # Display image if debug is on and Save extra analysis images
    if DEBUG:
        extra_debug_image_analysis(banded_image, k_image, output_filename)

    return final_mask


def extra_debug_image_analysis(banded_image, k_image, output_filename):
    """
    Create an image that can be used to compare and contrast how the bands will look in the final mask

    :param banded_image:    Banded mask image without kimage combination or bitwise_and
    :param k_image:         KMeans mask
    :param output_filename: Output file to save the comparison to
    """
    print("DEBUG: Creating side by side quad image")

    # Shrink k_image and banded_image down so we can fit them side by side easier
    k_image_small = cv2.resize(k_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    banded_image_small = cv2.resize(banded_image, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    # Stick the shrunk images side by side
    sbs = np.hstack((k_image_small, banded_image_small))  # stacking images side-by-side

    # Create an overlay and a grey-scale version
    overlayed = overlay_image(banded_image, 0.7, k_image)
    banded_gray = create_banding_gray(k_image)

    # Convert banded_gray to 3 channels so we can slap them together
    banded_gray_3channel = np.stack((banded_gray,) * 3, -1)

    # Shrink those down
    overlayed_small = cv2.resize(overlayed, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)
    banded_gray_small = cv2.resize(banded_gray_3channel, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_CUBIC)

    # Combine overlayed and banded to side by side
    sbs2 = np.hstack((overlayed_small, banded_gray_small))  # stacking images side-by-side

    # Combine both side by side horizontal stacks into a 2x2 quad
    all_four = np.vstack((sbs, sbs2))  # stacking images side-by-side
    cv2.imwrite('all_four_' + output_filename, all_four)


def print_help(script_name):
    """
    Print out command line usage and arguments

    :param script_name: Name of the script, used just to make the help print more specific to this file
    """
    print("Usage:   " + script_name + " -h -o <output_mask_filename> -i <input_all_subtracted_filename> -k <kvalue> -b <bands>")
    print("")
    print(" -h, --help")
    print("    This message is printed only")
    print(" -o, --outfile")
    print("    Output mask filename")
    print("    default: mask.png")
    print(" -i, --infile")
    print("    Input all averaged image generated from image_subtractor.py")
    print("    default: ../image_subtractor/all_combined.png")
    print(" -k, --kvalue")
    print("    The number of 'bands' to target with the k-means algorithm")
    print("    default: 3")
    print(" -b, --banding")
    print("    The height of the horizontal area to make all the same color")
    print("    default: 20")
    print(" -d, --debug")
    print("    Turn debug mode on")
    print("    WARNING: This will slow down the process")
    print("")
    print("Example: " + script_name + ' -o mask.png -i "C:\\images\\"')


def load_arguments(argv):
    """
    Load all arguments that were passed in on the command line, and set parameters used in other locations

    :param argv:    The arguments from the command line
    :return:        The set values or defaults for: source_filename, output_filename, k_value
    """
    global DEBUG
    script_name = argv[0]  # Snag the first argument (The script name)

    # Default values for parameters/arguments
    source_filename = DEFAULT_SOURCE_FILENAME
    output_filename = DEFAULT_OUTPUT_FILENAME
    k_value = DEFAULT_K_VALUE
    band_size = DEFAULT_BAND_SIZE

    # No reason to parse the options if there are none, just use the defaults
    if len(argv) > 1:
        try:
            single_character_options = "ho:i:k:b:d"  # : indicates a required value with the value
            full_word_options = ["help", "outfile=", "infile=", "kvalue=", "banding=", "debug"]

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
            elif opt in ("-i", "--infile"):
                source_filename = arg
            elif opt in ("-k", "--kvalue"):
                k_value = arg
            elif opt in ("-b", "--banding"):
                band_size = arg
            elif opt in ("-d", "--debug"):
                DEBUG = True  # Global variable for printing out debug information

    print("Using parameters:")
    print("Output File:     ", output_filename)
    print("Input File:      ", source_filename)
    print("K-Value:         ", k_value)
    print("Banding Size:    ", band_size)
    print("Debug:           ", DEBUG)
    print("")

    return source_filename, output_filename, k_value


def main(argv):
    print("Mask generation from temporally subtracted image")
    print("")

    # Load all the arguments and return them
    source_filename, output_filename, k_value = load_arguments(argv)

    final_mask = build_mask(source_filename, k_value, output_filename)

    print("")
    print("Successfully completed, look for file", output_filename)

    return final_mask


if __name__ == "__main__":
    main(sys.argv)
