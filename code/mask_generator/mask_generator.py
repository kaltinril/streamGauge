import cv2                  # Used to load, blur, and save the images
import numpy as np          # Used for image and array manipulation
import sys                  # Used to get the sys.argv options
import getopt               # Friendly command line options

# Global Defaults
DEBUG = False
DEFAULT_SOURCE_FILENAME = "../image_subtractor/all_combined2.png"
DEFAULT_OUTPUT_FILENAME = "mask.png"
DEFAULT_K_VALUE = 3
DEFAULT_BAND_SIZE = 20


# https://docs.opencv.org/3.0-beta/doc/py_tutorials/py_ml/py_kmeans/py_kmeans_opencv/py_kmeans_opencv.html
def try_k_means(img_color, k_value=DEFAULT_K_VALUE):
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
    img = np.asarray(list(image))

    if len(image.shape) > 2:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # sample an width by band_size area
    for i in range(0, img.shape[0], band_size):
        roi = img[i:i+band_size, :]
        unique, counts = np.unique(roi, return_counts=True)
        array_position = counts.argmax()
        largest_occurrence = unique[array_position]
        img[i:i+band_size, :] = largest_occurrence
        print(np.asarray((unique, counts)).T) if DEBUG else None

    return img


def create_banding_color(image, band_size=DEFAULT_BAND_SIZE):
    img = np.asarray(list(image))

    # sample an width by band_size area
    for i in range(0, img.shape[0], band_size):
        roi = img[i:i+band_size, :]

        color = []
        for c in range(3):
            hist = cv2.calcHist([roi], [c], None, [256], [0, 256])
            array_position = hist.argmax()
            color.append(array_position)

        combined_color = (color[0], color[1], color[2])
        img[i:i+band_size, :] = combined_color

    return img


def overlay_image(overlay, alpha, background):
    output = np.asarray(list(background))
    cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

    return output


def build_mask(k_value, output_filename, source_filename):
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
    banded_image = create_banding_color(k_image)
    final_mask = cv2.bitwise_and(banded_image, k_image)

    # Do the same thing for greyscale
    k_image_gray = cv2.cvtColor(k_image, cv2.COLOR_BGR2GRAY)
    banded_gray = create_banding_gray(k_image_gray)
    final_mask_gray = cv2.bitwise_and(banded_gray, k_image_gray)

    # Add the bottom 35 rows of pixels back as a copy of the existing bottom 35 pixels
    k_image = np.vstack((k_image, k_image[-35:, :]))
    banded_image = np.vstack((banded_image, banded_image[-35:, :]))
    final_mask = np.vstack((final_mask, final_mask[-35:, :]))
    final_mask_gray = np.vstack((final_mask_gray, final_mask_gray[-35:, :]))

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
    print(" -s, --save")
    print("    Save each blurred image to a sub-directory")
    print("    WARNING: This will take longer.")
    print(" -d, --debug")
    print("    Turn debug mode on")
    print("    WARNING: This will slow down the process")
    print("")
    print("Example: " + script_name + ' -o outputfile.png -i "C:\\images\\"')


def load_arguments(argv):
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

    final_mask = build_mask(k_value, output_filename, source_filename)

    print("")
    print("Successfully completed, look for file", output_filename)

    return final_mask


if __name__ == "__main__":
    main(sys.argv)
