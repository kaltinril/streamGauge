import numpy as np
import cv2
import matplotlib.pyplot as plt  # Only used for debug imaging
import sys
import time

DEBUG = False


# Take data, plot it, and return a numpy array of the plotted graph image
def data2np(graph_data, line_format='-'):
    # Generate a figure with matplotlib
    fig = plt.figure()
    plot = fig.add_subplot(111)

    # Resize
    fig.set_size_inches(7, 5)

    # Plot the data and draw the canvas
    plot.plot(graph_data, line_format)
    fig.tight_layout(pad=0)  # Reduce the white border
    fig.canvas.draw()  # It must be drawn before we can extract it

    # Now we can save it to a numpy array.
    data = np.fromstring(fig.canvas.tostring_rgb(), dtype=np.uint8, sep='')
    data = data.reshape(fig.canvas.get_width_height()[::-1] + (3,))

    # Cleanup
    plt.clf()
    plt.close(fig)

    return data


def build_filters(orientations=16, ksize=31):
    filters = []
    for theta in np.arange(0, np.pi, np.pi / orientations):
        kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        kern /= 1.5*kern.sum()
        filters.append(kern)
    return filters


def process(img, filters):
    kernal_results = np.empty((len(filters), img.shape[0], img.shape[1]), dtype=np.uint8)

    combined = np.zeros_like(img)
    kern_index = 0
    for kern in filters:
        fimg = cv2.filter2D(img, cv2.CV_8UC3, kern)
        kernal_results[kern_index, :, :] = fimg  # Put the image into an array we can use it outside
        kern_index += 1

        np.maximum(combined, fimg, combined) if DEBUG else None
    return combined, kernal_results


def run_gabor(color_image, filters, mask):
    orientations = 16

    # Convert to grayscale so we don't get multiple "votes" per pixel per kernal/orientation
    img = cv2.cvtColor(color_image, cv2.COLOR_BGR2GRAY)

    # run each of the N Gabor kernals (directions) once across the entire image producing N images (one for each direction/kernal)
    combined_image, kernal_results = process(img, filters)

    # From those, at each pixel position, take the maximum response at that position indicating the "direction" of that pixel. 0, 1, 2, .... , N
    # produce an array with these direction values for each pixel
    pixel_directions = np.argmax(kernal_results, axis=0)
    pixel_directions = pixel_directions.astype(np.uint8)

    print(pixel_directions.shape)

    output_directory = './HOG_GABOR/'
    image_filename = 'baboon'
    # output_file = open(output_directory + image_filename + "_hog.csv", 'w')

    # grab ROI from the produced array (say 45x45, or whatever)
    roi_size_y = 31
    roi_size_x = 31
    roi_size = (roi_size_y, roi_size_x)
    for y in range(0, pixel_directions.shape[0], roi_size[0]):
        for x in range(0, pixel_directions.shape[1], roi_size[1]):

            # Get the masks Region of Interest
            roi_mask = mask[y:y + roi_size[0], x:x + roi_size[1]]

            # Check if this region is completely inside a band
            if np.min(roi_mask) == np.max(roi_mask):
                # calculate/Count up the total of each value in image as a histogram of directions in the image
                roi = pixel_directions[y:y + roi_size[0], x:x + roi_size[1]]
                unique, unique_counts = np.unique(roi, return_counts=True)

                # A region may not always have values for ALL bins, so create an array and place what we get in it
                bins = np.zeros((orientations,))
                bins[unique] = unique_counts

                # store those ROI Histograms in a file

                # Display the Region during debug mode for examples
                if (np.argmax(bins) > 0) and DEBUG is True:
                    display_histogram(bins, color_image, combined_image, img, roi, roi_size, x, y)

            else:
                print("Tossing region", y, x) if DEBUG else None


    # run the data from the files through ANN


def display_histogram(bins, color_image, combined_image, img, roi, roi_size, x, y):
    # Plot and draw the bins histogram, then convert to numpy array
    result = data2np(bins)

    # Get the ROI for the items we are going to show the user
    gray_roi = img[y:y + roi_size[0], x:x + roi_size[1]]  # Get the greyscale ROI
    max_roi = combined_image[y:y + roi_size[0], x:x + roi_size[1]]  # Get max value per pixel in region for all Kernals
    roi_color = color_image[y:y + roi_size[0], x:x + roi_size[1]]

    # Exaggerate the ROI orientation pixel direction values for viewing
    sample = (roi + 1) * 15

    # Combine the three single channel color images
    stacked = np.hstack((gray_roi, max_roi, sample))

    # Resize them so they are larger
    stacked = cv2.resize(stacked, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)
    roi_color = cv2.resize(roi_color, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    # Replace the pixel values in the upper right corner with the stacked image
    result[5:stacked.shape[0] + 5, -(5 + stacked.shape[1]):-5, 0] = stacked
    result[5:stacked.shape[0] + 5, -(5 + stacked.shape[1]):-5, 1] = stacked
    result[5:stacked.shape[0] + 5, -(5 + stacked.shape[1]):-5, 2] = stacked

    # Add the color image to the upper right
    result[5:stacked.shape[0] + 5, -(5 + stacked.shape[1] + roi_color.shape[1]):-(5 + stacked.shape[1])] = roi_color

    # Display it all
    cv2.imshow('chart', result)
    cv2.waitKey()
    cv2.destroyAllWindows()


def create_color_histogram(image, bins=8):
    """
    Calculates the Color histogram for an image and returns a 3 element Python array of arrays of size bins
    The size of each array element is [(blues),(greens),(reds)].  Where (color) length = bins count.

    :param image: any size image to perform the histogram on
    :param bins: (default: 8) - the number of separate slots/groups/bins.  8 means 0-7 is bin 1, 8-15 is bin 2, etc
    :return: A Python List of length 3, where each element contains an array of bin values for that color channel.
    """

    colors = []
    # Loop over the three colors (Blue, Green, Red) (OpenCV has this order)
    for i in range(0, 3):
        colors.append(cv2.calcHist([image], [i], None, [bins], [0, 256]))

    return colors


# Save the hog information for a single HOG/Region
# Along with the coordinates, and the band prediction.
def save_hogs(hog_info, region_coords, band, output_file):
    """
    Saves the HOG generated from the single ROI to the file

    :param hog_info: the data from the HOG to be saved
    :param region_coords: The X,Y position of the upper left of the ROI
    :param band: This is the "Y" value or prediction/category/classification
    :param output_file: already opened file to save to
    :return: nothing
    """
    csv_hog_info = ','.join([('%f' % num).rstrip('0').rstrip('.') for num in hog_info])
    csv_region_coords = ','.join(['%d' % num for num in region_coords])

    output_file.write(str(band))
    output_file.write(',')
    output_file.write(str(csv_hog_info))
    output_file.write(',')
    output_file.write(str(csv_region_coords))
    output_file.write('\n')


def load_hogs_csv(directory):
    """
    Retrieves the data from all files in a folder, and returns the data and filenames

    :param directory: A string that represents the path of the folder containing the hog files
    :return: An ndarray containing the all the instances of the hog data, the coordinates, bands
    """
    all_hogs = []

    # Load images from folders in loop
    for filename in os.listdir(directory):
        combined_filename = os.path.join(directory, filename)
        print("Loading:", filename)
        all_hogs.append(np.loadtxt(combined_filename, delimiter=','))

    all_hogs = np.vstack(all_hogs)

    return all_hogs

if __name__ == '__main__':
    mask_filename = r"../mask_generator/gray-mask.png"

    # real mask
    mask = cv2.imread(mask_filename)

    print(__doc__)
    try:
        img_fn = sys.argv[1]
    except:
        img_fn = './baboon.jpg'

    img = cv2.imread(img_fn)
    if img is None:
        print('Failed to load image file:', img_fn)
        sys.exit(1)

    filters = build_filters(16)

    res1_start = time.time()
    res1 = run_gabor(img, filters, mask)
    print("Single Threaded:", time.time() - res1_start)
